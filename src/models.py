import pytorch_lightning as pl
from adabelief_pytorch import AdaBelief
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from ogb.lsc.pcqm4m import PCQM4MEvaluator
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from .nn import *
from .utils import *
from .attack import *

__all__ = ["GatedGCNNet"]


class BaseModule(pl.LightningModule):
    """
    Base LightningModule with optimization, training, validation loop. Needs network
    and forward method.
    """

    def __init__(
        self,
        lr=0.001,
        eps=1e-16,
        weight_decay=0,
        max_epochs=100,
        batch_size=4096,
        opt="AdaBelief",
        **kwargs
    ):
        super(BaseModule, self).__init__()
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.evaluator = PCQM4MEvaluator()
        self.opt = opt

    def forward(self, batch):
        return NotImplementedError

    def training_step(self, batch, *args, **kwargs):
        y = batch["y"]
        yhat = self(batch)
        loss = F.l1_loss(yhat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        y = batch["y"]
        yhat = self(batch)
        return {"yhat": yhat, "y": y}

    def validation_epoch_end(self, outputs):
        """
        Gather and evaluate validation step outputs with OGB Evaluator.
        """
        yhats = []
        ys = []
        for o in outputs:
            yhats.append(o["yhat"])
            ys.append(o["y"])
        yhat = th.cat(yhats)
        y = th.cat(ys)

        res = self.evaluator.eval({"y_true": y.squeeze(), "y_pred": yhat.squeeze()})
        self.log("val_mae", res["mae"])

    def configure_optimizers(self):
        opt = AdaBelief(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=self.eps,
            print_change_log=False,
        )

        sched = {
            "scheduler": th.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lambda epoch: max(1e-7, 1 - epoch / self.max_epochs)
            ),
            "reduce_on_plateau": False,
            "interval": "epoch",
            "frequency": 1,
        }
        return [opt], [sched]

    def on_train_start(self) -> None:
        self.logger.watch(self, log="all")


class GatedGCNNet(BaseModule):
    def __init__(
        self,
        node_feats=get_atom_feature_dims(),
        edge_feats=get_bond_feature_dims(),
        pos_feats=8,
        # model params
        dim=256,
        n_layers=7,
        dropout=0.2,
        act="ReLU",
        pool_type="deepset",
        flag="False",
        # learning params
        **kwargs
    ):
        super(GatedGCNNet, self).__init__(**kwargs)
        self.save_hyperparameters()
        act_fn = {"GELU": nn.GELU, "ReLU": nn.ReLU, "leaky_ReLU": nn.LeakyReLU}[act]
        self.atom_embed = MultiLabelEmbed(node_feats, dim)
        self.bond_embed = MultiLabelEmbed(edge_feats, dim)
        self.pos_embed = nn.Linear(pos_feats, dim) if pos_feats is not None else None
        self.gn_blocks = nn.ModuleList(
            GatedGCNLayer(dim, dropout, act_fn) for _ in range(n_layers)
        )
        if pool_type == "deepset":
            self.readout = DeepSets(dim, dim, dropout=dropout)
        elif pool_type == "mean_max":
            self.readout = MeanMaxPool(dim)
        elif pool_type == "mean":
            self.readout = MeanPool()
        self.homolumo = nn.Sequential(nn.Linear(dim, 1), act_fn())

    def forward(self, batch, n_perturb=None):
        g, y = pluck(batch, "graph", "y")
        n = (
            self.atom_embed(g.ndata["feat"])
            if n_perturb is None
            else self.atom_embed(g.ndata["feat"]) + n_perturb
        )
        if self.pos_embed is not None:
            n = n + self.pos_embed(self.flip_signs(g.ndata["pos_enc"]))
        e = self.bond_embed(g.edata["feat"])
        for block in self.gn_blocks:
            n, e = block(g, n, e)
        G = self.readout(g, n)
        y_hat = self.homolumo(G)
        return y_hat

    def training_step(self, batch, *args, **kwargs):
        y = batch["y"]

        loss_fn = nn.L1Loss()

        if self.hparams.flag == "True":
            fwd = lambda n_perturb: self(batch, n_perturb).to(th.float32)
            y = y.to(th.float32)
            n_perturb_shape = (batch["graph"].num_nodes(), self.hparams.dim)
            loss, losses = flag(fwd, n_perturb_shape, y, 0.001, 3, y.device, loss_fn)
            self.log("train_loss", sum(losses))
        else:
            yhat = self(batch)
            loss = loss_fn(yhat, y)
            self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        yhat = th.clamp(self(batch), min=0.0, max=50.0)
        return {"yhat": yhat, "y": batch["y"]}

    def flip_signs(self, x):
        sign_flip = th.rand(8, device=self.device)
        sign_flip[sign_flip >= 0.5] = 1.0
        sign_flip[sign_flip < 0.5] = -1.0
        return x * sign_flip.unsqueeze(0)


class GatedGCNNet_v2(BaseModule):
    def __init__(
        self,
        node_feats=get_atom_feature_dims(),
        edge_feats=get_bond_feature_dims(),
        pos_enc_dim=8,
        emb_dim=128,
        n_layers=12,
        dropout=0.5,
        batch_norm=True,
        norm_type="gn",
        residual=True,
        pool_type="mean",
        virtualnode=False,
        # learning params
        **kwargs
    ):
        super().__init__()

        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.norm_type = norm_type
        self.residual = residual
        self.pos_enc_dim = pos_enc_dim
        self.pool_type = pool_type
        self.virtualnode = virtualnode

        hidden_dim = emb_dim * 4
        self.hidden_dim = hidden_dim

        self.atom_embed = MultiLabelEmbed(node_feats, emb_dim)
        self.bond_embed = MultiLabelEmbed(edge_feats, emb_dim)

        if self.pos_enc_dim > 0:
            self.pos_encoder_h = nn.Sequential(
                nn.Linear(emb_dim + pos_enc_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim, bias=True),
            )

        self.layers = nn.ModuleList(
            [
                GatedGCNLayer_v2(
                    input_dim=emb_dim,
                    output_dim=emb_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    norm_type=norm_type,
                    residual=residual,
                )
                for _ in range(n_layers)
            ]
        )

        if self.virtualnode:
            self.virtualnode_emb = th.nn.Embedding(1, emb_dim)
            th.nn.init.constant_(self.virtualnode_emb.weight.data, 0)
            self.virtualnode_ff = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.BatchNorm1d(emb_dim),
                        nn.Linear(emb_dim, hidden_dim, bias=True),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, emb_dim, bias=True),
                    )
                    for _ in range(n_layers - 1)
                ]
            )

        self.pooling_h = {
            "mean": AvgPooling(),
            "sum": SumPooling(),
            "max": MaxPooling(),
            "deepset": DeepSetsNodes(emb_dim, emb_dim),
        }.get(pool_type, AvgPooling())

        self.pooling_e = {
            "mean": AvgPoolingEdges(),
            "sum": SumPoolingEdges(),
            "max": MaxPoolingEdges(),
            "deepset": DeepSetsEdges(emb_dim, emb_dim),
        }.get(pool_type, AvgPoolingEdges())

        self.homo_lumo = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1, bias=True),
        )

    def forward(self, batch):
        g = batch["graph"].to(self.device)
        h = self.atom_embed(g.ndata["feat"])
        e = self.bond_embed(g.edata["feat"])

        if self.pos_enc_dim > 0:
            pe_h = g.ndata["pos_enc"]
            if self.training:
                sign_flip = th.randint(
                    low=0, high=2, size=(1, pe_h.size(1)), device=pe_h.device
                )
                sign_flip[sign_flip == 0.0] = -1
                pe_h = pe_h * sign_flip
            h = self.pos_encoder_h(th.cat((h, pe_h), dim=-1))

        if self.virtualnode:
            # Initialize virtual node
            vn_h = self.virtualnode_emb(th.zeros(g.batch_size).long().to(h.device))
            batch_list = g.batch_num_nodes().long().to(h.device)
            batch_index = (
                th.arange(g.batch_size)
                .long()
                .to(h.device)
                .repeat_interleave(batch_list)
            )

        for layer_idx in range(self.n_layers):
            if self.virtualnode:
                h = h + vn_h[batch_index]

            h, e = self.layers[layer_idx](g, h, e)

            if self.virtualnode and layer_idx < self.n_layers - 1:
                vn_h = vn_h + self.pooling_h(g, h)
                vn_h = self.virtualnode_ff[layer_idx](vn_h)

        g.ndata["h"] = h
        g.edata["e"] = e

        hg = th.cat((self.pooling_h(g, h), self.pooling_e(g, e)), dim=-1)

        return self.homo_lumo(hg)

    def training_step(self, batch, *args, **kwargs):
        g, y = pluck(batch, "graph", "y")
        yhat = self(batch)

        loss = F.l1_loss(yhat, y)
        self.log_dict({"train_loss": loss})
        return loss

    def validation_step(self, batch, *args, **kwargs):
        yhat = th.clamp(self(batch), min=0.0, max=50.0)
        return {"yhat": yhat, "y": batch["y"]}
