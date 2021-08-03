from typing import List

import dgl
import dgl.function as fn
import torch as th
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

from .utils import exists

# --------------------
# Basic layers
# --------------------


class ReZero(nn.Module):
    """
    Classic residual conn:
        x = x + F(x)
    ReZero usage:
        x = x + a(F(x))
        where a is a scalar initialized to 0. This gates the "wet" signal allowing
        more efficient training for deeper nets.
    """

    def __init__(self):
        super(ReZero, self).__init__()
        self.g = nn.Parameter(th.zeros(1))

    def forward(self, x):
        return x * self.g


class MultiLabelEmbed(nn.Module):
    """
    Multi-label embedder which returns the sum of the embedding vector for each label
    element.
    """

    def __init__(self, label_counts: List[int], output_dim: int, normalize=True):
        super(MultiLabelEmbed, self).__init__()
        self.embedders = nn.ModuleList(
            [nn.Embedding(n, output_dim) for n in label_counts]
        )
        self.normalize = normalize
        self.label_counts = label_counts

    def forward(self, x: th.LongTensor):
        """
        :param x: Multi-label encoded long tensor, i.e. [1, 3, 4, 9, 0]
        :return: The final dimension is replaced by the sum of embeddings of each label.
        """
        if x.dim() == 1:
            x = x.unsqueeze(1)
        out = 0
        for i in range(x.shape[1]):
            out += self.embedders[i](x[:, i])
        if self.normalize:
            # scale by n^{-1/2} where n is number of embeddings
            out *= len(self.label_counts) ** -0.5
        return out


# --------------------
# Normalization
# --------------------


class GraphNorm(nn.Module):
    """
    Allows selection between batch, graph, and layer norm for node level features.
    """

    def __init__(self, norm_type, hidden_dim=300, print_info=None):
        super(GraphNorm, self).__init__()
        assert norm_type in ["bn", "gn", "ln", None]
        self.norm = None
        self.print_info = print_info
        if norm_type == "bn":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "ln":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "gn":
            self.norm = norm_type
            self.gnw = nn.Parameter(th.ones(hidden_dim))
            self.gnb = nn.Parameter(th.zeros(hidden_dim))
            self.msc = nn.Parameter(th.ones(hidden_dim))

    def forward(self, x, batch_num):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(x)
        elif self.norm is None:
            return x
        else:
            batch_size = len(batch_num)
            batch_list = th.as_tensor(batch_num, dtype=th.long, device=x.device)
            batch_index = th.arange(batch_size, device=x.device).repeat_interleave(
                batch_list
            )
            batch_index = batch_index.view((-1,) + (1,) * (x.dim() - 1)).expand_as(x)
            mean = th.zeros(batch_size, *x.shape[1:], device=x.device)
            mean = mean.scatter_add_(0, batch_index, x)
            mean = (mean.T / batch_list).T
            mean = mean.repeat_interleave(batch_list, dim=0)
            sub = x - mean * self.msc
            std = th.zeros(batch_size, *x.shape[1:], device=x.device)
            std = std.scatter_add_(0, batch_index, sub.pow(2))
            std = ((std.T / batch_list).T + 1e-6).sqrt()
            std = std.repeat_interleave(batch_list, dim=0)
            # return sub / std
            return self.gnw * sub / std + self.gnb


# --------------
# Graph Conv Layers
# --------------


class GatedGCNLayer(nn.Module):
    """
    ResGatedGCN: Residual Gated Graph ConvNets
    An Experimental Study of Neural Networks for Variable Graphs (Xavier Bresson and
    Thomas Laurent, ICLR 2018) https://arxiv.org/pdf/1711.07553v2.pdf
    """

    def __init__(self, dim, dropout=0.1, act_fn=nn.ReLU, rezero=True):
        super().__init__()
        self.lin_node = nn.Linear(dim, dim * 4, bias=True)
        self.lin_edge = nn.Linear(dim, dim, bias=True)
        self.bn_node = nn.BatchNorm1d(dim)
        self.bn_edge = nn.BatchNorm1d(dim)
        self.dropout = dropout
        self.activation = act_fn()
        self.rezero = ReZero() if rezero else nn.Identity()
        self.dim = dim

    def forward(self, g: dgl.DGLGraph, h: th.tensor, e: th.tensor):
        with g.local_scope():
            # residual connections
            h_in = h
            e_in = e

            # Norm
            h = self.bn_node(h)
            e = self.bn_edge(e)

            # Dropout
            h = F.dropout(h, self.dropout, training=self.training)
            e = F.dropout(e, self.dropout, training=self.training)

            # Conv
            H = self.lin_node(h)
            g.ndata["Ah"] = H[:, : self.dim]  # for gating
            g.ndata["Bh"] = H[:, self.dim : self.dim * 2]  # for gating
            g.ndata["Dh"] = H[:, self.dim * 2 : self.dim * 3]  # src to edge
            g.ndata["Eh"] = H[:, self.dim * 3 : self.dim * 4]  # dst to edge
            g.edata["Ce"] = self.lin_edge(e)
            g.apply_edges(fn.u_add_v("Dh", "Eh", "DEh"))
            g.edata["e"] = g.edata["DEh"] + g.edata["Ce"]
            g.edata["sigma"] = th.sigmoid(g.edata["e"])  # edge gates
            g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
            g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
            g.ndata["h"] = g.ndata["Ah"] + g.ndata["sum_sigma_h"] / (
                g.ndata["sum_sigma"] + 1e-6
            )
            h = g.ndata["h"]
            e = g.edata["e"]

            # Activate
            h = self.activation(h)
            e = self.activation(e)

            # ReZero (residual)
            h = h_in + self.rezero(h)
            e = e_in + self.rezero(e)

        return h, e


class GatedGCNLayer_v2(nn.Module):
    """
    Gated Graph Convolution layer with a dense attention mechanism from "Benchmarking Graph Neural Networks" (https://arxiv.org/abs/2003.00982).
    Implemented within a Transformer block backbone.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        dropout=0.0,
        batch_norm=True,
        norm_type="gn",
        residual=True,
        rezero=True,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.rezero = ReZero() if rezero else nn.Identity()

        if input_dim != output_dim:
            self.residual = False

        # Linear transformations for dense attention mechanism
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)

        # MLPs for node and edge features
        self.ff_h = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )
        self.ff_e = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True),
        )

        if batch_norm == True:
            self.norm1_h = GraphNorm(norm_type, output_dim)
            self.norm1_e = GraphNorm(norm_type, output_dim)
            self.norm2_h = GraphNorm(norm_type, output_dim)
            self.norm2_e = GraphNorm(norm_type, output_dim)

    def forward(self, g, h, e):

        ########## Message-passing sub-layer ##########

        h_in = h  # for residual connection
        e_in = e  # for residual connection

        if self.batch_norm == True:
            h = self.norm1_h(h, g.batch_num_nodes())  # graph normalization
            e = self.norm1_e(e, g.batch_num_edges())  # graph normalization

        # Linear transformations of nodes and edges
        g.ndata["h"] = h
        g.edata["e"] = e
        g.ndata["Ah"] = self.A(h)  # node update, self-connection
        g.ndata["Bh"] = self.B(h)  # node update, neighbor projection
        g.ndata["Ch"] = self.C(h)  # edge update, source node projection
        g.ndata["Dh"] = self.D(h)  # edge update, destination node projection
        g.edata["Ee"] = self.E(e)  # edge update, edge projection

        # Graph convolution with dense attention mechanism
        g.apply_edges(fn.u_add_v("Ch", "Dh", "CDh"))
        g.edata["e"] = g.edata["CDh"] + g.edata["Ee"]
        # Dense attention mechanism
        g.edata["sigma"] = th.sigmoid(g.edata["e"])
        g.update_all(fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h"))
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        # Gated-Mean aggregation
        g.ndata["h"] = g.ndata["Ah"] + g.ndata["sum_sigma_h"] / (
            g.ndata["sum_sigma"] + 1e-10
        )
        h = g.ndata["h"]  # result of graph convolution
        e = g.edata["e"]  # result of graph convolution

        if self.residual == True:
            h = h_in + self.rezero(h)  # residual connection
            e = e_in + self.rezero(e)  # residual connection

        ############ Feedforward sub-layer ############

        h_in = h  # for residual connection
        e_in = e  # for residual connection

        if self.batch_norm == True:
            h = self.norm2_h(h, g.batch_num_nodes())  # graph normalization
            e = self.norm2_e(e, g.batch_num_edges())  # graph normalization

        # MLPs on updated node and edge features
        h = self.ff_h(h)
        e = self.ff_e(e)

        if self.residual == True:
            h = h_in + self.rezero(h)  # residual connection
            e = e_in + self.rezero(e)  # residual connection

        return h, e


# --------------------
# Readout
# --------------------


class DeepSetsNodes(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super(DeepSetsNodes, self).__init__()
        self.glu = nn.Sequential(nn.Linear(d_in, d_in * 2), nn.GLU())
        self.agg = nn.Sequential(
            nn.BatchNorm1d(d_in), nn.Dropout(dropout), nn.Linear(d_in, d_out)
        )

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.ndata["n"] = self.glu(feat)
            readout = self.agg(dgl.sum_nodes(graph, "n"))
            return readout


class DeepSetsEdges(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super(DeepSetsEdges, self).__init__()
        self.glu = nn.Sequential(nn.Linear(d_in, d_in * 2), nn.GLU())
        self.agg = nn.Sequential(
            nn.BatchNorm1d(d_in), nn.Dropout(dropout), nn.Linear(d_in, d_out)
        )

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata["e"] = self.glu(feat)
            readout = self.agg(dgl.sum_edges(graph, "e"))
            return readout


class MeanMaxPool(nn.Module):
    """mean | max -> linear = graph latent."""

    def __init__(self, dim):
        super(MeanMaxPool, self).__init__()
        self.lin = nn.Linear(dim * 2, dim)

    def forward(self, g, n, key="to_meanmax"):
        g.ndata[key] = n
        max = dgl.readout_nodes(g, key, op="max")
        mean = dgl.readout_nodes(g, key, op="mean")
        out = th.cat([max, mean], dim=-1)
        return self.lin(out)


class SumPoolingEdges(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata["e"] = feat
            readout = dgl.sum_edges(graph, "e")
            return readout


class AvgPoolingEdges(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata["e"] = feat
            readout = dgl.mean_edges(graph, "e")
            return readout


class MaxPoolingEdges(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata["e"] = feat
            readout = dgl.max_edges(graph, "e")
            return readout


class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y


# --------------------
# Activation
# --------------------


class GRU(nn.Module):
    """
        Wrapper class for the GRU used by the GNN framework, nn.GRU is used for the Gated Recurrent Unit itself
    """

    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def forward(self, x, y):
        """
        :param x:   shape: (B, N, Din) where Din <= input_size (difference is padded)
        :param y:   shape: (B, N, Dh) where Dh <= hidden_size (difference is padded)
        :return:    shape: (B, N, Dh)
        """
        assert (x.shape[-1] <= self.input_size and y.shape[-1] <= self.hidden_size)
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        x = self.gru(x, y)[1]
        x = x.squeeze()
        return x


class LowRankGlobalAttention(nn.Module):
    """
    LRGA adapted from https://github.com/omri1348/LRGA
    LRGA(X) = [1/η(X) U (V_T Z), T]
    where U = m1(X)
          V = m2(X)
          Z = m3(X)
          T = m4(X)
    and η = normalization factor using U and V_T
    The output of LRGA (global) is concatenated with X and GNN output (local) to form
    vertex features of next layer.
    for i in layers:
        x = dimension_reduce(global | local | x)
    """

    def __init__(self, d=256, k=32, dropout=0.1):
        """
        :param d: input features
        :param k: attention rank (dimension of inner projections)
        :param dropout: output dropout probability
        """
        super().__init__()
        self.w = nn.Sequential(nn.Linear(d, 4 * k), nn.ReLU())
        self.activation = nn.ReLU()
        self.apply(self.weight_init)
        self.k = k
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        X = self.w(x)
        U = X[:, : self.k]
        V = X[:, self.k : 2 * self.k]
        Z = X[:, 2 * self.k : 3 * self.k]
        T = X[:, 3 * self.k :]
        Vt = V.t
        D = self.joint_normalize2(U, Vt)
        res = th.mm(U, th.mm(Vt, Z))
        res = th.cat((res * D, T), dim=1)
        return self.dropout(res)

    @staticmethod
    def weight_init(layer):
        if type(layer) == nn.Linear:
            nn.init.xavier_normal_(layer.weight.data)
            if layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0)

    @staticmethod
    def joint_normalize2(U, Vt):
        # U and V_T are in block diagonal form
        tmp_ones = th.ones((Vt.shape[1], 1), device=Vt.device)
        norm_factor = th.mm(U, th.mm(Vt, tmp_ones))
        norm_factor = (th.sum(norm_factor) / U.shape[0]) + 1e-6
        return 1 / norm_factor


# --------------------
# MLP
# --------------------


SUPPORTED_ACTIVATION_MAP = {'ReLU', 'Sigmoid', 'Tanh', 'ELU', 'SELU', 'GLU', 'LeakyReLU', 'Softplus', 'None'}


def get_activation(activation):
    """ returns the activation function represented by the input string """
    if activation and callable(activation):
        # activation is already a function
        return activation
    # search in SUPPORTED_ACTIVATION_MAP a torch.nn.modules.activation
    activation = [x for x in SUPPORTED_ACTIVATION_MAP if activation.lower() == x.lower()]
    assert len(activation) == 1 and isinstance(activation[0], str), 'Unhandled activation function'
    activation = activation[0]
    if activation.lower() == 'none':
        return None
    return vars(th.nn.modules.activation)[activation]()


class FCLayer(nn.Module):

    def __init__(self, input_dim, output_dim, activation='relu', dropout=0., b_norm=False, bias=True, init_fn=None,
                 ):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = input_dim
        self.out_size = output_dim
        self.bias = bias
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(output_dim)
        self.activation = get_activation(activation)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'


class MLP(nn.Module):
    """
        Simple multi-layer perceptron, built of a series of FCLayers
    """

    def __init__(self, emb_dim, hidden_dim, out_dim, layers, mid_activation='relu', last_activation='none',
                 dropout=0., mid_b_norm=False, last_b_norm=False):
        super(MLP, self).__init__()

        self.in_size = emb_dim
        self.hidden_size = hidden_dim
        self.out_size = out_dim

        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(FCLayer(emb_dim, out_dim, activation=last_activation, b_norm=last_b_norm,
                                                dropout=dropout))
        else:
            self.fully_connected.append(FCLayer(emb_dim, hidden_dim, activation=mid_activation, b_norm=mid_b_norm,
                                                dropout=dropout))
            for _ in range(layers - 2):
                self.fully_connected.append(FCLayer(hidden_dim, hidden_dim, activation=mid_activation,
                                                    b_norm=mid_b_norm, dropout=dropout))
            self.fully_connected.append(FCLayer(hidden_dim, out_dim, activation=last_activation, b_norm=last_b_norm,
                                                dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_size) + ' -> ' \
               + str(self.out_size) + ')'