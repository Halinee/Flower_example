from collections import OrderedDict

import flwr as fl
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk

from src.models import GatedGCNNet_v2
from src.utils import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Load model
    net = GatedGCNNet_v2().to(DEVICE)
    # Load data
    trainloader, testloader = load_data()

    # Flower client
    class GraphRepClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=1)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss = test(net, testloader)
            return float(loss), len(testloader), {"mae": float(loss)}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=GraphRepClient())


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(batch), batch["y"].to(DEVICE))
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.L1Loss()
    net.eval()

    loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            yhat = net(batch)
            y = batch["y"].to(DEVICE)
            loss += criterion(yhat, y).item()

        loss /= len(testloader)
        return loss


def dict_collate(batch):
    """
    Collate special handling graph key in batch dict with dgl.batch
    """
    graphs = dgl.batch([x.pop("graph") for x in batch])
    default = {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    return {"graph": graphs, **default}


def load_data():
    """Load PCQM4M Dataset (Training and Valid set)"""
    dataset = load_from_disk("/raid/data/cache/smiles2graph_pos_enc")
    dataset.set_transform(
        lambda batch: {
            "graph": [to_dgl(g) for g in batch["graph"]],
            "y": torch.as_tensor([[y] for y in batch["homolumogap"]]),
        },
        columns=["graph", "homolumogap"],
    )

    train_loader = DataLoader(
        dataset["train"].shuffle().select(range(len(dataset["train"]) // 10)),
        batch_size=512,
        shuffle=True,
        collate_fn=dict_collate,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=3,
    )
    test_loader = DataLoader(
        dataset["valid"],
        batch_size=512,
        shuffle=True,
        collate_fn=dict_collate,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=3,
    )

    return train_loader, test_loader


if __name__ == "__main__":
    main()
