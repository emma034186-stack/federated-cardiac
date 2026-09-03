import torch
import torch.nn as nn
import numpy as np
import flwr as fl
from collections import OrderedDict
from models.unet import build_model
from utils.trainer import train_one_epoch, evaluate
from utils.metrics import dice_score
from config import DEVICE, LR, LOCAL_EPOCHS


def get_parameters(model):
    # 全部轉 float32 numpy 送給 Flower（含 BN running stats）
    return [val.cpu().float().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    original_sd = model.state_dict()
    keys = list(original_sd.keys())
    state_dict = OrderedDict()
    for k, v in zip(keys, parameters):
        # 還原原始 dtype（num_batches_tracked 是 int64，不能讓它變 float）
        tensor = torch.from_numpy(v).to(dtype=original_sd[k].dtype, device=DEVICE)
        state_dict[k] = tensor
    model.load_state_dict(state_dict, strict=True)


class CardiacClient(fl.client.NumPyClient):
    """
    Flower client that wraps local U-Net training on one hospital's data.
    Implements the standard fit / evaluate interface for FedAvg.
    """

    def __init__(self, client_id: int, train_loader, val_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = build_model().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()

        for _ in range(LOCAL_EPOCHS):
            train_one_epoch(
                self.model, self.train_loader,
                self.optimizer, self.criterion, DEVICE
            )

        n_samples = len(self.train_loader.dataset)
        return get_parameters(self.model), n_samples, {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        dice = evaluate(self.model, self.val_loader, DEVICE)
        n_samples = len(self.val_loader.dataset)
        # Flower expects (loss, n_samples, metrics)
        loss = 1.0 - dice
        return float(loss), n_samples, {"dice": float(dice)}


def make_client_fn(train_loaders, val_loaders):
    """
    Returns a client_fn compatible with fl.simulation.start_simulation().
    client_fn(cid: str) -> fl.client.Client
    """
    def client_fn(cid: str) -> fl.client.Client:
        cid = int(cid)
        return CardiacClient(cid, train_loaders[cid], val_loaders[cid]).to_client()

    return client_fn
