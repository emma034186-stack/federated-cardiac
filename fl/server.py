import torch
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from models.unet import build_model
from utils.trainer import evaluate as eval_model
from utils.metrics import dice_per_class
from config import (
    DEVICE, NUM_CLIENTS, NUM_ROUNDS, FRACTION_FIT, MIN_FIT_CLIENTS,
    DP_NOISE_MULTIPLIER, DP_CLIPPING_NORM, DP_NOISE_SCALE,
)


def build_strategy(test_loader=None, use_dp=False, history=None):
    """
    Build Flower strategy.
    - use_dp: wrap FedAvg with server-side differential privacy
    - history: list to append per-round metrics for plotting
    """
    init_model = build_model()
    init_params = ndarrays_to_parameters(
        [val.cpu().float().numpy() for _, val in init_model.state_dict().items()]
    )

    def evaluate_fn(server_round, parameters, config):
        """Called by the server after each aggregation round."""
        if test_loader is None:
            return None

        model = build_model().to(DEVICE)
        original_sd = model.state_dict()
        keys = list(original_sd.keys())
        from collections import OrderedDict
        state_dict = OrderedDict()
        for k, v in zip(keys, parameters):
            tensor = torch.from_numpy(v).to(dtype=original_sd[k].dtype, device=DEVICE)
            state_dict[k] = tensor
        model.load_state_dict(state_dict, strict=True)

        dice = eval_model(model, test_loader, DEVICE)
        loss = 1.0 - dice
        metrics = {"dice": dice}

        if history is not None:
            history.append({"round": server_round, "dice": dice})

        print(f"  [Round {server_round:02d}] Global Dice: {dice:.4f}")
        return float(loss), metrics

    strategy = FedAvg(
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_FIT_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
    )

    if use_dp:
        base_strategy = strategy

        class _DPFedAvg(type(base_strategy)):
            def aggregate_fit(self, server_round, results, failures):
                aggregated = super().aggregate_fit(server_round, results, failures)
                if aggregated is None:
                    return None
                params, metrics = aggregated
                ndarrays = parameters_to_ndarrays(params)
                noisy = [
                    arr + np.random.normal(0, DP_NOISE_SCALE, arr.shape).astype(arr.dtype)
                    for arr in ndarrays
                ]
                print(f"  [DP] noise_scale={DP_NOISE_SCALE} applied at round {server_round}")
                return ndarrays_to_parameters(noisy), metrics

        strategy.__class__ = _DPFedAvg

    return strategy
