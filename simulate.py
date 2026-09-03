"""
Federated learning simulation using Flower's start_simulation().
Supports three modes:
  --mode iid       : FedAvg with IID data split
  --mode noniid    : FedAvg with Non-IID (pathology-based) split
  --mode noniid_dp : FedAvg + Differential Privacy with Non-IID split
"""
import os
import json
import argparse
import flwr as fl

from fl.client import make_client_fn
from fl.server import build_strategy
from data.split import build_iid_splits, build_noniid_splits
from config import NUM_CLIENTS, NUM_ROUNDS, RESULTS_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["iid", "noniid", "noniid_dp"],
        default="noniid",
        help="Federated split mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n=== Federated Learning Simulation | mode={args.mode} ===")

    use_dp = args.mode == "noniid_dp"
    if args.mode == "iid":
        train_loaders, val_loaders, test_loader = build_iid_splits()
    else:
        train_loaders, val_loaders, test_loader = build_noniid_splits()

    history = []   # filled by server-side evaluate_fn each round
    strategy = build_strategy(test_loader=test_loader, use_dp=use_dp, history=history)
    client_fn = make_client_fn(train_loaders, val_loaders)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"object_store_memory": 1 * 1024 ** 3},
    )

    out_path = os.path.join(RESULTS_DIR, f"fedavg_{args.mode}_history.json")
    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved {out_path}")

    if history:
        best = max(history, key=lambda x: x["dice"])
        print(f"Best Global Dice: {best['dice']:.4f} at round {best['round']}")


if __name__ == "__main__":
    main()
