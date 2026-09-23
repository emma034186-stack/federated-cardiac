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
import torch
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
    parser.add_argument(
        "--backend",
        choices=["ray", "inprocess"],
        default="ray",
        help="ray: Flower's start_simulation (Ray actors). inprocess: run the same "
             "clients and the same strategy sequentially in one process, for machines "
             "where Ray's shared-memory object store fails (e.g. Windows with a small page file).",
    )
    return parser.parse_args()


def run_inprocess(strategy, train_loaders, val_loaders):
    """
    Sequential FedAvg driver that reuses the Flower strategy (aggregate_fit /
    evaluate, including the noise wrapper) and CardiacClient unchanged. A fresh
    client is built every round, as Flower's simulation does via client_fn, so
    no optimizer state leaks across rounds.
    """
    from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
    from fl.client import CardiacClient

    parameters = strategy.initialize_parameters(client_manager=None)
    strategy.evaluate(0, parameters)
    for server_round in range(1, NUM_ROUNDS + 1):
        global_nd = parameters_to_ndarrays(parameters)
        results = []
        for cid in range(NUM_CLIENTS):
            client = CardiacClient(cid, train_loaders[cid], val_loaders[cid])
            nd, n_examples, metrics = client.fit(global_nd, {})
            results.append((None, FitRes(status=Status(code=Code.OK, message=""),
                                         parameters=ndarrays_to_parameters(nd),
                                         num_examples=n_examples, metrics=metrics)))
        parameters, _ = strategy.aggregate_fit(server_round, results, [])
        strategy.evaluate(server_round, parameters)


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

    if args.backend == "inprocess":
        run_inprocess(strategy, train_loaders, val_loaders)
    else:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
            strategy=strategy,
            # Ray hides GPUs from actors that request none, so give each client a
            # share of the GPU when CUDA is available (3 clients fit on one card).
            client_resources={"num_cpus": 1, "num_gpus": 0.3 if torch.cuda.is_available() else 0.0},
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
