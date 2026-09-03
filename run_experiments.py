"""
One-shot script to run all four experiments sequentially and produce comparison plots.

Usage:
    python run_experiments.py

Experiments:
  1. single_site  - Hospital A only
  2. fedavg_iid   - FedAvg, IID split
  3. fedavg_noniid     - FedAvg, Non-IID split
  4. fedavg_noniid_dp  - FedAvg + DP, Non-IID split
"""
import os
import subprocess
import sys

EXPERIMENTS = [
    ("Single-site baseline",         ["python", "train_single_site.py"]),
    ("FedAvg IID",                   ["python", "simulate.py", "--mode", "iid"]),
    ("FedAvg Non-IID",               ["python", "simulate.py", "--mode", "noniid"]),
    ("FedAvg Non-IID + DP",          ["python", "simulate.py", "--mode", "noniid_dp"]),
]


def main():
    for name, cmd in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"  Running: {name}")
        print(f"{'='*60}")
        result = subprocess.run(cmd, check=True)
        if result.returncode != 0:
            print(f"[Error] {name} failed.")
            sys.exit(1)

    print("\n\nAll experiments done. Generating plots...")
    subprocess.run(["python", "utils/visualization.py"], check=True)
    print("Plots saved to results/")


if __name__ == "__main__":
    main()
