import os
import torch

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "acdc")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Model
NUM_CLASSES = 4          # background, RV, myocardium, LV
IN_CHANNELS = 1
IMG_SIZE = (256, 256)

# Training
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 8
LR = 1e-3
LOCAL_EPOCHS = 5
NUM_ROUNDS = 20
NUM_CLIENTS = 3

# Federated
MIN_FIT_CLIENTS = 3
FRACTION_FIT = 1.0

# Differential Privacy
DP_NOISE_MULTIPLIER = 0.1
DP_CLIPPING_NORM = 5.0
DP_NOISE_SCALE = 0.005   # per-element Gaussian noise added after FedAvg aggregation

# ACDC pathology groups per hospital (Non-IID split)
# Hospital A: mostly normal & dilated — simulates a general cardiology center
# Hospital B: mostly hypertrophic — simulates a specialized HCM clinic
# Hospital C: mostly infarction & RV — simulates a post-MI/RV center
HOSPITAL_GROUPS = {
    0: ["NOR", "DCM"],
    1: ["HCM", "DCM"],
    2: ["MINF", "RV"],
}

# IID split uses random equal partition across all 3 clients
IID_SPLIT = True   # set False for Non-IID

# Experiment names (used for saving results)
EXPERIMENTS = ["single_site", "fedavg_iid", "fedavg_noniid", "fedavg_noniid_dp"]
