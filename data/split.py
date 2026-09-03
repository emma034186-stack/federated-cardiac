import random
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from data.acdc_dataset import ACDCSliceDataset, scan_acdc
from config import DATA_DIR, BATCH_SIZE, HOSPITAL_GROUPS, NUM_CLIENTS


def _collect_slices(patients):
    """Flatten patient list → list of (img, gt) slices."""
    all_slices = []
    for p in patients:
        all_slices.extend(p["slices"])
    return all_slices


def build_noniid_splits(data_dir=DATA_DIR, val_ratio=0.15, seed=42):
    """
    Non-IID split: each hospital gets patients whose pathology group matches
    HOSPITAL_GROUPS in config.py.
    Returns (client_train_loaders, client_val_loaders, test_loader).
    """
    random.seed(seed)
    patients = scan_acdc(data_dir)

    # Group patients by pathology
    group_map: Dict[str, list] = {}
    for p in patients:
        group_map.setdefault(p["group"], []).append(p)

    # Reserve 10 patients per group for global test set
    test_patients = []
    train_val_patients = []
    for group, plist in group_map.items():
        random.shuffle(plist)
        n_test = max(1, int(len(plist) * 0.1))
        test_patients.extend(plist[:n_test])
        train_val_patients.extend(plist[n_test:])

    # Build per-hospital patient lists
    hospital_patients: Dict[int, list] = {i: [] for i in range(NUM_CLIENTS)}
    for p in train_val_patients:
        for hospital_id, groups in HOSPITAL_GROUPS.items():
            if p["group"] in groups:
                hospital_patients[hospital_id].append(p)
                break

    client_train_loaders, client_val_loaders = [], []
    for i in range(NUM_CLIENTS):
        plist = hospital_patients[i]
        random.shuffle(plist)
        n_val = max(1, int(len(plist) * val_ratio))
        val_slices   = _collect_slices(plist[:n_val])
        train_slices = _collect_slices(plist[n_val:])

        train_ds = ACDCSliceDataset(train_slices, augment=True)
        val_ds   = ACDCSliceDataset(val_slices)

        client_train_loaders.append(
            DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        )
        client_val_loaders.append(
            DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        )

    test_slices = _collect_slices(test_patients)
    test_loader = DataLoader(
        ACDCSliceDataset(test_slices),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    _print_split_stats("Non-IID", hospital_patients, test_patients)
    return client_train_loaders, client_val_loaders, test_loader


def build_iid_splits(data_dir=DATA_DIR, val_ratio=0.15, seed=42):
    """
    IID split: randomly distribute all patients equally across hospitals.
    Returns (client_train_loaders, client_val_loaders, test_loader).
    """
    random.seed(seed)
    patients = scan_acdc(data_dir)
    random.shuffle(patients)

    n_test = max(NUM_CLIENTS, int(len(patients) * 0.1))
    test_patients   = patients[:n_test]
    train_val_pats  = patients[n_test:]

    chunk = len(train_val_pats) // NUM_CLIENTS
    client_train_loaders, client_val_loaders = [], []

    for i in range(NUM_CLIENTS):
        plist = train_val_pats[i * chunk: (i + 1) * chunk]
        n_val = max(1, int(len(plist) * val_ratio))
        val_slices   = _collect_slices(plist[:n_val])
        train_slices = _collect_slices(plist[n_val:])

        train_ds = ACDCSliceDataset(train_slices, augment=True)
        val_ds   = ACDCSliceDataset(val_slices)

        client_train_loaders.append(
            DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        )
        client_val_loaders.append(
            DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        )

    test_slices = _collect_slices(test_patients)
    test_loader = DataLoader(
        ACDCSliceDataset(test_slices),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    hospital_patients = {
        i: train_val_pats[i * chunk: (i + 1) * chunk] for i in range(NUM_CLIENTS)
    }
    _print_split_stats("IID", hospital_patients, test_patients)
    return client_train_loaders, client_val_loaders, test_loader


def build_single_site(data_dir=DATA_DIR, hospital_id=0, val_ratio=0.15, seed=42):
    """
    Return train/val/test loaders using only data from one hospital (hospital_id).
    Used as the single-site baseline.
    """
    train_loaders, val_loaders, test_loader = build_noniid_splits(data_dir, val_ratio, seed)
    return train_loaders[hospital_id], val_loaders[hospital_id], test_loader


def _print_split_stats(label, hospital_patients, test_patients):
    print(f"\n[Split: {label}]")
    for i, plist in hospital_patients.items():
        groups = [p["group"] for p in plist]
        slices = sum(len(p["slices"]) for p in plist)
        from collections import Counter
        print(f"  Hospital {i}: {len(plist)} patients, {slices} slices | {dict(Counter(groups))}")
    test_slices = sum(len(p["slices"]) for p in test_patients)
    print(f"  Test set  : {len(test_patients)} patients, {test_slices} slices\n")
