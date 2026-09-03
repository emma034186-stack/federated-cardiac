import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from monai.transforms import Compose, RandFlip, RandRotate, RandZoom
from config import IMG_SIZE, NUM_CLASSES


def read_patient_group(patient_dir: str) -> str:
    """Read pathology group from ACDC Info.cfg (e.g. 'DCM', 'NOR', ...)."""
    cfg_path = os.path.join(patient_dir, "Info.cfg")
    with open(cfg_path) as f:
        for line in f:
            if line.startswith("Group:"):
                return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def load_patient_frames(patient_dir: str):
    """
    Load ED and ES frames + ground truth masks for one patient.
    Returns list of (image_2d_slice, label_2d_slice) tuples.
    """
    slices = []
    for suffix in ["_frame01", "_frame12"]:
        patient_id = os.path.basename(patient_dir)
        img_path = os.path.join(patient_dir, f"{patient_id}{suffix}.nii.gz")
        gt_path  = os.path.join(patient_dir, f"{patient_id}{suffix}_gt.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(gt_path):
            continue

        img_nii = nib.load(img_path)
        gt_nii  = nib.load(gt_path)
        img_vol = img_nii.get_fdata()   # (H, W, D)
        gt_vol  = gt_nii.get_fdata()

        # Iterate over slices along z-axis
        for z in range(img_vol.shape[2]):
            img_slice = img_vol[:, :, z].astype(np.float32)
            gt_slice  = gt_vol[:, :, z].astype(np.int64)
            # Skip near-empty slices (less than 1% foreground)
            if (gt_slice > 0).mean() < 0.01:
                continue
            slices.append((img_slice, gt_slice))

    return slices


def scan_acdc(data_dir: str):
    """
    Scan ACDC training directory and return a list of dicts:
    [{"patient_dir": ..., "group": ..., "slices": [(img, gt), ...]}, ...]
    """
    patients = []
    training_dir = os.path.join(data_dir, "training")
    if not os.path.isdir(training_dir):
        raise FileNotFoundError(
            f"ACDC training directory not found at {training_dir}.\n"
            "Please download from https://acdc.creatis.insa-lyon.fr/ and "
            "extract to data/acdc/training/"
        )

    for name in sorted(os.listdir(training_dir)):
        patient_dir = os.path.join(training_dir, name)
        if not os.path.isdir(patient_dir):
            continue
        group = read_patient_group(patient_dir)
        slices = load_patient_frames(patient_dir)
        if slices:
            patients.append({"patient_dir": patient_dir, "group": group, "slices": slices})

    return patients


class ACDCSliceDataset(Dataset):
    """2-D slice-level dataset from a list of (img_array, gt_array) pairs."""

    def __init__(self, slice_pairs, augment=False):
        self.slices = slice_pairs
        self.augment = augment
        self.aug_transform = Compose([
            RandFlip(spatial_axis=1, prob=0.5),
            RandRotate(range_x=0.3, prob=0.5, keep_size=True),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3, keep_size=True),
        ])

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img, gt = self.slices[idx]

        # Resize to fixed size
        img = torch.from_numpy(img).unsqueeze(0)  # (1, H, W)
        gt  = torch.from_numpy(gt.astype(np.float32)).unsqueeze(0)

        img = F.interpolate(
            img.unsqueeze(0), size=IMG_SIZE, mode="bilinear", align_corners=False
        ).squeeze(0)
        gt = F.interpolate(
            gt.unsqueeze(0), size=IMG_SIZE, mode="nearest"
        ).squeeze(0).long().squeeze(0)

        if self.augment:
            # MONAI transforms expect numpy (C, H, W); output is MetaTensor → convert back
            img_np = img.numpy()
            gt_np  = gt.unsqueeze(0).numpy().astype(np.float32)
            img_np = np.array(self.aug_transform(img_np))
            gt_np  = np.array(self.aug_transform(gt_np))
            img = torch.from_numpy(img_np)
            gt  = torch.from_numpy(gt_np).squeeze(0).long()

        # Normalize intensity to [0, 1]
        img_min, img_max = img.min(), img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        return img, gt
