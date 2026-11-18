"""
Fair Medical Image Segmentation Utilities
=========================================
Core utility classes and functions for:
1. Multi-vendor, multi-demographic medical image data simulation.
2. In-processing fairness optimization (Group DRO and Dynamic Inverse-Dice Reweighting).
3. Sample-wise Dice + Cross-Entropy composite loss.
4. Comprehensive multi-metric computation (Dice, IoU, HD95, Sensitivity, Specificity).
5. Plotting and visual auditing routines.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset, Sampler
from monai.losses import DiceLoss
from monai.metrics import compute_hausdorff_distance


def generate_medical_slice(
    seed: int,
    vendor: str = "Siemens",
    age: int = 55,
    sex: str = "Female",
    size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates a 2D cardiac/abdominal MRI cross-sectional slice with realistic
    anatomical structures, demographic variations, and scanner-vendor artifacts.

    Demographic & Hardware Physics:
    - Target: Myocardial/Organ muscular ring (Foreground Class 1).
    - Cavity / Blood pool: High-signal inner core.
    - Parenchyma / Background: Low-signal background tissue.
    - Sex: Modulates anatomical volume (male ~10% larger cavity radius).
    - Age: Modulates wall thickness (older age >70 exhibits thinner myocardium).
    - Scanner Vendor:
        * Siemens: High SNR (sigma=0.02), sharp anatomical borders.
        * Philips: Moderate SNR (sigma=0.035), spatial polynomial bias field.
        * GE: Lower contrast, Gaussian blur, Rician noise (sigma=0.075), mimicking older field strength.

    Returns:
        img: np.ndarray of shape (1, size, size) normalized to [0, 1]
        mask: np.ndarray of shape (1, size, size) with values in {0.0, 1.0}
    """
    rng = np.random.RandomState(seed)
    y, x = np.ogrid[:size, :size]

    # Center with slight anatomical translation
    cy = size // 2 + rng.randint(-3, 4)
    cx = size // 2 + rng.randint(-3, 4)

    # Sex-specific anatomical scale
    scale = 1.08 if sex == "Male" else 0.95
    r_outer = int((34 + rng.randint(-2, 3)) * scale)

    # Age-specific wall thickness
    if age > 70:
        thickness = rng.randint(9, 12)  # Wall thinning in senescence
    elif age > 50:
        thickness = rng.randint(12, 15)
    else:
        thickness = rng.randint(15, 18)
    r_inner = max(r_outer - thickness, 8)

    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Ground truth foreground mask (e.g. Myocardial Wall)
    mask = ((dist <= r_outer) & (dist >= r_inner)).astype(np.float32)

    # Base MRI signal intensities
    img = np.zeros((size, size), dtype=np.float32)
    img[dist < r_inner] = 0.72   # blood cavity
    img[mask == 1.0] = 0.46     # target myocardium
    img[dist > r_outer] = 0.18  # background lung / thorax tissue

    # Vendor-specific acquisition physics
    if vendor == "Siemens":
        noise = rng.normal(0, 0.02, size=(size, size))
        img = img + noise
    elif vendor == "Philips":
        # Smooth spatial intensity bias field (B1 inhomogeneity)
        bias = 0.90 + 0.20 * (x / float(size))
        noise = rng.normal(0, 0.035, size=(size, size))
        img = img * bias + noise
    elif vendor == "GE":
        # Attenuated tissue contrast + spatial blur + Rician noise
        img = 0.30 + (img - 0.18) * 0.70
        img = gaussian_filter(img, sigma=0.8)
        n1 = rng.normal(0, 0.075, size=(size, size))
        n2 = rng.normal(0, 0.075, size=(size, size))
        img = np.sqrt((img + n1) ** 2 + n2 ** 2)

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img[np.newaxis, ...], mask[np.newaxis, ...]


class MultiVendorDataset(Dataset):
    """
    PyTorch Dataset of multi-vendor, multi-demographic medical imaging slices.
    """
    def __init__(
        self,
        n_samples: int = 300,
        seed: int = 42,
        vendor_ratios: Tuple[float, float, float] = (0.60, 0.25, 0.15)
    ):
        super().__init__()
        self.seed_base = seed
        rng = np.random.RandomState(seed)

        n_siemens = int(n_samples * vendor_ratios[0])
        n_philips = int(n_samples * vendor_ratios[1])
        n_ge = n_samples - n_siemens - n_philips

        self.vendors = ["Siemens"] * n_siemens + ["Philips"] * n_philips + ["GE"] * n_ge
        # Age distribution: young (<50), middle (50-70), elderly (>70)
        self.ages = rng.choice(
            [28, 38, 48, 58, 68, 76, 84],
            size=n_samples,
            p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.06, 0.04]
        )
        self.sexes = rng.choice(["Male", "Female"], size=n_samples, p=[0.50, 0.50])
        self.patient_ids = [f"PAT_{i+1:04d}" for i in range(n_samples)]

        self.vendor_to_id = {"Siemens": 0, "Philips": 1, "GE": 2}
        self.group_ids = [self.vendor_to_id[v] for v in self.vendors]

    def __len__(self) -> int:
        return len(self.vendors)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        img, mask = generate_medical_slice(
            seed=self.seed_base + idx * 13,
            vendor=self.vendors[idx],
            age=int(self.ages[idx]),
            sex=self.sexes[idx]
        )
        return {
            "image": torch.tensor(img, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "group": torch.tensor(self.group_ids[idx], dtype=torch.long),
            "vendor": self.vendors[idx],
            "age": int(self.ages[idx]),
            "sex": self.sexes[idx],
            "patient_id": self.patient_ids[idx]
        }

    def get_metadata_df(self) -> pd.DataFrame:
        """Returns metadata DataFrame for auditing."""
        return pd.DataFrame({
            "patient_id": self.patient_ids,
            "vendor": self.vendors,
            "age": self.ages,
            "sex": self.sexes,
            "group_id": self.group_ids,
            "age_bracket": [
                "<50" if a < 50 else ("50-70" if a <= 70 else ">70")
                for a in self.ages
            ]
        })


class FairGroupBatchSampler(Sampler):
    """
    Stratified batch sampler that guarantees every batch contains samples
    from every subgroup (e.g. all 3 scanner vendors), preventing gradient
    starvation and zero-sample variance in Group DRO.
    """
    def __init__(self, group_ids: List[int], batch_size: int = 12):
        self.group_ids = np.array(group_ids)
        self.unique_groups = np.unique(self.group_ids)
        self.num_groups = len(self.unique_groups)
        self.batch_size = batch_size
        self.samples_per_group = max(1, batch_size // self.num_groups)

        self.group_indices = {
            g: np.where(self.group_ids == g)[0] for g in self.unique_groups
        }
        self.max_len = max(len(idx) for idx in self.group_indices.values())

    def __iter__(self):
        # Shuffle indices within each group
        shuffled = {
            g: np.random.permutation(self.group_indices[g])
            for g in self.unique_groups
        }
        ptrs = {g: 0 for g in self.unique_groups}
        n_batches = len(self.group_ids) // self.batch_size

        for _ in range(n_batches):
            batch = []
            for g in self.unique_groups:
                g_idx = shuffled[g]
                p = ptrs[g]
                take = self.samples_per_group
                if p + take > len(g_idx):
                    # Wrap around
                    idx_subset = np.concatenate([g_idx[p:], g_idx[: (p + take) % len(g_idx)]])
                    ptrs[g] = (p + take) % len(g_idx)
                else:
                    idx_subset = g_idx[p : p + take]
                    ptrs[g] = p + take
                batch.extend(idx_subset.tolist())
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return len(self.group_ids) // self.batch_size


class CombinedSampleDiceCELoss(nn.Module):
    """
    Computes sample-wise composite Dice Loss + Cross Entropy Loss.
    Outputs loss of shape (B,) without spatial or batch reduction.
    """
    def __init__(self):
        super().__init__()
        self.dice_fn = DiceLoss(to_onehot_y=True, softmax=True, reduction="none")
        self.ce_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: (B, C, H, W) raw network output
            targets: (B, 1, H, W) ground truth mask with integer class labels

        Returns:
            total_sample_loss: (B,)
            dice_sample_loss: (B,)
            ce_sample_loss: (B,)
        """
        # Dice loss per sample: shape (B, C, 1, 1) -> mean over classes
        dice_sample = self.dice_fn(logits, targets).mean(dim=[1, 2, 3])
        # CE loss per sample: shape (B, H, W) -> mean over spatial dimensions
        ce_sample = self.ce_fn(logits, targets.squeeze(1).long()).mean(dim=[1, 2])
        total_sample = dice_sample + ce_sample
        return total_sample, dice_sample, ce_sample


class GroupDROLoss(nn.Module):
    """
    Group Distributionally Robust Optimization (Group DRO) Module.
    Minimizes worst-group risk over G demographic/scanner groups:
        min_theta max_{q in Delta_G} sum_{g=1}^G q_g * R_g(theta)

    Weights q_g are updated online via exponentiated gradient ascent:
        q_g <- q_g * exp(eta_q * loss_g)
        q   <- q / sum(q)

    References:
        Sagawa et al., 'Distributionally Robust Neural Networks for Group Shifts:
        On the Importance of Regularization for Worst-Case Generalization', ICLR 2020.
    """
    def __init__(self, num_groups: int = 3, step_size: float = 0.05):
        super().__init__()
        self.num_groups = num_groups
        self.step_size = step_size
        # Initialize uniform weights on simplex
        self.register_buffer("group_weights", torch.ones(num_groups) / num_groups)
        self.sample_loss_fn = CombinedSampleDiceCELoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: Scaled scalar DRO objective for backpropagation
            group_losses: (G,) individual group losses
            current_q: (G,) updated group weights
        """
        sample_losses, _, _ = self.sample_loss_fn(logits, targets)

        group_losses = torch.zeros(self.num_groups, device=logits.device)
        active_groups = []

        for g in range(self.num_groups):
            mask = (group_indices == g)
            if mask.sum() > 0:
                group_losses[g] = sample_losses[mask].mean()
                active_groups.append(g)

        with torch.no_grad():
            for g in active_groups:
                self.group_weights[g] *= torch.exp(self.step_size * group_losses[g])
            # Re-normalize to simplex
            self.group_weights /= self.group_weights.sum()

        loss = torch.sum(self.group_weights[active_groups] * group_losses[active_groups])
        return loss, group_losses.detach(), self.group_weights.detach()


class DynamicInverseDiceLoss(nn.Module):
    """
    Dynamic Inverse-Dice Group Reweighting Module.
    Dynamically upweights underperforming subgroups based on running Dice scores:
        w_g = G * (1 - Dice_g + eps)^gamma / sum_{g'} (1 - Dice_{g'} + eps)^gamma

    Subgroup Dice scores are tracked via Exponential Moving Average (EMA).
    """
    def __init__(
        self,
        num_groups: int = 3,
        gamma: float = 2.0,
        momentum: float = 0.85,
        eps: float = 1e-4
    ):
        super().__init__()
        self.num_groups = num_groups
        self.gamma = gamma
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("group_dice_ema", torch.full((num_groups,), 0.70))
        self.register_buffer("group_weights", torch.ones(num_groups))
        self.sample_loss_fn = CombinedSampleDiceCELoss()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        group_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: Weighted scalar loss for backpropagation
            group_losses: (G,) individual group losses
            group_dice_ema: (G,) EMA running Dice score per group
            group_weights: (G,) dynamic group weights w_g
        """
        sample_losses, dice_losses, _ = self.sample_loss_fn(logits, targets)
        sample_dice = (1.0 - dice_losses).clamp(0.0, 1.0)

        group_losses = torch.zeros(self.num_groups, device=logits.device)
        active_groups = []

        for g in range(self.num_groups):
            mask = (group_indices == g)
            if mask.sum() > 0:
                group_losses[g] = sample_losses[mask].mean()
                batch_g_dice = sample_dice[mask].mean()
                with torch.no_grad():
                    self.group_dice_ema[g] = (
                        self.momentum * self.group_dice_ema[g] + (1 - self.momentum) * batch_g_dice
                    )
                active_groups.append(g)

        with torch.no_grad():
            raw_w = (1.0 - self.group_dice_ema + self.eps) ** self.gamma
            self.group_weights = (raw_w / raw_w.sum()) * self.num_groups

        loss = torch.sum(self.group_weights[active_groups] * group_losses[active_groups]) / len(active_groups)
        return loss, group_losses.detach(), self.group_dice_ema.detach(), self.group_weights.detach()


def compute_segmentation_metrics(
    pred_mask: np.ndarray,
    true_mask: np.ndarray
) -> Dict[str, float]:
    """
    Computes comprehensive segmentation metrics for a single patient slice:
    - Dice Similarity Coefficient (DSC)
    - Jaccard Index (Intersection-over-Union, IoU)
    - Sensitivity / Recall (TPR)
    - Specificity (TNR)
    - Precision / PPV
    - Hausdorff Distance 95th Percentile (HD95 in pixels)
    """
    pred = (pred_mask > 0.5).astype(bool)
    true = (true_mask > 0.5).astype(bool)

    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    tn = np.logical_and(~pred, ~true).sum()

    eps = 1e-6
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    precision = tp / (tp + fp + eps)

    # Compute HD95 via MONAI
    try:
        t_pred = torch.tensor(pred[np.newaxis, np.newaxis, ...], dtype=torch.float32)
        t_true = torch.tensor(true[np.newaxis, np.newaxis, ...], dtype=torch.float32)
        hd95_val = compute_hausdorff_distance(t_pred, t_true, percentile=95).item()
        if np.isinf(hd95_val) or np.isnan(hd95_val):
            hd95_val = 20.0
    except Exception:
        hd95_val = 20.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "hd95": float(hd95_val)
    }
