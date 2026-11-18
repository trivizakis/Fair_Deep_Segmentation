"""
In-Processing Training Execution Script
=======================================
Trains 3 MONAI UNet models on multi-vendor medical imaging data:
1. Baseline ERM (Empirical Risk Minimization)
2. Group DRO (Group Distributionally Robust Optimization)
3. Dynamic Inverse-Dice Group Reweighting

Saves trained weights to checkpoints/ and exports evaluation metadata.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss

# Import utility classes
from fair_med_utils import (
    MultiVendorDataset,
    FairGroupBatchSampler,
    CombinedSampleDiceCELoss,
    GroupDROLoss,
    DynamicInverseDiceLoss,
    compute_segmentation_metrics
)

import fairmedfm as fm


def build_model(device: torch.device) -> nn.Module:
    """Builds a MONAI 2D UNet with residual units."""
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64),
        strides=(2, 2),
        num_res_units=2
    ).to(device)


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # Dataset splits
    train_ds = MultiVendorDataset(n_samples=300, seed=100)
    val_ds = MultiVendorDataset(n_samples=60, seed=999)
    test_ds = MultiVendorDataset(n_samples=150, seed=5000)

    train_sampler = FairGroupBatchSampler(train_ds.group_ids, batch_size=12)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    num_epochs = 12
    num_groups = 3

    # =========================================================================
    # 1. Baseline ERM Training
    # =========================================================================
    print("\n[1/3] Training Baseline ERM Model...")
    torch.manual_seed(42)
    model_erm = build_model(device)
    opt_erm = torch.optim.AdamW(model_erm.parameters(), lr=2.5e-3, weight_decay=1e-4)
    loss_fn = CombinedSampleDiceCELoss()

    erm_history = {"loss": [], "siemens_loss": [], "philips_loss": [], "ge_loss": []}

    for epoch in range(num_epochs):
        model_erm.train()
        epoch_losses = []
        for b in train_loader:
            x = b["image"].to(device)
            y = b["mask"].to(device)
            logits = model_erm(x)
            losses, _, _ = loss_fn(logits, y)
            loss = losses.mean()

            opt_erm.zero_grad()
            loss.backward()
            opt_erm.step()
            epoch_losses.append(loss.item())

        erm_history["loss"].append(np.mean(epoch_losses))
        if (epoch + 1) % 4 == 0:
            print(f"  Epoch {epoch+1:02d}/{num_epochs} - Loss: {erm_history['loss'][-1]:.4f}")

    # =========================================================================
    # 2. Group DRO Training
    # =========================================================================
    print("\n[2/3] Training Group DRO Model...")
    torch.manual_seed(42)
    model_dro = build_model(device)
    opt_dro = torch.optim.AdamW(model_dro.parameters(), lr=2.5e-3, weight_decay=1e-4)
    dro_loss_fn = GroupDROLoss(num_groups=num_groups, step_size=0.08).to(device)

    dro_history = {"loss": [], "weights_siemens": [], "weights_philips": [], "weights_ge": []}

    for epoch in range(num_epochs):
        model_dro.train()
        epoch_losses = []
        for b in train_loader:
            x = b["image"].to(device)
            y = b["mask"].to(device)
            groups = b["group"].to(device)
            logits = model_dro(x)

            loss, g_losses, q_weights = dro_loss_fn(logits, y, groups)

            opt_dro.zero_grad()
            loss.backward()
            opt_dro.step()
            epoch_losses.append(loss.item())

        dro_history["loss"].append(np.mean(epoch_losses))
        q_np = dro_loss_fn.group_weights.cpu().numpy()
        dro_history["weights_siemens"].append(q_np[0])
        dro_history["weights_philips"].append(q_np[1])
        dro_history["weights_ge"].append(q_np[2])

        if (epoch + 1) % 4 == 0:
            print(f"  Epoch {epoch+1:02d}/{num_epochs} - Loss: {dro_history['loss'][-1]:.4f} | Weights (S/P/GE): {q_np.round(3)}")

    # =========================================================================
    # 3. Dynamic Inverse-Dice Group Reweighting Training
    # =========================================================================
    print("\n[3/3] Training Dynamic Inverse-Dice Reweighted Model...")
    torch.manual_seed(42)
    model_dyn = build_model(device)
    opt_dyn = torch.optim.AdamW(model_dyn.parameters(), lr=2.5e-3, weight_decay=1e-4)
    dyn_loss_fn = DynamicInverseDiceLoss(num_groups=num_groups, gamma=2.0, momentum=0.85).to(device)

    dyn_history = {"loss": [], "dice_ema_ge": [], "weight_ge": []}

    for epoch in range(num_epochs):
        model_dyn.train()
        epoch_losses = []
        for b in train_loader:
            x = b["image"].to(device)
            y = b["mask"].to(device)
            groups = b["group"].to(device)
            logits = model_dyn(x)

            loss, g_losses, dice_ema, w_weights = dyn_loss_fn(logits, y, groups)

            opt_dyn.zero_grad()
            loss.backward()
            opt_dyn.step()
            epoch_losses.append(loss.item())

        dyn_history["loss"].append(np.mean(epoch_losses))
        w_np = dyn_loss_fn.group_weights.cpu().numpy()
        dice_ema_np = dyn_loss_fn.group_dice_ema.cpu().numpy()
        dyn_history["weight_ge"].append(w_np[2])
        dyn_history["dice_ema_ge"].append(dice_ema_np[2])

        if (epoch + 1) % 4 == 0:
            print(f"  Epoch {epoch+1:02d}/{num_epochs} - Loss: {dyn_history['loss'][-1]:.4f} | Weights: {w_np.round(3)}")

    # =========================================================================
    # Save Checkpoints
    # =========================================================================
    os.makedirs("checkpoints", exist_ok=True)
    target_dirs = ["checkpoints", "../FairMedFM example/checkpoints"]
    for d in target_dirs:
        os.makedirs(d, exist_ok=True)
        torch.save(model_erm.state_dict(), os.path.join(d, "baseline_erm_unet.pt"))
        torch.save(model_dro.state_dict(), os.path.join(d, "group_dro_unet.pt"))
        torch.save(model_dyn.state_dict(), os.path.join(d, "dynamic_dice_unet.pt"))
    print("\n[*] Saved all model checkpoints successfully.")

    # Save Test Cohort Metadata and Predictions
    test_meta = test_ds.get_metadata_df()
    test_meta.to_csv("checkpoints/test_cohort_metadata.csv", index=False)
    test_meta.to_csv("../FairMedFM example/checkpoints/test_cohort_metadata.csv", index=False)
    print("[*] Saved test cohort metadata to CSV.")

    # =========================================================================
    # Quick Test Evaluation
    # =========================================================================
    def evaluate_model(m):
        m.eval()
        scores = []
        with torch.no_grad():
            for b in test_loader:
                x = b["image"].to(device)
                y = b["mask"].to(device)
                pred = (m(x).argmax(dim=1, keepdim=True) == 1).float()
                d = compute_segmentation_metrics(pred.cpu().numpy()[0, 0], y.cpu().numpy()[0, 0])["dice"]
                scores.append(d)
        return np.array(scores)

    erm_scores = evaluate_model(model_erm)
    dro_scores = evaluate_model(model_dro)
    dyn_scores = evaluate_model(model_dyn)

    vendors = test_ds.vendors

    print("\n" + "=" * 70)
    print(f"{'Method':<20} | {'Mean Dice':<12} | {'Dice Gap':<12} | {'Equity-Scaled Dice':<18}")
    print("-" * 70)
    for name, s in [("Baseline ERM", erm_scores), ("Group DRO", dro_scores), ("Dynamic Inv-Dice", dyn_scores)]:
        gap = fm.dice_gap(dice=s, sensitive_features=vendors)
        es = fm.equity_scaled_dice(dice=s, sensitive_features=vendors)
        print(f"{name:<20} | {s.mean():<12.4f} | {gap:<12.4f} | {es:<18.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_training()

