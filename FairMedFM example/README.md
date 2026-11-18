# Post-Hoc Fairness Auditing in Medical Image Segmentation: FairMedFM & Fairlearn

This directory contains the auditing pipeline, evaluation protocols, and interactive Jupyter notebook for conducting **post-hoc fairness audits** of medical image segmentation models using **FairMedFM** and **Fairlearn's MetricFrame**.

---

## 1. Directory Structure

```
FairMedFM example/
├── 02_posthoc_fairness_auditing_fairmedfm_fairlearn.ipynb  # Interactive, pre-rendered Jupyter Notebook
├── audit_fairness.py                                       # Standalone CLI auditing script
├── fair_med_utils.py                                        # Multi-metric computation & data loading utilities
├── checkpoints/                                             # Evaluated model checkpoints & test metadata
│   ├── baseline_erm_unet.pt                                 # Baseline ERM UNet
│   ├── group_dro_unet.pt                                    # Group DRO UNet
│   ├── dynamic_dice_unet.pt                                 # Dynamic Inverse-Dice UNet
│   └── test_cohort_metadata.csv                             # Demographic & vendor metadata
├── audit_reports/                                           # Exported audit cards and raw results
│   ├── fairness_audit_report.md                             # Formatted clinical audit report (Markdown)
│   ├── fairness_audit_summary.json                          # Structured model card metadata (JSON)
│   └── patient_evaluation_results.csv                       # Per-patient multi-metric evaluation table
└── figures/                                                 # Diagnostic visualization panels
    ├── fairmedfm_disparity_comparison.png                   # Bar charts comparing ΔDSC & Equity-Scaled Dice
    ├── subgroup_dice_distributions.png                      # Subgroup violin plots across vendors & age
    ├── intersectional_disparities.png                       # Cross-attribute evaluation (Vendor × Age)
    └── spatial_error_decomposition.png                      # Pixel error maps (TP vs FP vs FN)
```

---

## 2. Core Auditing Metrics & Theoretical Formulations

### FairMedFM Metrics
**FairMedFM** is an open-source benchmarking and auditing framework designed specifically for evaluating disparities in medical foundation and deep learning models.

1. **Dice Gap ($\Delta \text{DSC}$ / `delta_dice`):**
   Measures the performance disparity between the highest- and lowest-performing subgroups:
   $$\Delta \text{DSC} = \max_{g \in \mathcal{G}} \text{DSC}_g - \min_{g \in \mathcal{G}} \text{DSC}_g$$
   *Range: $[0, 1]$; $0.0$ indicates perfect demographic parity across groups.*

2. **Equity-Scaled Dice ($\text{DSC}_{\text{ES}}$ / `es_dice`):**
   Balances aggregate diagnostic utility with subgroup equality by penalizing the cross-group standard deviation:
   $$\text{DSC}_{\text{ES}} = \frac{\mu_{\text{DSC}}}{1 + \sigma_{\text{group}}}$$
   *where $\mu_{\text{DSC}}$ is the mean Dice across all samples, and $\sigma_{\text{group}}$ is the standard deviation of subgroup mean Dice scores.*

3. **Worst-Group Dice (`worst_group_dice`):**
   $$\text{DSC}_{\text{worst}} = \min_{g \in \mathcal{G}} \text{DSC}_g$$
   Establishes the minimum safety floor for vulnerable patient populations.

4. **Continuous Attribute Binning:**
   FairMedFM allows continuous clinical parameters (such as patient age) to be partitioned into discrete clinical brackets:
   ```python
   report = fm.evaluate_segmentation(sensitive_features=df['age'], dice=dice_scores, bins={'age': [50, 70]})
   ```

---

### Fairlearn's `MetricFrame` for Segmentation
**Fairlearn's `MetricFrame`** disaggregates any arbitrary collection of clinical segmentation metrics (Dice, IoU, Sensitivity, Specificity, HD95) across sensitive features and computes disparity ratios:

```python
from fairlearn.metrics import MetricFrame

def mean_score(y_true, y_pred):
    return float(np.mean(y_pred))

mf = MetricFrame(
    metrics={'Dice': mean_score, 'IoU': mean_score},
    y_true=dice_scores,
    y_pred=dice_scores,
    sensitive_features=df['vendor']
)

# Overall aggregate performance
print(mf.overall)

# Performance disaggregated by subgroup
print(mf.by_group)

# Absolute disparity (Max - Min)
print(mf.difference())

# Disparity ratio (Min / Max) -> Four-fifths compliance
print(mf.ratio())
```

---

## 3. Regulatory Certification & Clinical Standards

Auditing performed in this module aligns with international regulatory standards:
- **FDA Good Machine Learning Practice (GMLP):** Principle 6 (Model performance is monitored across clinically relevant subgroups).
- **EU AI Act (Annex IV):** Mandatory bias auditing, risk mitigation documentation, and performance equity certification prior to CE-marking for medical devices.
- **EEOC Four-Fifths (80%) Rule:** The disparity ratio $\min / \max$ must exceed $0.80$ to avoid a finding of adverse disparate impact.

All audited models demonstrated four-fifths compliance with disparity ratios $>0.98$.

