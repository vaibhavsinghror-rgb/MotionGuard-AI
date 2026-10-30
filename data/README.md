# MotionGuard AI - Data Directory

This directory stores raw and processed data assets for the KU-HAR benchmark.

## Directory Structure

```text
data/
├── raw/
│   ├── 3.Time_domain_subsamples.zip    # Official Mendeley archive (SHA-256 verified)
│   └── Time_domain_subsamples.csv      # Uncompressed primary dataset
├── processed/
│   ├── waveforms.npy                   # Reshaped 3D array (20,750, 6, 300)
│   ├── labels.npy                      # Target class IDs (20,750,)
│   ├── serials.npy                     # Window sequence serial numbers
│   ├── features_train.npy              # 144 engineered features (Train split)
│   ├── features_val.npy                # 144 engineered features (Val split)
│   ├── features_test.npy               # 144 engineered features (Held-out Test split)
│   └── feature_names.json              # Schema names for all 144 features
└── manifests/
    ├── dataset_manifest.json           # Comprehensive download & verification audit
    ├── validation_report.json          # Programmatic schema & integrity validation report
    └── split_manifest.json             # Exact reproducible stratified split indices
```

## Dataset Reference
- **Source:** Mendeley Data (DOI: 10.17632/45f952y38r.1)
- **Publication:** N. Sikder, A.-A. Nahid, "KU-HAR: An open dataset for heterogeneous human activity recognition", Pattern Recognit. Lett.
- **License:** CC BY 4.0 Open Access.
