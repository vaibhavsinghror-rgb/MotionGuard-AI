# KU-HAR Dataset Specification and Provenance

## 1. Overview
The **KU-HAR** dataset (Khulna University Human Activity Recognition) is a benchmark dataset collected from 90 participants (75 male, 15 female) using smartphone sensors (Accelerometer and Gyroscope) placed in trousers pockets.

- **Permanent DOI:** `10.17632/45f952y38r.1`
- **Mendeley Dataset URL:** [https://data.mendeley.com/datasets/45f952y38r/1](https://data.mendeley.com/datasets/45f952y38r/1)
- **Direct Archive:** `3.Time_domain_subsamples.zip`
- **Archive SHA-256:** `bb804c1338f75cff72e9714f086e0b548a8002baf74a2c6e74a422846e0851fb`
- **File Size:** 123,005,404 bytes (~117.3 MB compressed)

## 2. Activity Classes (18 Target Classes)
| Class ID | Activity Name | Description | Protocol / Count |
|---|---|---|---|
| 0 | **Stand** | Standing still | 1 min continuous |
| 1 | **Sit** | Sitting still | 1 min continuous |
| 2 | **Talk-sit** | Talking with hand gestures while sitting | 1 min continuous |
| 3 | **Talk-stand** | Talking with gestures while standing/walking | 1 min continuous |
| 4 | **Stand-sit** | Repeatedly standing up and sitting down | 5 repetitions |
| 5 | **Lay** | Laying still | 1 min continuous |
| 6 | **Lay-stand** | Repeatedly standing up and laying down | 5 repetitions |
| 7 | **Pick** | Picking up an object from the floor | 10 repetitions |
| 8 | **Jump** | Jumping vertically repeatedly | 10 repetitions |
| 9 | **Push-up** | Full push-ups on the floor | 5 repetitions |
| 10 | **Sit-up** | Sit-ups on the floor | 5 repetitions |
| 11 | **Walk** | Walking forward | 20 meters (~12 s) |
| 12 | **Walk-backward** | Walking backward | 20 meters (~20 s) |
| 13 | **Walk-circle** | Walking along a circular path | ~20 s |
| 14 | **Run** | Running forward | 20 meters (~7 s) |
| 15 | **Stair-up** | Ascending a flight of stairs | ~1 min |
| 16 | **Stair-down** | Descending a flight of stairs | ~50 s |
| 17 | **Table-tennis** | Playing recreational table tennis | 1 min |

## 3. Sensor Layout & Physical Dimensions
- **Sampling Frequency:** 100 Hz
- **Window Length:** 3.0 seconds (300 samples per channel)
- **Sensor Channels (6 total):**
  1. `acc_x` (m/s²): Acceleration along X-axis
  2. `acc_y` (m/s²): Acceleration along Y-axis
  3. `acc_z` (m/s²): Acceleration along Z-axis
  4. `gyro_x` (rad/s): Angular velocity around X-axis
  5. `gyro_y` (rad/s): Angular velocity around Y-axis
  6. `gyro_z` (rad/s): Angular velocity around Z-axis

## 4. CSV Column Layout (`Time_domain_subsamples.csv`)
- **Columns 0 – 299:** Accel X readings (300 timepoints)
- **Columns 300 – 599:** Accel Y readings (300 timepoints)
- **Columns 600 – 899:** Accel Z readings (300 timepoints)
- **Columns 900 – 1199:** Gyro X readings (300 timepoints)
- **Columns 1200 – 1499:** Gyro Y readings (300 timepoints)
- **Columns 1500 – 1799:** Gyro Z readings (300 timepoints)
- **Column 1800:** Ground truth class ID (integer 0 to 17)
- **Column 1801:** Window sample length per channel (constant 300)
- **Column 1802:** Subsample serial identifier

## 5. Splitting Protocol
To eliminate data leakage, MotionGuard AI enforces stratified partitioning:
- **Training Set:** 70% (14,525 windows)
- **Validation Set:** 15% (3,112 windows) - Used strictly for model tuning and selection
- **Held-Out Test Set:** 15% (3,113 windows) - Frozen until final model promotion
- Split indices are deterministically stored in `data/manifests/split_manifest.json`.
