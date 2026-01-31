# EEG Motor Imagery Classification

Clean, reproducible pipeline for binary motor imagery classification using the PhysioNet EEG Motor Movement/Imagery (EEGMMI) dataset v1.0.0.

## Task

Binary classification: **LEFT fist vs RIGHT fist** motor imagery
- Task 2 runs: R04, R08, R12 per subject
- Annotations: T1 = left onset, T2 = right onset, T0 = rest (ignored)
- Sampling rate: 160 Hz
- Channels: 64 EEG channels

## Dataset Layout

Expected dataset structure:
```
data_dir/
├── S001/
│   ├── S001R04.edf
│   ├── S001R08.edf
│   ├── S001R12.edf
│   └── ...
├── S002/
│   └── ...
└── ...
```

### Using Additional .mat Files

The pipeline now supports MATLAB `.mat` files in addition to EDF files. You can add `.mat` files containing preprocessed EEG data to supplement your dataset.

**Supported .mat file formats:**
- Data shape: `(n_trials, n_channels, n_times)` or `(n_channels, n_times)` for single trial
- Labels: Optional, can be in separate key or auto-detected
- Common variable names: `data`, `eeg`, `X`, `signal`, `EEG`, `eeg_data`, `trials`
- Common label names: `labels`, `y`, `class`, `target`, `label`, `Y`

**File naming conventions:**
- `S001R04.mat` (same as EDF naming)
- `S001_R04.mat`
- `subject_1_run_4.mat`
- `sub001_run04.mat`

**Usage:** The pipeline will automatically discover and use `.mat` files alongside EDF files. You can also manually specify `.mat` files using the dataset API (see `src/dataset.py`).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Baseline Models (CSP + LDA / CSP + SVM)

**Train on ALL 109 subjects (recommended):**
```bash
python -m scripts.train_baseline --data_dir . --runs 4 8 12 --model lda
```

**Or test on a few subjects first:**
```bash
python -m scripts.train_baseline --data_dir . --subjects 1 2 3 --runs 4 8 12 --model lda
```

**Note:** If you don't specify `--subjects`, the script will automatically use all available subjects (109 total).

**Options:**
- `--data_dir`: Path to dataset root directory
- `--subjects`: Space-separated list of subject IDs (optional - if not provided, uses all 109 available subjects)
- `--runs`: Run IDs to use (default: `4 8 12`)
- `--model`: Model type (`lda` or `svm`)
- `--cv_folds`: Number of CV folds (default: 5)
- `--seed`: Random seed (default: 42)
- `--filter_low`: Low cutoff frequency (overrides config)
- `--filter_high`: High cutoff frequency (overrides config)
- `--tmin`: Epoch start time in seconds (overrides config)
- `--tmax`: Epoch end time in seconds (overrides config)
- `--csp_components`: Number of CSP components (overrides config)
- `--results_dir`: Results directory (default: `results/baseline`)

### EEGNet (Deep Learning)

**Within-Subject Mode (Default) - ALL 109 subjects:**
```bash
python -m scripts.train_eegnet --data_dir . --runs 4 8 12 --device cuda
```

**Or test on a few subjects first:**
```bash
python -m scripts.train_eegnet --data_dir . --subjects 1 2 3 --runs 4 8 12 --device cuda
```

**Cross-Subject Mode (Train on some subjects, test on others):**
```bash
# Use all 109 subjects (recommended for cross-subject)
python -m scripts.train_eegnet --data_dir . --runs 4 8 12 --split_mode cross_subject --device cuda

# Or specify subjects (will warn if <=10 subjects)
python -m scripts.train_eegnet --data_dir . --subjects 1 2 3 4 5 6 7 8 9 10 --runs 4 8 12 --split_mode cross_subject --device cuda
```

**Options:**
- `--split_mode`: `within_subject` (default) or `cross_subject`
- `--subjects`: Subject IDs (optional - if not provided, uses all available subjects)
- `--device`: Device (`cuda` or `cpu`, default: `cuda`)
- `--n_epochs`: Maximum epochs (default: 200)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--patience`: Early stopping patience (default: 50)
- Other options same as baseline

**Note:** To use all 109 subjects in cross-subject mode, **do not pass `--subjects`**. The script will automatically discover and use all available subjects.

### Hybrid Ensemble (CSP+SVM + EEGNet)

Combine CSP+SVM and EEGNet using ensemble method (weighted average of predictions):

```bash
python -m scripts.train_hybrid --data_dir . --runs 4 8 12 --device cuda
```

**Options:**
- `--ensemble_weight_csp`: Weight for CSP+SVM (default: 0.5)
- `--ensemble_weight_eegnet`: Weight for EEGNet (default: 0.5)
- Other options same as EEGNet

**Example with custom weights (70% EEGNet, 30% CSP+SVM):**
```bash
python -m scripts.train_hybrid --data_dir . --runs 4 8 12 --device cuda --ensemble_weight_csp 0.3 --ensemble_weight_eegnet 0.7
```

### Plotting Training Logs

Plot training curves for a specific subject:
```bash
python -m scripts.plot_training_logs --results_dir results/eegnet --subject_id 1
```

Plot with smoothing:
```bash
python -m scripts.plot_training_logs --results_dir results/eegnet --subject_id 1 --smooth --smooth_window 5
```

## Results Structure

### Baseline Results
```
results/baseline/
├── subject_001_metrics.json    # Per-subject metrics
├── subject_002_metrics.json
├── ...
├── summary.csv                  # Summary across all subjects
└── summary_lda.png              # Bar plot of subject accuracies
```

### EEGNet Results
```
results/eegnet/
├── subject_001_metrics.json
├── subject_001_fold_1_best.pth          # Best model checkpoint
├── subject_001_fold_1_train_log.csv     # Training log
├── subject_001_fold_2_best.pth
├── subject_001_fold_2_train_log.csv
├── ...
└── subject_001_training_curves.png      # Training curves plot
```

## Understanding the Plots

### Baseline Summary Plot
- Bar chart showing accuracy per subject
- Error bars show standard deviation across CV folds
- Red dashed line indicates chance level (50%)

### Training Curves Plot
- **Left panel**: Loss curves (train vs validation)
- **Right panel**: Accuracy curves (train vs validation)
- **Bold lines**: Mean across folds
- **Shaded regions**: Mean ± 1 standard deviation
- **Faint lines**: Individual fold curves

The mean ± std visualization shows:
- **Mean line**: Average performance across all folds
- **Shaded region**: Variability (one standard deviation above and below the mean)
- **Per-fold curves**: Individual training runs (shown faintly)

## Configuration

Default configuration is in `configs/default.json`:
- Preprocessing: notch filter (50 Hz), bandpass (8-30 Hz), CAR re-referencing
- Epochs: 0.0 to 4.0 seconds
- CSP: 4 components
- CV: 5-fold stratified

## Evaluation

- **Within-subject evaluation**: Default workflow uses stratified K-fold CV within each subject
- **Cross-subject evaluation**: Train on some subjects, validate/test on others (subject-level split: 70% train, 15% val, 15% test)
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **No data leakage**: All normalization and feature extraction fitted on training data only

## Reproducibility

- Random seeds set for numpy, random, and PyTorch
- Deterministic behavior where possible
- All preprocessing steps logged

## Project Structure

```
.
├── src/
│   ├── dataset.py              # Dataset discovery
│   ├── preprocessing.py        # MNE preprocessing + epoching
│   ├── features_baseline.py   # CSP extraction
│   ├── models_baseline.py     # LDA + SVM pipelines
│   ├── models_eegnet.py       # EEGNet implementation
│   ├── evaluation.py          # Metrics computation
│   ├── plotting.py            # Visualization
│   └── utils.py               # Utilities (seeding, logging)
├── configs/
│   └── default.json           # Default configuration
├── scripts/
│   ├── train_baseline.py      # Baseline training script
│   ├── train_eegnet.py        # EEGNet training script
│   └── plot_training_logs.py  # Plotting script
└── results/                   # Output directory (auto-created)
```

## Notes

- Label mapping: T1 (left) → 0, T2 (right) → 1
- Baseline correction: None by default (configurable)
- Artifact rejection: Off by default (configurable via `reject` parameter)
- Cross-subject evaluation: Available via `--split_mode cross_subject` (uses all available subjects by default)
