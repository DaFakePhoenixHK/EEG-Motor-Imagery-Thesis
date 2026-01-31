# MATLAB Data Integration Guide

## Overview

The pipeline now supports loading MATLAB `.mat` files alongside EDF files. All data (from both sources) is combined and then **randomly split** into train/validation/test sets with **strict separation** - validation and test data are never used during training.

## Data Structure

Your MATLAB files should be named:
- `S01T.mat` - Training data for subject 1
- `S01E.mat` - Evaluation data for subject 1
- `S02T.mat`, `S02E.mat` - Subject 2, etc.

The `.mat` files should contain:
- **Data**: Variable named `data`, `eeg`, `X`, `signal`, `EEG`, `eeg_data`, or `trials`
- **Shape**: `(n_trials, n_channels, n_times)` or `(n_channels, n_times)` for single trial
- **Labels** (optional): Variable named `labels`, `y`, `class`, `target`, `label`, or `Y`
- **Shape**: `(n_trials,)` with values 0 (LEFT) and 1 (RIGHT)

## Usage

### Basic Training with MATLAB Data

```bash
python -m scripts.train_eegnet \
    --data_dir . \
    --matlab_data_dir Matlab_EEG_data \
    --split_mode cross_subject \
    --device cuda \
    --seed 42
```

### Hybrid Ensemble Training

```bash
python -m scripts.train_hybrid \
    --data_dir . \
    --matlab_data_dir Matlab_EEG_data \
    --device cuda \
    --seed 42
```

## How It Works

1. **Data Loading**: 
   - Loads EDF files from `--data_dir` (subject directories like `S001/`, `S002/`, etc.)
   - Loads MATLAB files from `--matlab_data_dir` (files like `S01T.mat`, `S01E.mat`, etc.)
   - Combines training (`T`) and evaluation (`E`) MATLAB files for each subject

2. **Random Split**:
   - **ALL data** (EDF + MATLAB) is combined
   - Randomly split into:
     - **70% Training** - Used for model training
     - **10% Validation** - Used for monitoring and checkpoint saving (NOT used during training)
     - **20% Test** - Used for final evaluation (NOT used during training or validation)
   - Split is **stratified** to maintain class balance
   - Uses `--seed` for reproducibility (if provided)

3. **Strict Separation**:
   - Validation data is **never** used during training
   - Test data is **never** used during training or validation
   - Only training data is used to fit models, compute normalization, etc.

## Key Features

- ✅ **Automatic Discovery**: Finds both EDF and MATLAB files automatically
- ✅ **Random Splits**: Each run with different seed gives different train/val/test splits
- ✅ **Strict Separation**: No data leakage - validation/test never used during training
- ✅ **Reproducible**: Use `--seed` to get same splits every time
- ✅ **Class Balanced**: Stratified splits maintain class distribution

## Example Output

```
Loading ALL data from train subjects...
Subject 1 (EDF): 45 epochs, time_dim=640
Subject 1 (MATLAB): 100 epochs, time_dim=640
Subject 2 (EDF): 45 epochs, time_dim=640
...

Combined data shape: (5000, 64, 640)
Total samples: 5000

============================================================
RANDOM DATA SPLIT (STRICT SEPARATION)
============================================================
Train: 3500 samples (70.0%)
Val: 500 samples (10.0%)
Test: 1000 samples (20.0%)
Class distribution - Train: LEFT=1750, RIGHT=1750
Class distribution - Val: LEFT=250, RIGHT=250
Class distribution - Test: LEFT=500, RIGHT=500
============================================================
IMPORTANT: Validation and test data are NOT used during training!
============================================================
```

## Testing Your MATLAB Files

Before training, test if your `.mat` files can be loaded:

```bash
python -m scripts.test_mat_loader --mat_file Matlab_EEG_data/S01T.mat
```

This will show you:
- Data shape and structure
- Label distribution
- Any errors or warnings

## Notes

- If MATLAB files don't have labels, dummy labels (all 0) will be created
- If time dimensions differ, all data is truncated to the minimum
- Missing subjects/files are skipped with warnings
- Both EDF and MATLAB data are treated equally - no preference given
