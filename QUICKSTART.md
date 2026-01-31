# Quick Start Guide

## Important: Use the New Clean Implementation

The new clean implementation is located at:
```
c:\Users\User\Desktop\Thesis\files\
```

**NOT** the old messy code in:
```
c:\Users\User\Downloads\eeg-motor-movementimagery-dataset-1.0.0\files\
```

## Setup

1. Navigate to the new project directory:
```powershell
cd c:\Users\User\Desktop\Thesis\files
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

## Test with Baseline Models First

### Test CSP+LDA on one subject:
```powershell
python -m scripts.train_baseline --data_dir . --subjects 1 --runs 4 8 12 --model lda
```

### Test CSP+SVM:
```powershell
python -m scripts.train_baseline --data_dir . --subjects 1 --runs 4 8 12 --model svm
```

### Run on multiple subjects:
```powershell
python -m scripts.train_baseline --data_dir . --subjects 1 2 3 --runs 4 8 12 --model lda
```

## Train EEGNet (After Baseline Works)

```powershell
python -m scripts.train_eegnet --data_dir . --subjects 1 2 3 --runs 4 8 12 --device cuda
```

If CUDA is not available, use:
```powershell
python -m scripts.train_eegnet --data_dir . --subjects 1 2 3 --runs 4 8 12 --device cpu
```

## Key Differences from Old Script

The new implementation has a **much simpler CLI**:

**Old script (messy):** 40+ arguments with complex augmentation options
**New script (clean):** Simple, focused arguments:
- `--data_dir` (required)
- `--subjects` (required)
- `--runs` (optional, default: 4 8 12)
- `--device` (optional, default: cuda)
- `--n_epochs`, `--batch_size`, `--learning_rate`, etc. (optional)

## Verify Installation

Check that you can import the modules:
```powershell
python -c "from src.dataset import EEGMMIDataset; print('OK')"
```

## Troubleshooting

If you get import errors, make sure you're in the correct directory:
```powershell
cd c:\Users\User\Desktop\Thesis\files
pwd  # Should show: c:\Users\User\Desktop\Thesis\files
```
