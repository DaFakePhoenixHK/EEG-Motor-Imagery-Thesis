# Fix NumPy Version Issue

## Problem
PyTorch was compiled with NumPy 1.x but you have NumPy 2.2.6 installed.

## Solution

Run this command to downgrade NumPy:

```powershell
pip install "numpy<2.0.0"
```

Or reinstall all requirements:

```powershell
pip install -r requirements.txt --upgrade
```

## Then Run EEGNet

Use `cuda` (not `gpu`) for the device:

```powershell
python -m scripts.train_eegnet --data_dir . --subjects 1 --runs 4 8 12 --device cuda
```

Or use CPU:

```powershell
python -m scripts.train_eegnet --data_dir . --subjects 1 --runs 4 8 12 --device cpu
```
