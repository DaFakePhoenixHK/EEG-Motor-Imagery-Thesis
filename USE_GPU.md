# Using Your RTX 4070 GPU

## Step 1: Check if CUDA is Available

Run this command:
```powershell
python check_cuda.py
```

If it shows:
- `CUDA available: True` → You're good to go! ✅
- `CUDA available: False` → You need to install PyTorch with CUDA support

## Step 2: Fix NumPy (Required First!)

Before running EEGNet, downgrade NumPy:
```powershell
pip install "numpy<2.0.0"
```

## Step 3: Run EEGNet with GPU

Once CUDA is available, use:
```powershell
python -m scripts.train_eegnet --data_dir . --subjects 1 --runs 4 8 12 --device cuda
```

## If CUDA is Not Available

You may need to install PyTorch with CUDA 12.1 support (for RTX 4070):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then verify:
```powershell
python check_cuda.py
```

## Expected Output with GPU

When running with `--device cuda`, you should see:
- Training will be much faster
- No "CUDA not available" warnings
- GPU memory usage in Task Manager

## Performance Tips

- RTX 4070 has 12GB VRAM - you can increase batch size:
  ```powershell
  python -m scripts.train_eegnet --data_dir . --subjects 1 --runs 4 8 12 --device cuda --batch_size 32
  ```
