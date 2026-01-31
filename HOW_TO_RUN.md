# How to Run - Step by Step

## Step 1: Open PowerShell/Terminal

Open PowerShell or Command Prompt.

## Step 2: Navigate to the Project Directory

```powershell
cd c:\Users\User\Desktop\Thesis\files
```

## Step 3: Verify You're in the Right Place

Check that you can see the subject folders:
```powershell
ls S001
```

You should see files like `S001R04.edf`, `S001R08.edf`, etc.

## Step 4: Install Dependencies (First Time Only)

```powershell
pip install -r requirements.txt
```

## Step 5: Run Baseline Model (Start Here - It's Faster!)

### Test on ONE subject first:
```powershell
python -m scripts.train_baseline --data_dir . --subjects 1 --runs 4 8 12 --model lda
```

### If that works, try multiple subjects:
```powershell
python -m scripts.train_baseline --data_dir . --subjects 1 2 3 --runs 4 8 12 --model lda
```

### Or try SVM:
```powershell
python -m scripts.train_baseline --data_dir . --subjects 1 2 3 --runs 4 8 12 --model svm
```

## Step 6: Train EEGNet (After Baseline Works)

### On CPU (if you don't have GPU):
```powershell
python -m scripts.train_eegnet --data_dir . --subjects 1 --runs 4 8 12 --device cpu
```

### On GPU (if you have CUDA):
```powershell
python -m scripts.train_eegnet --data_dir . --subjects 1 --runs 4 8 12 --device cuda
```

## What Each Part Means

- `--data_dir .` = Use current directory (where S001, S002 folders are)
- `--subjects 1` = Process subject 1 (you can add more: `1 2 3`)
- `--runs 4 8 12` = Use runs R04, R08, R12 (Task 2)
- `--model lda` = Use LDA classifier (or `svm` for SVM)
- `--device cuda` = Use GPU (or `cpu` for CPU)

## Expected Output

You should see:
- Logging messages about loading files
- Preprocessing steps
- Training progress
- Results saved to `results/baseline/` or `results/eegnet/`

## Troubleshooting

**Error: "No module named 'src'"**
- Make sure you're in `c:\Users\User\Desktop\Thesis\files`
- Run: `cd c:\Users\User\Desktop\Thesis\files`

**Error: "Data directory does not exist"**
- Make sure the S001, S002 folders are in the current directory
- Check with: `ls S001`

**Error: "CUDA not available"**
- Use `--device cpu` instead of `--device cuda`
