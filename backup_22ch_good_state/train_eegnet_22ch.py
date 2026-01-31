"""CLI script for training EEGNet models on 22-channel EEG MATLAB data only (EOG neglected)."""

import argparse
import sys
import json
import logging
import csv
import warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

# Suppress cuDNN warnings (they're harmless - PyTorch falls back to working implementation)
warnings.filterwarnings('ignore', category=UserWarning, message='.*cudnn.*')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import EEGMMIDataset
from src.preprocessing import load_matlab_subject_data
from src.models_eegnet import EEGNet
from src.evaluation import compute_metrics, print_confusion_matrix, log_metrics
from src.plotting import plot_training_curves, plot_test_results
from src.utils import set_seed, setup_logging, log_class_distribution

logger = setup_logging()


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data.
    
    NOTE: This dataset does NOT apply any data augmentation.
    Data is returned as-is without any transformations.
    """
    
    def __init__(self, X, y):
        """Initialize dataset.
        
        Args:
            X: EEG data (n_trials, n_channels, n_times)
            y: Labels (n_trials,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # No augmentation - return data as-is
        return self.X[idx], self.y[idx]


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch.
    
    NOTE: No data augmentation is applied during training.
    Data is used as-is from the dataset.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # No augmentation applied here - using raw data from dataset
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def train_fold(
    X_train, y_train, X_val, y_val,
    n_channels, n_times, n_classes,
    device, results_dir, subject_id, fold_idx,
    n_epochs=400, batch_size=64, learning_rate=5e-4,
    weight_decay=5e-4, label_smoothing=0.05,
    patience=50, min_epochs=50  # patience and min_epochs are NOT used (early stopping disabled)
):
    """Train EEGNet for one fold."""
    
    # Normalize per channel (fit on train only)
    # Compute mean and std across (trials, times) for each channel
    X_train_mean = X_train.mean(axis=(0, 2), keepdims=True)
    X_train_std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8  # Add small epsilon
    
    X_train_norm = (X_train - X_train_mean) / X_train_std
    X_val_norm = (X_val - X_train_mean) / X_train_std
    
    # Check for class imbalance BEFORE creating datasets
    class_counts = np.bincount(y_train)
    logger.info(f"Training class distribution: LEFT (0)={class_counts[0]}, RIGHT (1)={class_counts[1]}")
    if len(class_counts) == 2:
        imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 1.0
        logger.info(f"Class imbalance ratio (LEFT/RIGHT): {imbalance_ratio:.3f}")
        if imbalance_ratio > 1.1 or imbalance_ratio < 0.9:
            logger.warning(f"Significant class imbalance detected! This may cause model bias.")
    
    # Create datasets (NO AUGMENTATION - using original normalized data)
    train_dataset = EEGDataset(X_train_norm, y_train)
    val_dataset = EEGDataset(X_val_norm, y_val)
    
    # Verify data separation
    logger.info(f"Data separation verification:")
    logger.info(f"  Train dataset: {len(train_dataset)} samples")
    logger.info(f"  Val dataset: {len(val_dataset)} samples (SEPARATE)")
    logger.info(f"  Train class dist: LEFT={np.sum(y_train==0)}, RIGHT={np.sum(y_train==1)}")
    logger.info(f"  Val class dist: LEFT={np.sum(y_val==0)}, RIGHT={np.sum(y_val==1)}")
    
    # Shuffle training data for randomization (different order each epoch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model - 22 EEG channels only
    model = EEGNet(
        n_channels=n_channels,  # 22 channels (EEG only)
        n_times=n_times,
        n_classes=n_classes,
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        pool_size=4,
        drop_prob=0.5,
        use_channel_projection=False
    ).to(device)
    
    # Use label smoothing. Weighted criterion for training only; unweighted for logging/plotting.
    criterion_unweighted = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if len(class_counts) == 2 and class_counts[0] > 0 and class_counts[1] > 0:
        total = class_counts.sum()
        if class_counts[0] > class_counts[1]:
            weight_ratio = class_counts[0] / class_counts[1]
            class_weights = torch.FloatTensor([1.0, weight_ratio * 1.2]).to(device)
            logger.info(f"LEFT class is more frequent ({class_counts[0]} vs {class_counts[1]}). Using class weights (no oversampling).")
        else:
            class_weights = torch.FloatTensor([total / (2 * class_counts[0]), total / (2 * class_counts[1])]).to(device)
        logger.info(f"Using class weights: LEFT={class_weights[0]:.3f}, RIGHT={class_weights[1]:.3f}")
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    best_train_acc = 0.0
    best_train_loss = float('inf')
    best_val_acc = 0.0  # Track best validation accuracy separately
    best_epoch = 0
    train_logs = []
    
    logger.info(f"Training fold {fold_idx + 1} for {n_epochs} epochs (NO early stopping)...")
    logger.info("NOTE: Checkpoint saved when:")
    logger.info("  (1) Training accuracy increases AND loss acceptable AND not overfitting, OR")
    logger.info("  (2) Validation accuracy increases (regardless of other conditions)")
    logger.info("CRITICAL: Validation data is NEVER used during training - only for evaluation and checkpointing")
    
    # Create validation loader for evaluation every epoch
    # IMPORTANT: Validation data is ONLY used for evaluation, NEVER for training
    val_loader_checkpoint = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Train set in eval mode (shuffle=False for reproducible evaluation)
    train_loader_eval = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    best_checkpoint_path = None
    
    logger.info("NOTE: train_eval_acc is computed with model.eval() (dropout off); train_acc is computed during training (dropout on).")
    
    # Training loop - Train for full epochs, save checkpoint when BOTH loss decreases AND accuracy increases
    for epoch in range(n_epochs):
        # CRITICAL: Ensure model is in train mode for training
        model.train()
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Sanity metric: evaluate on TRAINING set with model.eval() (dropout off, BN use running stats)
        model.eval()
        train_eval_loss = 0.0
        train_eval_correct = 0
        train_eval_total = 0
        with torch.no_grad():
            for X_batch, y_batch in train_loader_eval:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion_unweighted(outputs, y_batch)
                train_eval_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_eval_total += y_batch.size(0)
                train_eval_correct += (predicted == y_batch).sum().item()
        train_eval_loss = train_eval_loss / len(train_loader_eval)
        train_eval_acc = train_eval_correct / train_eval_total if train_eval_total else 0.0
        
        # Evaluate on validation set: weighted and unweighted loss
        val_loss_weighted = 0.0
        val_loss_unweighted = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader_checkpoint:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                val_loss_weighted += criterion(outputs, y_batch).item()
                val_loss_unweighted += criterion_unweighted(outputs, y_batch).item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()
        val_loss_weighted = val_loss_weighted / len(val_loader_checkpoint)
        val_loss_unweighted = val_loss_unweighted / len(val_loader_checkpoint)
        val_acc = val_correct / val_total
        val_loss = val_loss_weighted  # keep for checkpoint/save logic below
        
        # Save checkpoint with improved rules:
        # 1. Training accuracy increases AND loss acceptable AND not overfitting, OR
        # 2. Validation accuracy increases (regardless of other conditions) - PRIORITY RULE
        # 
        # CRITICAL: Validation data is NEVER used during training - only for evaluation
        # The model.train() call ensures dropout/batch norm use training mode
        # Validation is only evaluated with model.eval() and torch.no_grad()
        
        acc_improved = train_acc > best_train_acc
        loss_decreased = train_loss < best_train_loss
        loss_small_increase = (train_loss - best_train_loss) < 0.05
        loss_acceptable = loss_decreased or loss_small_increase
        
        # Overfitting check: val_acc should not be more than 0.2 lower than train_acc
        overfitting_threshold = 0.2
        not_overfitting = (train_acc - val_acc) <= overfitting_threshold
        
        # PRIORITY: If validation accuracy increases, save regardless of other conditions
        val_acc_improved = val_acc > best_val_acc
        
        should_save = False
        save_reason = []
        
        if val_acc_improved:
            # Validation accuracy improved - save regardless of other conditions
            should_save = True
            save_reason.append(f"val_acc improved ({best_val_acc:.4f} -> {val_acc:.4f})")
            best_val_acc = val_acc
        elif acc_improved and loss_acceptable and not_overfitting:
            # Standard condition: train acc improved, loss acceptable, not overfitting
            should_save = True
            save_reason.append(f"train_acc improved ({best_train_acc:.4f} -> {train_acc:.4f})")
            best_train_acc = train_acc
            best_train_loss = train_loss
        
        if should_save:
            best_epoch = epoch
            if not val_acc_improved:
                best_train_acc = train_acc
                best_train_loss = train_loss
            
            if subject_id == 0:  # Cross-subject mode
                best_checkpoint_path = results_dir / f"cross_subject_fold_{fold_idx + 1}_best.pth"
            else:
                best_checkpoint_path = results_dir / f"subject_{subject_id:03d}_fold_{fold_idx + 1}_best.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, best_checkpoint_path)
            
            loss_indicator = "↓" if loss_decreased else f"↑{train_loss - best_train_loss:.4f}"
            val_indicator = "↑" if val_acc_improved else ""
            logger.info(f"Epoch {epoch + 1}/{n_epochs}: train_loss={train_loss:.4f}{loss_indicator}, train_acc={train_acc:.4f}, train_eval_acc={train_eval_acc:.4f}, val_loss_unw={val_loss_unweighted:.4f}, val_acc={val_acc:.4f}{val_indicator} [BEST - saved: {', '.join(save_reason)}]")
        else:
            reason = []
            if not acc_improved and not val_acc_improved:
                reason.append(f"train_acc not improved (best: {best_train_acc:.4f})")
                reason.append(f"val_acc not improved (best: {best_val_acc:.4f})")
            if not loss_acceptable:
                loss_diff = train_loss - best_train_loss
                reason.append(f"loss increased too much (diff: {loss_diff:.4f}, threshold: 0.05)")
            if not not_overfitting:
                gap = train_acc - val_acc
                reason.append(f"overfitting detected (train-val gap: {gap:.4f} > {overfitting_threshold})")
            logger.info(f"Epoch {epoch + 1}/{n_epochs}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_eval_acc={train_eval_acc:.4f}, val_loss_unw={val_loss_unweighted:.4f}, val_acc={val_acc:.4f} [No save: {', '.join(reason)}]")
        
        train_logs.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_eval_acc': train_eval_acc,
            'train_eval_loss': train_eval_loss,
            'val_loss': val_loss_weighted,
            'val_loss_unweighted': val_loss_unweighted,
            'val_acc': val_acc
        })
        model.train()  # back to train mode for next epoch
        
        # Save training curve plot periodically (every 20 epochs) for real-time monitoring
        if (epoch + 1) % 20 == 0:
            try:
                from src.plotting import plot_training_curves
                
                # Save temporary log file
                if subject_id == 0:  # Cross-subject mode
                    temp_log_file = results_dir / f"cross_subject_fold_{fold_idx + 1}_train_log_temp.csv"
                    temp_plot_file = results_dir / f"cross_subject_fold_{fold_idx + 1}_training_progress.png"
                else:
                    temp_log_file = results_dir / f"subject_{subject_id:03d}_fold_{fold_idx + 1}_train_log_temp.csv"
                    temp_plot_file = results_dir / f"subject_{subject_id:03d}_fold_{fold_idx + 1}_training_progress.png"
                
                # Write current logs to temp CSV
                with open(temp_log_file, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'train_eval_acc', 'train_eval_loss', 'val_loss', 'val_loss_unweighted', 'val_acc'])
                    writer.writeheader()
                    for log in train_logs:
                        writer.writerow({
                            'epoch': log['epoch'],
                            'train_loss': log['train_loss'],
                            'train_acc': log['train_acc'],
                            'train_eval_acc': log['train_eval_acc'],
                            'train_eval_loss': log['train_eval_loss'],
                            'val_loss': log['val_loss'],
                            'val_loss_unweighted': log['val_loss_unweighted'],
                            'val_acc': log['val_acc']
                        })
                
                # Plot current progress
                plot_training_curves(
                    [temp_log_file], 
                    temp_plot_file, 
                    smooth=False, 
                    title=f"Training Progress (Epoch {epoch + 1}/{n_epochs})"
                )
                logger.debug(f"Saved progress plot: {temp_plot_file}")
            except Exception as e:
                logger.debug(f"Could not save progress plot: {e}")
    
    # Load best model (not final epoch) to prevent using broken model if loss exploded
    if best_checkpoint_path and best_checkpoint_path.exists():
        logger.info(f"\nTraining completed. Best checkpoint: loss={best_train_loss:.4f}, acc={best_train_acc:.4f} at epoch {best_epoch + 1}")
        logger.info(f"Loading best checkpoint from epoch {best_epoch + 1} (prevents using broken model if loss exploded)")
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        logger.warning("No best checkpoint found, using final epoch model")
    
    # NOW evaluate on validation set (completely separate, not used during training)
    logger.info("\nEvaluating on VALIDATION set (held-out subjects, not used during training)...")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    y_pred_list = []
    y_proba_list = []
    y_true_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            y_pred_list.append(predicted.cpu().numpy())
            y_proba_list.append(probs.cpu().numpy())
            y_true_list.append(y_batch.numpy())
    
    y_pred = np.concatenate(y_pred_list)
    y_proba = np.concatenate(y_proba_list)
    y_true = np.concatenate(y_true_list)
    
    metrics = compute_metrics(y_true, y_pred, y_proba)
    metrics['final_epoch'] = n_epochs
    metrics['best_epoch'] = best_epoch + 1 if best_checkpoint_path else n_epochs
    metrics['best_train_acc'] = best_train_acc if best_checkpoint_path else None
    metrics['best_train_loss'] = best_train_loss if best_checkpoint_path else None
    metrics['note'] = 'Validation set was NOT used during training - only for evaluation and checkpointing. Best checkpoint saved when: (1) validation accuracy increased (priority, regardless of other conditions), OR (2) training accuracy increased AND loss acceptable AND train-val gap <= 0.2.'
    
    # Plot validation results (for within-subject mode, this is per-fold validation)
    if subject_id != 0:  # Within-subject mode
        val_plot_path = results_dir / f"subject_{subject_id:03d}_fold_{fold_idx + 1}_val_results.png"
        plot_test_results(
            y_true, y_pred, y_proba, metrics,
            val_plot_path,
            title=f"Subject {subject_id:03d} - Fold {fold_idx + 1} Validation Results"
        )
    
    # Save training log
    if subject_id == 0:  # Cross-subject mode
        log_file = results_dir / f"cross_subject_fold_{fold_idx + 1}_train_log.csv"
    else:
        log_file = results_dir / f"subject_{subject_id:03d}_fold_{fold_idx + 1}_train_log.csv"
    
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_acc', 'train_eval_acc', 'train_eval_loss', 'val_loss', 'val_loss_unweighted', 'val_acc'])
        writer.writeheader()
        for log in train_logs:
            writer.writerow({
                'epoch': log['epoch'],
                'train_loss': log['train_loss'],
                'train_acc': log['train_acc'],
                'train_eval_acc': log['train_eval_acc'],
                'train_eval_loss': log['train_eval_loss'],
                'val_loss': log['val_loss'],
                'val_loss_unweighted': log['val_loss_unweighted'],
                'val_acc': log['val_acc']
            })
    
    logger.info(f"Saved training log: {log_file}")
    
    return metrics, log_file


def train_subject_eegnet(
    dataset: EEGMMIDataset,
    subject_id: int,
    config: dict,
    results_dir: Path,
    device: str,
    cv_folds: int = 5,
    seed: int = 42,
    n_epochs: int = 400,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.05,
    patience: int = 50
):
    """Train EEGNet for a single subject with within-subject CV (22-channel EEG MATLAB data only)."""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Subject {subject_id:03d} - EEGNet (22-channel EEG MATLAB)")
    logger.info(f"{'='*60}")
    
    # Get MATLAB files
    matlab_files = dataset.get_subject_matlab_files(subject_id, include_training=True, include_evaluation=True)
    if not matlab_files:
        logger.error(f"No MATLAB files found for subject {subject_id}")
        return None
    
    # Load MATLAB data
    try:
        X_train_mat, y_train_mat, X_eval_mat, y_eval_mat = load_matlab_subject_data(
            matlab_files,
            data_key=None,
            labels_key=None,
            sfreq=160.0,
            use_graz_format=True,
            extract_classes=[1, 2],  # Extract left arm (1) and right arm (2)
            target_channels=22,  # 22 EEG channels
            n_eeg_channels=22   # Keep only EEG; neglect EOG
        )
        
        # Combine training and evaluation MATLAB data
        mat_X_list = []
        mat_y_list = []
        if X_train_mat is not None and y_train_mat is not None:
            mat_X_list.append(X_train_mat)
            mat_y_list.append(y_train_mat)
        if X_eval_mat is not None and y_eval_mat is not None:
            mat_X_list.append(X_eval_mat)
            mat_y_list.append(y_eval_mat)
        
        if not mat_X_list:
            logger.error(f"No MATLAB data loaded for subject {subject_id}")
            return None
        
        X = np.concatenate(mat_X_list, axis=0)
        y = np.concatenate(mat_y_list, axis=0)
        
    except Exception as e:
        logger.error(f"Failed to load MATLAB data for subject {subject_id}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None
    
    logger.info(f"Total epochs: {len(X)}")
    log_class_distribution(y, logger)
    
    # Get data dimensions
    n_channels, n_times = X.shape[1], X.shape[2]
    n_classes = len(np.unique(y))
    
    logger.info(f"Data shape: {X.shape} (channels={n_channels}, times={n_times})")
    
    if n_channels != 22:
        logger.warning(f"Expected 22 channels, got {n_channels}. This script is for 22-channel EEG MATLAB data only.")
    
    # Within-subject stratified K-fold CV
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    
    fold_metrics = []
    log_files = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{cv_folds} ---")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
        
        metrics, log_file = train_fold(
            X_train, y_train, X_val, y_val,
            n_channels, n_times, n_classes,
            device, results_dir, subject_id, fold_idx,
            n_epochs, batch_size, learning_rate, weight_decay, label_smoothing, patience
        )
        
        metrics['fold'] = fold_idx + 1
        fold_metrics.append(metrics)
        log_files.append(log_file)
        
        logger.info(f"Fold {fold_idx + 1} - Val Accuracy: {metrics['accuracy']:.4f}")
    
    # Aggregate
    accuracies = [m['accuracy'] for m in fold_metrics]
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Subject {subject_id:03d} Summary:")
    logger.info(f"  Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    logger.info(f"{'='*60}\n")
    
    # Save results
    subject_results = {
        'subject_id': subject_id,
        'model_type': 'eegnet_22ch',
        'n_folds': cv_folds,
        'mean_accuracy': float(mean_accuracy),
        'std_accuracy': float(std_accuracy),
        'fold_metrics': fold_metrics
    }
    
    results_file = results_dir / f"subject_{subject_id:03d}_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(subject_results, f, indent=2)
    
    # Plot training curves
    plot_file = results_dir / f"subject_{subject_id:03d}_training_curves.png"
    plot_training_curves(log_files, plot_file, smooth=False, title=f"Subject {subject_id:03d} - EEGNet Training (22ch)")
    
    return subject_results


def train_cross_subject_eegnet(
    dataset: EEGMMIDataset,
    train_subjects: list,
    val_subjects: list,
    test_subjects: list,
    config: dict,
    results_dir: Path,
    device: str,
    seed: int = 42,
    n_epochs: int = 400,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    weight_decay: float = 5e-4,
    label_smoothing: float = 0.05,
    patience: int = 50
):
    """Train EEGNet in cross-subject mode (22-channel EEG MATLAB data only)."""
    
    logger.info(f"\n{'='*60}")
    logger.info("CROSS-SUBJECT MODE - EEGNet (22-channel EEG MATLAB)")
    logger.info(f"{'='*60}")
    logger.info(f"Train subjects ({len(train_subjects)}): {train_subjects}")
    logger.info(f"Val subjects ({len(val_subjects)}): {val_subjects}")
    logger.info(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    logger.info(f"{'='*60}")
    
    # Load data from MATLAB files only (22 EEG channels; EOG neglected)
    def load_subjects_data_22ch(subject_list):
        """Load 22-channel EEG MATLAB data from subjects (EOG neglected)."""
        X_22ch = []
        y_22ch = []
        time_dims_22ch = []
        
        for subject_id in subject_list:
            # Load MATLAB files (22 EEG channels only)
            matlab_files = dataset.get_subject_matlab_files(subject_id, include_training=True, include_evaluation=True)
            if matlab_files:
                try:
                    X_train_mat, y_train_mat, X_eval_mat, y_eval_mat = load_matlab_subject_data(
                        matlab_files,
                        data_key=None,
                        labels_key=None,
                        sfreq=160.0,
                        use_graz_format=True,
                        extract_classes=[1, 2],  # Extract left arm (1) and right arm (2)
                        target_channels=22,  # 22 EEG channels
                        n_eeg_channels=22   # Keep only EEG; neglect EOG
                    )
                    
                    # Combine training and evaluation MATLAB data
                    mat_X_list = []
                    mat_y_list = []
                    if X_train_mat is not None and y_train_mat is not None:
                        mat_X_list.append(X_train_mat)
                        mat_y_list.append(y_train_mat)
                    if X_eval_mat is not None and y_eval_mat is not None:
                        mat_X_list.append(X_eval_mat)
                        mat_y_list.append(y_eval_mat)
                    
                    if mat_X_list:
                        X_mat = np.concatenate(mat_X_list, axis=0)
                        y_mat = np.concatenate(mat_y_list, axis=0)
                        X_22ch.append(X_mat)
                        y_22ch.append(y_mat)
                        time_dims_22ch.append(X_mat.shape[2])
                        logger.info(f"Subject {subject_id} (MATLAB, 22ch): {len(X_mat)} epochs, time_dim={X_mat.shape[2]}")
                except Exception as e:
                    logger.warning(f"Failed to load MATLAB data for subject {subject_id}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
        
        if not X_22ch:
            return None
        
        # Process 22-channel data
        time_dims_unique = list(set(time_dims_22ch))
        if len(time_dims_unique) > 1:
            min_time = min(time_dims_unique)
            logger.warning(f"22ch data: different time dimensions {time_dims_unique}, truncating to {min_time}")
            X_22ch = [X[:, :, :min_time] if X.shape[2] > min_time else X for X in X_22ch]
        
        # Ensure all arrays have same time dimension before concatenation
        if X_22ch:
            final_time_dim = X_22ch[0].shape[2]
            X_22ch_aligned = []
            for X in X_22ch:
                if X.shape[2] != final_time_dim:
                    X = X[:, :, :final_time_dim] if X.shape[2] > final_time_dim else X
                X_22ch_aligned.append(X)
            X_22ch = X_22ch_aligned
        
        X_concat_22ch = np.concatenate(X_22ch, axis=0)
        y_concat_22ch = np.concatenate(y_22ch, axis=0)
        logger.info(f"22-channel data: {X_concat_22ch.shape}")
        
        return (X_concat_22ch, y_concat_22ch)
    
    # Load train/val/test data (22-channel MATLAB only)
    logger.info("\nLoading training data (22-channel MATLAB)...")
    data_22ch_train = load_subjects_data_22ch(train_subjects)
    
    logger.info("Loading validation data (22-channel MATLAB)...")
    data_22ch_val = load_subjects_data_22ch(val_subjects)
    
    logger.info("Loading test data (22-channel MATLAB)...")
    data_22ch_test = load_subjects_data_22ch(test_subjects)
    
    # Log what was loaded
    logger.info("\n" + "="*60)
    logger.info("DATA LOADING SUMMARY")
    logger.info("="*60)
    logger.info(f"22-channel MATLAB data:")
    logger.info(f"  Train: {'Loaded' if data_22ch_train is not None else 'NOT FOUND'}")
    logger.info(f"  Val: {'Loaded' if data_22ch_val is not None else 'NOT FOUND'}")
    logger.info(f"  Test: {'Loaded' if data_22ch_test is not None else 'NOT FOUND'}")
    logger.info("="*60 + "\n")
    
    if data_22ch_train is None or data_22ch_val is None:
        logger.error("22-channel MATLAB data missing - cannot train model")
        return None
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING 22-CHANNEL MODEL (MATLAB data)")
    logger.info("="*60)
    
    X_train_22ch, y_train_22ch = data_22ch_train
    X_val_22ch, y_val_22ch = data_22ch_val
    X_test_22ch, y_test_22ch = data_22ch_test if data_22ch_test is not None else (None, None)
    
    # NO trial-level re-split: Train = all trials from train_subjects, Val = all from val_subjects, Test = all from test_subjects.
    # This avoids subject-level leakage (same subject in train and val/test).
    X_train_22ch_final = X_train_22ch
    y_train_22ch_final = y_train_22ch
    X_val_22ch_final = X_val_22ch
    y_val_22ch_final = y_val_22ch
    X_test_22ch_final = X_test_22ch if X_test_22ch is not None else None
    y_test_22ch_final = y_test_22ch if y_test_22ch is not None else None
    
    # Align time dimension across train/val/test (truncate to minimum)
    time_dims = [X_train_22ch_final.shape[2], X_val_22ch_final.shape[2]]
    if X_test_22ch_final is not None:
        time_dims.append(X_test_22ch_final.shape[2])
    n_times_22ch = min(time_dims)
    if len(set(time_dims)) > 1:
        logger.warning(f"22ch time dimension mismatch: {time_dims}, truncating to {n_times_22ch}")
    X_train_22ch_final = X_train_22ch_final[:, :, :n_times_22ch]
    X_val_22ch_final = X_val_22ch_final[:, :, :n_times_22ch]
    if X_test_22ch_final is not None:
        X_test_22ch_final = X_test_22ch_final[:, :, :n_times_22ch]
    
    logger.info("")
    logger.info("CROSS-SUBJECT SPLIT (no subject overlap):")
    logger.info(f"  Train set = all trials from train_subjects only: {train_subjects}")
    logger.info(f"  Val set   = all trials from val_subjects only:   {val_subjects}")
    logger.info(f"  Test set  = all trials from test_subjects only:   {test_subjects}")
    overlap_train_val = set(train_subjects) & set(val_subjects)
    overlap_train_test = set(train_subjects) & set(test_subjects)
    overlap_val_test = set(val_subjects) & set(test_subjects)
    if overlap_train_val or overlap_train_test or overlap_val_test:
        logger.error("SUBJECT LEAKAGE: train/val/test subject sets must not overlap!")
    else:
        logger.info("  Confirmed: no subject ID overlap between train, val, and test.")
    logger.info(f"  22ch data: Train={X_train_22ch_final.shape}, Val={X_val_22ch_final.shape}, Test={X_test_22ch_final.shape if X_test_22ch_final is not None else None}")
    logger.info("")
    
    # Class distribution (no oversampling; we use class_weights only)
    train_class_counts = np.bincount(y_train_22ch_final)
    logger.info(f"Training data class distribution (class_weights used, no oversampling):")
    logger.info(f"  LEFT (0): {train_class_counts[0]} samples ({100*train_class_counts[0]/len(y_train_22ch_final):.1f}%)")
    logger.info(f"  RIGHT (1): {train_class_counts[1]} samples ({100*train_class_counts[1]/len(y_train_22ch_final):.1f}%)")
    
    X_train_22ch_original = X_train_22ch_final  # for test normalization (no oversampling, so same as final)
    
    n_channels_22ch = 22
    n_classes = len(np.unique(y_train_22ch_final))
    
    # Train 22-channel model
    metrics_22ch, log_file_22ch = train_fold(
        X_train_22ch_final, y_train_22ch_final, X_val_22ch_final, y_val_22ch_final,
        n_channels_22ch, n_times_22ch, n_classes,
        device, results_dir, 0, 0,
        n_epochs, batch_size, learning_rate, weight_decay, label_smoothing, patience
    )
    
    # Test 22-channel model (ALWAYS run test evaluation after training)
    logger.info("\n" + "="*60)
    logger.info("TEST SET EVALUATION (HELD-OUT DATA, NOT USED DURING TRAINING)")
    logger.info("="*60)
    test_metrics_22ch = None
    if X_test_22ch_final is not None and len(X_test_22ch_final) > 0:
        logger.info("Evaluating 22-channel model on test set...")
        
        checkpoint_path_22ch = results_dir / "cross_subject_fold_1_best.pth"
        if checkpoint_path_22ch.exists():
            checkpoint = torch.load(checkpoint_path_22ch)
            
            # CRITICAL: Infer n_times from checkpoint to avoid size mismatch
            # The checkpoint was saved with a specific n_times that we need to match
            inferred_n_times = n_times_22ch  # Default to current value
            fc_weight = checkpoint['model_state_dict'].get('fc.weight', None)
            if fc_weight is not None:
                # fc.weight shape is (n_classes, flatten_size)
                # flatten_size = F2 * ((n_times // pool_size) // 8)
                # With F2=16, pool_size=4: flatten_size = 16 * ((n_times // 4) // 8) = 16 * (n_times // 32)
                # So: n_times = (flatten_size / 16) * 32 = flatten_size * 2
                checkpoint_flatten_size = fc_weight.shape[1]
                inferred_n_times = (checkpoint_flatten_size // 16) * 32
                logger.info(f"Checkpoint fc.weight shape: {fc_weight.shape} -> flatten_size={checkpoint_flatten_size}")
                logger.info(f"Inferred n_times from checkpoint: {inferred_n_times} (original: {n_times_22ch})")
                
                if inferred_n_times != n_times_22ch:
                    logger.warning(f"Time dimension mismatch: using checkpoint-inferred value {inferred_n_times} instead of {n_times_22ch}")
                    n_times_22ch = inferred_n_times
            
            # Truncate test data to match inferred n_times
            if X_test_22ch_final.shape[2] != n_times_22ch:
                logger.info(f"Truncating test data from {X_test_22ch_final.shape[2]} to {n_times_22ch} time points")
                X_test_22ch_final = X_test_22ch_final[:, :, :n_times_22ch]
            
            # IMPORTANT: Use ORIGINAL training data (before oversampling) for normalization stats
            # While train_fold normalizes from oversampled data, using original data is more statistically correct
            # (oversampling just duplicates samples, so mean/std are identical, but original represents true distribution)
            # Truncate original training data to match n_times for normalization stats
            X_train_22ch_for_norm = X_train_22ch_original
            if X_train_22ch_original.shape[2] != n_times_22ch:
                X_train_22ch_for_norm = X_train_22ch_original[:, :, :n_times_22ch]
            
            X_train_mean_22ch = X_train_22ch_for_norm.mean(axis=(0, 2), keepdims=True)
            X_train_std_22ch = X_train_22ch_for_norm.std(axis=(0, 2), keepdims=True) + 1e-8
            
            X_test_norm_22ch = (X_test_22ch_final - X_train_mean_22ch) / X_train_std_22ch
            
            model_22ch = EEGNet(n_channels=22, n_times=n_times_22ch, n_classes=n_classes, use_channel_projection=False).to(device)
            model_22ch.load_state_dict(checkpoint['model_state_dict'])
            model_22ch.eval()
            
            test_dataset_22ch = EEGDataset(X_test_norm_22ch, y_test_22ch_final)
            test_loader_22ch = DataLoader(test_dataset_22ch, batch_size=batch_size, shuffle=False)
            
            y_pred_22ch, y_proba_22ch, y_true_22ch = [], [], []
            with torch.no_grad():
                for X_batch, y_batch in test_loader_22ch:
                    X_batch = X_batch.to(device)
                    outputs = model_22ch(X_batch)
                    probs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    y_pred_22ch.append(predicted.cpu().numpy())
                    y_proba_22ch.append(probs.cpu().numpy())
                    y_true_22ch.append(y_batch.numpy())
            
            y_pred_22ch = np.concatenate(y_pred_22ch)
            y_proba_22ch = np.concatenate(y_proba_22ch)
            y_true_22ch = np.concatenate(y_true_22ch)
            
            test_metrics_22ch = compute_metrics(y_true_22ch, y_pred_22ch, y_proba_22ch)
            logger.info(f"\nTest Set Results (on HELD-OUT test subjects, not used during training):")
            log_metrics(test_metrics_22ch, logger)
            print_confusion_matrix(
                np.array(test_metrics_22ch['confusion_matrix']),
                labels=['LEFT (0)', 'RIGHT (1)']
            )
            
            # Plot test results (ROC curve, confusion matrix, etc.)
            test_plot_path = results_dir / "test_results.png"
            plot_test_results(
                y_true_22ch, y_pred_22ch, y_proba_22ch, test_metrics_22ch,
                test_plot_path,
                title=f"Test Set Results (Cross-Subject, {len(test_subjects)} held-out subjects)"
            )
            logger.info(f"Saved test results plot: {test_plot_path}")
    
    # Save results
    results = {
        'split_mode': 'cross_subject',
        'model_type': 'eegnet_22ch',
        'train_subjects': train_subjects,
        'val_subjects': val_subjects,
        'test_subjects': test_subjects,
        'val_metrics': metrics_22ch,
        'test_metrics': test_metrics_22ch
    }
    
    results_file = results_dir / "cross_subject_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved results to: {results_file}")
    
    # Plot training curves
    log_file = results_dir / "cross_subject_fold_1_train_log.csv"
    if log_file.exists():
        plot_file = results_dir / "cross_subject_training_curves.png"
        plot_training_curves([log_file], plot_file, smooth=False, title="Cross-Subject EEGNet Training (22ch)")
        logger.info(f"Saved training curves plot: {plot_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train EEGNet on 22-channel EEG MATLAB data only (EOG neglected)')
    parser.add_argument('--matlab_data_dir', type=str, required=True, help='Path to MATLAB .mat files directory (e.g., "Four class motor imagery (001-2014)")')
    parser.add_argument('--data_dir', type=str, default='.', help='Path to dataset root (for dataset initialization, not used for loading)')
    parser.add_argument('--subjects', type=int, nargs='+', default=None, help='Subject IDs (default: all available MATLAB subjects)')
    parser.add_argument('--split_mode', type=str, choices=['within_subject', 'cross_subject'], default='cross_subject', help='Split mode: within_subject or cross_subject (default: cross_subject)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--config', type=str, default='configs/default.json', help='Config file')
    parser.add_argument('--results_dir', type=str, default='results/eegnet_22ch', help='Results directory')
    parser.add_argument('--cv_folds', type=int, default=5, help='CV folds (within-subject only)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None for different random splits each run)')
    parser.add_argument('--n_epochs', type=int, default=400, help='Max epochs (default: 400)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing (default: 0.05)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience (DISABLED - not used)')
    
    args = parser.parse_args()
    
    # Set seed (if provided, otherwise use random seed for different results each run)
    if args.seed is not None:
        set_seed(args.seed)
        logger.info(f"Using fixed seed: {args.seed} (results will be reproducible)")
    else:
        actual_seed = set_seed(None)
        logger.info(f"Using random seed: {actual_seed} (results will differ each run)")
        args.seed = actual_seed
    
    # Device (handle 'gpu' as alias for 'cuda')
    if args.device.lower() == 'gpu':
        args.device = 'cuda'
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = 'cpu'
    elif args.device not in ['cpu', 'cuda']:
        logger.warning(f"Invalid device '{args.device}', using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(config_path)
    
    # Setup results directory
    results_dir = Path(__file__).parent.parent / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Log split mode and results directory
    logger.info(f"\n{'='*60}")
    logger.info(f"Split Mode: {args.split_mode.upper()}")
    logger.info(f"Results Directory: {results_dir.absolute()}")
    logger.info(f"{'='*60}")
    
    # Initialize dataset (MATLAB data only)
    dataset = EEGMMIDataset(args.data_dir, matlab_data_dir=args.matlab_data_dir)
    
    # Discover MATLAB subjects
    matlab_subjects_available = []
    logger.info("Discovering subjects with MATLAB files...")
    for test_subj_id in range(1, 20):  # Check subjects 1-19
        matlab_files = dataset.get_subject_matlab_files(test_subj_id, include_training=True, include_evaluation=True)
        if matlab_files:
            matlab_subjects_available.append(test_subj_id)
    logger.info(f"Found MATLAB files for subjects: {matlab_subjects_available}")
    
    if not matlab_subjects_available:
        logger.error("No MATLAB subjects found! Check --matlab_data_dir path.")
        return
    
    # Determine subjects to use
    if args.subjects is None:
        valid_subjects = matlab_subjects_available
        logger.info(f"Using all {len(valid_subjects)} available MATLAB subjects")
    else:
        valid_subjects = [s for s in args.subjects if s in matlab_subjects_available]
        if len(valid_subjects) < len(args.subjects):
            missing = set(args.subjects) - set(valid_subjects)
            logger.warning(f"Some subjects not found: {missing}")
    
    if not valid_subjects:
        logger.error(f"No valid MATLAB subjects found")
        return
    
    # Handle cross-subject mode
    if args.split_mode == 'cross_subject':
        if len(valid_subjects) <= 3:
            logger.warning(f"WARNING: Only {len(valid_subjects)} subjects available for cross-subject training.")
        
        # Split subjects
        if args.seed is not None:
            rng = np.random.default_rng(args.seed)
        else:
            rng = np.random.default_rng()
        
        subjects_shuffled = valid_subjects.copy()
        rng.shuffle(subjects_shuffled)
        
        # 75% train, 10% val, 15% test at SUBJECT level (no trial-level re-split)
        n = len(subjects_shuffled)
        n_train = max(1, int(n * 0.75))
        n_val = max(1, int(n * 0.10))
        n_test = max(0, n - n_train - n_val)
        if n_test == 0 and n_val + n_train < n:
            n_test = n - n_train - n_val
        
        train_subjects = sorted(subjects_shuffled[:n_train])
        val_subjects = sorted(subjects_shuffled[n_train:n_train + n_val])
        test_subjects = sorted(subjects_shuffled[n_train + n_val:])
        
        logger.info(f"Cross-subject split at SUBJECT level (75/10/15, seed={args.seed}): {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test subjects")
        logger.info(f"Train subjects: {train_subjects}")
        logger.info(f"Val subjects: {val_subjects}")
        logger.info(f"Test subjects: {test_subjects}")
        
        # Train cross-subject model
        result = train_cross_subject_eegnet(
            dataset, train_subjects, val_subjects, test_subjects,
            config, results_dir, device,
            args.seed, args.n_epochs, args.batch_size,
            args.learning_rate, args.weight_decay, args.label_smoothing, args.patience
        )
        
        logger.info(f"\nAll results saved to: {results_dir}")
        return
    
    # Within-subject mode (default)
    logger.info(f"Training EEGNet on {len(valid_subjects)} MATLAB subjects: {valid_subjects}")
    
    # Train each subject
    all_results = []
    for subject_id in valid_subjects:
        result = train_subject_eegnet(
            dataset, subject_id, config, results_dir, device,
            args.cv_folds, args.seed, args.n_epochs, args.batch_size,
            args.learning_rate, args.weight_decay, args.label_smoothing, args.patience
        )
        if result:
            all_results.append(result)
    
    logger.info(f"\nAll results saved to: {results_dir}")


if __name__ == '__main__':
    main()
