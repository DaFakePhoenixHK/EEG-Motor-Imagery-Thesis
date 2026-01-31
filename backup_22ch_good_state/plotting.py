"""Plotting utilities for baseline results and training curves."""

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def plot_baseline_summary(
    summary_df: pd.DataFrame,
    output_path: Path,
    model_name: str = "Baseline"
):
    """Plot bar chart of subject accuracies.
    
    Args:
        summary_df: DataFrame with columns ['subject_id', 'mean_accuracy', 'std_accuracy']
        output_path: Path to save plot
        model_name: Name of model for title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    subjects = summary_df['subject_id'].values
    means = summary_df['mean_accuracy'].values
    stds = summary_df['std_accuracy'].values if 'std_accuracy' in summary_df.columns else None
    
    x_pos = np.arange(len(subjects))
    
    if stds is not None:
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='steelblue')
    else:
        ax.bar(x_pos, means, alpha=0.7, color='steelblue')
    
    ax.set_xlabel('Subject ID', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title(f'{model_name} - Subject-wise Accuracy (Within-Subject CV)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'S{s:03d}' for s in subjects], rotation=45, ha='right')
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance (50%)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved baseline summary plot: {output_path}")


def plot_training_curves(
    log_files: List[Path],
    output_path: Path,
    smooth: bool = False,
    smooth_window: int = 5,
    title: str = "Training Curves"
):
    """Plot training curves from CSV log files.
    
    Args:
        log_files: List of paths to CSV log files (one per fold)
        output_path: Path to save plot
        smooth: Whether to apply smoothing
        smooth_window: Smoothing window size
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    all_train_loss = []
    all_train_acc = []
    all_train_eval_acc = []
    all_val_loss = []
    all_val_loss_unweighted = []
    all_val_acc = []
    epochs_list = []
    has_validation = False
    has_train_eval_acc = False
    has_val_loss_unweighted = False
    
    # Load all fold data
    for log_file in log_files:
        if not log_file.exists():
            logger.warning(f"Log file not found: {log_file}")
            continue
        
        df = pd.read_csv(log_file)
        
        epochs = df['epoch'].values
        train_loss = df['train_loss'].values
        train_acc = df['train_acc'].values
        
        epochs_list.append(epochs)
        all_train_loss.append(train_loss)
        all_train_acc.append(train_acc)
        
        if 'train_eval_acc' in df.columns and df['train_eval_acc'].notna().any():
            train_eval_acc = df['train_eval_acc'].dropna().values
            if len(train_eval_acc) > 0:
                all_train_eval_acc.append(train_eval_acc[:len(epochs)])
                has_train_eval_acc = True
        if 'val_loss_unweighted' in df.columns and df['val_loss_unweighted'].notna().any():
            val_loss_uw = df['val_loss_unweighted'].dropna().values
            if len(val_loss_uw) > 0:
                all_val_loss_unweighted.append(val_loss_uw[:len(epochs)])
                has_val_loss_unweighted = True
        
        # Check if validation columns exist (may not be present if validation not used during training)
        if 'val_loss' in df.columns and 'val_acc' in df.columns and df['val_loss'].notna().any():
            val_loss = df['val_loss'].dropna().values
            val_acc = df['val_acc'].dropna().values
            if len(val_loss) > 0:
                all_val_loss.append(val_loss)
                all_val_acc.append(val_acc)
                has_validation = True
        
        # Plot individual fold curves (faint)
        axes[0].plot(epochs, train_loss, alpha=0.2, color='blue', linewidth=0.5)
        axes[1].plot(epochs, train_acc, alpha=0.2, color='blue', linewidth=0.5)
        
        # Plot validation curves if available
        if has_validation and len(all_val_loss) > 0 and len(epochs) == len(val_loss):
            axes[0].plot(epochs, val_loss, alpha=0.2, color='red', linewidth=0.5)
            axes[1].plot(epochs, val_acc, alpha=0.2, color='red', linewidth=0.5)
        if has_train_eval_acc and len(all_train_eval_acc) > 0 and len(all_train_eval_acc[-1]) == len(epochs):
            axes[1].plot(epochs, all_train_eval_acc[-1], alpha=0.2, color='green', linewidth=0.5)
    
    if not epochs_list:
        logger.error("No log data found to plot")
        plt.close()
        return
    
    # Find common epoch range
    min_epochs = min(len(e) for e in epochs_list)
    epochs_common = epochs_list[0][:min_epochs]
    
    # Align all arrays to same length
    all_train_loss = [arr[:min_epochs] for arr in all_train_loss]
    all_train_acc = [arr[:min_epochs] for arr in all_train_acc]
    
    # Compute mean and std for training
    train_loss_mean = np.mean(all_train_loss, axis=0)
    train_loss_std = np.std(all_train_loss, axis=0)
    train_acc_mean = np.mean(all_train_acc, axis=0)
    train_acc_std = np.std(all_train_acc, axis=0)
    
    # Compute mean and std for validation if available
    val_loss_mean = None
    val_loss_std = None
    val_loss_unweighted_mean = None
    val_loss_unweighted_std = None
    val_acc_mean = None
    val_acc_std = None
    train_eval_acc_mean = None
    train_eval_acc_std = None
    
    if has_validation and len(all_val_loss) > 0:
        all_val_loss_aligned = [arr[:min_epochs] if len(arr) >= min_epochs else arr for arr in all_val_loss]
        all_val_acc_aligned = [arr[:min_epochs] if len(arr) >= min_epochs else arr for arr in all_val_acc]
        if len(all_val_loss_aligned) > 0 and all(len(arr) == min_epochs for arr in all_val_loss_aligned):
            val_loss_mean = np.mean(all_val_loss_aligned, axis=0)
            val_loss_std = np.std(all_val_loss_aligned, axis=0)
            val_acc_mean = np.mean(all_val_acc_aligned, axis=0)
            val_acc_std = np.std(all_val_acc_aligned, axis=0)
    
    if has_val_loss_unweighted and len(all_val_loss_unweighted) > 0:
        all_val_uw_aligned = [arr[:min_epochs] if len(arr) >= min_epochs else arr for arr in all_val_loss_unweighted]
        if all(len(arr) == min_epochs for arr in all_val_uw_aligned):
            val_loss_unweighted_mean = np.mean(all_val_uw_aligned, axis=0)
            val_loss_unweighted_std = np.std(all_val_uw_aligned, axis=0)
    
    if has_train_eval_acc and len(all_train_eval_acc) > 0:
        all_train_eval_aligned = [arr[:min_epochs] if len(arr) >= min_epochs else arr for arr in all_train_eval_acc]
        if all(len(arr) == min_epochs for arr in all_train_eval_aligned):
            train_eval_acc_mean = np.mean(all_train_eval_aligned, axis=0)
            train_eval_acc_std = np.std(all_train_eval_aligned, axis=0)
    
    # Apply smoothing if requested
    if smooth:
        try:
            from scipy.ndimage import uniform_filter1d
            train_loss_mean = uniform_filter1d(train_loss_mean, size=smooth_window, mode='nearest')
            train_acc_mean = uniform_filter1d(train_acc_mean, size=smooth_window, mode='nearest')
            if val_loss_mean is not None:
                val_loss_mean = uniform_filter1d(val_loss_mean, size=smooth_window, mode='nearest')
                val_acc_mean = uniform_filter1d(val_acc_mean, size=smooth_window, mode='nearest')
            if val_loss_unweighted_mean is not None:
                val_loss_unweighted_mean = uniform_filter1d(val_loss_unweighted_mean, size=smooth_window, mode='nearest')
            if train_eval_acc_mean is not None:
                train_eval_acc_mean = uniform_filter1d(train_eval_acc_mean, size=smooth_window, mode='nearest')
        except ImportError:
            logger.warning("scipy not available, skipping smoothing")
    
    # Plot mean ± std for training
    axes[0].plot(epochs_common, train_loss_mean, 'b-', label='Train (mean)', linewidth=2)
    axes[0].fill_between(epochs_common, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std, 
                         alpha=0.3, color='blue')
    
    axes[1].plot(epochs_common, train_acc_mean, 'b-', label='Train (mean)', linewidth=2)
    axes[1].fill_between(epochs_common, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std, 
                         alpha=0.3, color='blue')
    
    # Plot train_eval_acc if available (sanity: dropout off on train set)
    if train_eval_acc_mean is not None:
        axes[1].plot(epochs_common, train_eval_acc_mean, 'g-', label='Train eval (mean)', linewidth=2)
        if train_eval_acc_std is not None:
            axes[1].fill_between(epochs_common, train_eval_acc_mean - train_eval_acc_std, train_eval_acc_mean + train_eval_acc_std,
                                alpha=0.3, color='green')
    
    # Plot validation if available (prefer unweighted val loss for fair comparison with train loss)
    if val_acc_mean is not None:
        axes[1].plot(epochs_common, val_acc_mean, 'r-', label='Val (mean)', linewidth=2)
        axes[1].fill_between(epochs_common, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, 
                             alpha=0.3, color='red')
    if val_loss_unweighted_mean is not None:
        axes[0].plot(epochs_common, val_loss_unweighted_mean, 'r-', label='Val unweighted (mean)', linewidth=2)
        if val_loss_unweighted_std is not None:
            axes[0].fill_between(epochs_common, val_loss_unweighted_mean - val_loss_unweighted_std, val_loss_unweighted_mean + val_loss_unweighted_std,
                                 alpha=0.3, color='red')
    elif val_loss_mean is not None:
        axes[0].plot(epochs_common, val_loss_mean, 'r-', label='Val (mean)', linewidth=2)
        axes[0].fill_between(epochs_common, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, 
                             alpha=0.3, color='red')
    if val_acc_mean is None and val_loss_mean is None and val_loss_unweighted_mean is None:
        axes[0].text(0.02, 0.98, 'Note: Validation held out\n(not used during training)', 
                    transform=axes[0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].text(0.02, 0.98, 'Note: Validation held out\n(not used during training)', 
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    if train_eval_acc_mean is not None:
        axes[1].text(0.02, 0.02, 'train_eval_acc: model.eval() (dropout off)\ntrain_acc: during training (dropout on)', 
                    transform=axes[1].transAxes, fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Formatting
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0, 1.0])
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved training curves plot: {output_path}")


def plot_test_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    metrics: Dict,
    output_path: Path,
    title: str = "Test Set Results"
):
    """Plot comprehensive test results including confusion matrix and ROC curve.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (shape: [n_samples, n_classes])
        metrics: Dictionary with computed metrics
        output_path: Path to save plot
        title: Plot title
    """
    from sklearn.metrics import roc_curve, auc
    
    fig = plt.figure(figsize=(16, 5))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(1, 3, 1)
    cm = np.array(metrics['confusion_matrix'])
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')
    
    labels = ['LEFT (0)', 'RIGHT (1)']
    ax1.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold', pad=10)
    
    # 2. ROC Curve
    ax2 = plt.subplot(1, 3, 2)
    if y_proba.shape[1] == 2:
        # Binary classification
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        ax2.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('False Positive Rate', fontsize=12)
        ax2.set_ylabel('True Positive Rate', fontsize=12)
        ax2.set_title('ROC Curve', fontsize=13, fontweight='bold', pad=10)
        ax2.legend(loc="lower right", fontsize=10)
        ax2.grid(alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'ROC curve\nnot available\nfor multi-class', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('ROC Curve', fontsize=13, fontweight='bold', pad=10)
    
    # 3. Metrics Summary
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    # Create metrics text
    metrics_text = f"""
    TEST SET PERFORMANCE
    {'='*30}
    
    Accuracy:     {metrics['accuracy']:.4f}
    Precision:    {metrics['precision']:.4f}
    Recall:       {metrics['recall']:.4f}
    F1-Score:     {metrics['f1_score']:.4f}
    ROC-AUC:      {metrics['roc_auc']:.4f}
    
    {'='*30}
    
    Class Distribution:
    LEFT (0):     {np.sum(y_true == 0)} samples
    RIGHT (1):    {np.sum(y_true == 1)} samples
    
    {'='*30}
    
    Predictions:
    LEFT (0):     {np.sum(y_pred == 0)} samples
    RIGHT (1):    {np.sum(y_pred == 1)} samples
    """
    
    ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
            fontsize=11, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved test results plot: {output_path}")
