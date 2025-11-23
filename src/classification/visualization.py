"""
Visualization utilities for trajectory classification.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional


def plot_training_history(history_dict: dict, save_path: str = None):
    """
    Plot training history.
    
    Args:
        history_dict: Dictionary with training history
        save_path: Optional path to save figure
    """
    metrics = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    n_metrics = len([m for m in metrics if m in history_dict])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    for metric in metrics:
        if metric not in history_dict:
            continue
            
        ax = axes[plot_idx]
        
        # Plot training metric
        ax.plot(history_dict[metric], label=f'Train {metric}', linewidth=2)
        
        # Plot validation metric if available
        val_metric = f'val_{metric}'
        if val_metric in history_dict:
            ax.plot(history_dict[val_metric], label=f'Val {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        plot_idx += 1
    
    # Remove unused subplots
    for idx in range(plot_idx, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_sample_predictions(
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    df: pd.DataFrame = None,
    n_samples: int = 8,
    save_path: str = None
):
    """
    Plot sample trajectories with predictions.
    
    Args:
        X: Input sequences (n_samples, seq_len, n_features)
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        df: Original dataframe (optional, to show actual trajectories)
        n_samples: Number of samples to plot
        save_path: Optional path to save figure
    """
    # Select samples: half correct, half incorrect
    correct_indices = np.where(y_true == y_pred)[0]
    incorrect_indices = np.where(y_true != y_pred)[0]
    
    n_correct = min(n_samples // 2, len(correct_indices))
    n_incorrect = min(n_samples - n_correct, len(incorrect_indices))
    
    sample_indices = np.concatenate([
        np.random.choice(correct_indices, n_correct, replace=False),
        np.random.choice(incorrect_indices, n_incorrect, replace=False)
    ])
    
    n_cols = 4
    n_rows = (len(sample_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    class_names = ['Line', 'Arc']
    
    for idx, sample_idx in enumerate(sample_indices):
        ax = axes[idx]
        
        # Get trajectory (using first 2 features which are relative x, y)
        traj = X[sample_idx, :, :2]
        
        # Remove padding (zero rows)
        non_zero_mask = ~np.all(traj == 0, axis=1)
        traj = traj[non_zero_mask]
        
        # Plot trajectory
        ax.plot(traj[:, 0], traj[:, 1], 'o-', linewidth=2, markersize=4)
        
        # Title with prediction info
        true_label = class_names[int(y_true[sample_idx])]
        pred_label = class_names[int(y_pred[sample_idx])]
        prob = y_prob[sample_idx]
        
        is_correct = y_true[sample_idx] == y_pred[sample_idx]
        color = 'green' if is_correct else 'red'
        
        title = f"True: {true_label}, Pred: {pred_label}\n"
        title += f"Confidence: {prob:.3f}"
        
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        ax.set_xlabel('X (relative)', fontsize=9)
        ax.set_ylabel('Y (relative)', fontsize=9)
        ax.grid(alpha=0.3)
        ax.axis('equal')
    
    # Remove unused subplots
    for idx in range(len(sample_indices), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions plot saved to {save_path}")
    
    plt.show()


def plot_feature_distributions(X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str] = None,
                               save_path: str = None):
    """
    Plot feature distributions for each class.
    
    Args:
        X: Feature sequences (n_samples, seq_len, n_features)
        y: Labels
        feature_names: Names of features
        save_path: Optional path to save figure
    """
    n_features = X.shape[2]
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]
    
    # Compute mean features across time for each sample
    X_mean = np.mean(X, axis=1)  # (n_samples, n_features)
    
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    class_names = ['Line', 'Arc']
    
    for feat_idx in range(n_features):
        ax = axes[feat_idx]
        
        for label, name in enumerate(class_names):
            data = X_mean[y == label, feat_idx]
            ax.hist(data, bins=30, alpha=0.6, label=name, edgecolor='black')
        
        ax.set_xlabel(feature_names[feat_idx], fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{feature_names[feat_idx]} Distribution', 
                    fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Remove unused subplots
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distribution plot saved to {save_path}")
    
    plt.show()


def plot_sequence_length_analysis(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str = None
):
    """
    Analyze and plot sequence length distributions.
    
    Args:
        X: Feature sequences
        y: Labels
        save_path: Optional path to save figure
    """
    # Compute actual sequence lengths (non-padded)
    lengths = []
    for seq in X:
        non_zero_mask = ~np.all(seq == 0, axis=1)
        lengths.append(np.sum(non_zero_mask))
    
    lengths = np.array(lengths)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    class_names = ['Line', 'Arc']
    
    # Overall distribution
    axes[0].hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Sequence Length', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Sequence Length Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Distribution by class
    for label, name in enumerate(class_names):
        class_lengths = lengths[y == label]
        axes[1].hist(class_lengths, bins=30, alpha=0.6, label=name, edgecolor='black')
    
    axes[1].set_xlabel('Sequence Length', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Sequence Length by Class', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sequence length analysis saved to {save_path}")
    
    plt.show()


def create_comprehensive_visualization(
    history: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    save_dir: str = None
):
    """
    Create comprehensive visualization report.
    
    Args:
        history: Training history
        X_test: Test sequences
        y_test: Test labels
        y_pred: Predictions
        y_prob: Probabilities
        save_dir: Directory to save figures
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    plot_training_history(
        history,
        save_path=f"{save_dir}/training_history.png" if save_dir else None
    )
    
    # Sample predictions
    plot_sample_predictions(
        X_test, y_test, y_pred, y_prob,
        n_samples=12,
        save_path=f"{save_dir}/sample_predictions.png" if save_dir else None
    )
    
    # Feature distributions
    feature_names = [
        'Rel X', 'Rel Y', 'dx', 'dy', 'Speed', 
        'Dist from Center', 'Angle', 'Curvature', 
        'Acceleration', 'Consecutive Dist'
    ]
    plot_feature_distributions(
        X_test, y_test, feature_names,
        save_path=f"{save_dir}/feature_distributions.png" if save_dir else None
    )
    
    # Sequence length analysis
    plot_sequence_length_analysis(
        X_test, y_test,
        save_path=f"{save_dir}/sequence_lengths.png" if save_dir else None
    )
    
    print(f"\nVisualization complete!")
    if save_dir:
        print(f"All figures saved to {save_dir}/")


if __name__ == '__main__':
    print("This module provides visualization utilities.")
    print("Import and use the functions in your analysis scripts.")

