"""
Evaluation and metrics for BI-LSTM classifier.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Tuple, Dict
import json


class ModelEvaluator:
    """Comprehensive model evaluation and analysis."""
    
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
        """
        Initialize evaluator.
        
        Args:
            y_true: True labels (0=line, 1=arc)
            y_pred: Predicted labels
            y_prob: Prediction probabilities
        """
        self.y_true = y_true
        self.y_pred = y_pred.flatten() if y_pred.ndim > 1 else y_pred
        self.y_prob = y_prob.flatten() if y_prob.ndim > 1 else y_prob
        
        self.class_names = ['Line', 'Arc']
        
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute comprehensive metrics.
        
        Returns:
            Dictionary of metrics
        """
        # Classification report
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y_true, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # ROC-AUC
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        # PR-AUC
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
        pr_auc = average_precision_score(self.y_true, self.y_prob)
        
        metrics = {
            'accuracy': report['accuracy'],
            'precision_line': report['Line']['precision'],
            'recall_line': report['Line']['recall'],
            'f1_line': report['Line']['f1-score'],
            'precision_arc': report['Arc']['precision'],
            'recall_arc': report['Arc']['recall'],
            'f1_arc': report['Arc']['f1-score'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        }
        
        return metrics
    
    def print_metrics(self):
        """Print formatted metrics."""
        metrics = self.compute_metrics()
        
        print("\n" + "="*60)
        print(" "*20 + "EVALUATION METRICS")
        print("="*60)
        
        print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
        
        print(f"\nLine Classification:")
        print(f"  Precision: {metrics['precision_line']:.4f}")
        print(f"  Recall:    {metrics['recall_line']:.4f}")
        print(f"  F1-Score:  {metrics['f1_line']:.4f}")
        
        print(f"\nArc Classification:")
        print(f"  Precision: {metrics['precision_arc']:.4f}")
        print(f"  Recall:    {metrics['recall_arc']:.4f}")
        print(f"  F1-Score:  {metrics['f1_arc']:.4f}")
        
        print(f"\nMacro Averages:")
        print(f"  Precision: {metrics['macro_precision']:.4f}")
        print(f"  Recall:    {metrics['macro_recall']:.4f}")
        print(f"  F1-Score:  {metrics['macro_f1']:.4f}")
        
        print(f"\nROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC Score:  {metrics['pr_auc']:.4f}")
        
        print(f"\nConfusion Matrix Values:")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print(f"  True Positives:  {metrics['true_positives']}")
        
        print("="*60)
        
        return metrics
    
    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plot confusion matrix.
        
        Args:
            save_path: Optional path to save figure
        """
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, save_path: str = None):
        """
        Plot ROC curve.
        
        Args:
            save_path: Optional path to save figure
        """
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                 label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                  fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, save_path: str = None):
        """
        Plot Precision-Recall curve.
        
        Args:
            save_path: Optional path to save figure
        """
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_prob)
        pr_auc = average_precision_score(self.y_true, self.y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_distribution(self, save_path: str = None):
        """
        Plot distribution of prediction probabilities.
        
        Args:
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Distribution by true label
        for label, name in enumerate(self.class_names):
            probs = self.y_prob[self.y_true == label]
            axes[0].hist(probs, bins=30, alpha=0.6, label=name, edgecolor='black')
        
        axes[0].set_xlabel('Predicted Probability (Arc)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Prediction Distribution by True Label', 
                          fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Distribution by prediction
        correct = self.y_true == self.y_pred
        axes[1].hist(self.y_prob[correct], bins=30, alpha=0.6, 
                    label='Correct', color='green', edgecolor='black')
        axes[1].hist(self.y_prob[~correct], bins=30, alpha=0.6, 
                    label='Incorrect', color='red', edgecolor='black')
        
        axes[1].set_xlabel('Predicted Probability (Arc)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Prediction Distribution by Correctness', 
                          fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def analyze_errors(self, X_test: np.ndarray = None, 
                      shape_ids: list = None) -> Dict:
        """
        Analyze misclassified samples.
        
        Args:
            X_test: Test sequences (optional, for detailed analysis)
            shape_ids: Shape IDs (optional)
            
        Returns:
            Dictionary with error analysis
        """
        # Find misclassified indices
        errors = self.y_true != self.y_pred
        error_indices = np.where(errors)[0]
        
        # False positives (predicted arc, actually line)
        fp_indices = np.where((self.y_pred == 1) & (self.y_true == 0))[0]
        
        # False negatives (predicted line, actually arc)
        fn_indices = np.where((self.y_pred == 0) & (self.y_true == 1))[0]
        
        analysis = {
            'total_errors': len(error_indices),
            'error_rate': len(error_indices) / len(self.y_true),
            'false_positives': len(fp_indices),
            'false_negatives': len(fn_indices),
            'error_indices': error_indices.tolist(),
            'fp_indices': fp_indices.tolist(),
            'fn_indices': fn_indices.tolist()
        }
        
        # Analyze confidence of errors
        if len(error_indices) > 0:
            error_probs = self.y_prob[error_indices]
            analysis['avg_error_confidence'] = float(np.mean(np.abs(error_probs - 0.5)))
            analysis['min_error_prob'] = float(np.min(error_probs))
            analysis['max_error_prob'] = float(np.max(error_probs))
        
        return analysis
    
    def generate_full_report(self, save_dir: str = None):
        """
        Generate comprehensive evaluation report with all plots.
        
        Args:
            save_dir: Directory to save plots and report
        """
        # Compute metrics
        metrics = self.print_metrics()
        
        # Generate plots
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            self.plot_confusion_matrix(f"{save_dir}/confusion_matrix.png")
            self.plot_roc_curve(f"{save_dir}/roc_curve.png")
            self.plot_precision_recall_curve(f"{save_dir}/pr_curve.png")
            self.plot_prediction_distribution(f"{save_dir}/prediction_dist.png")
            
            # Save metrics as JSON
            with open(f"{save_dir}/metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Error analysis
            error_analysis = self.analyze_errors()
            with open(f"{save_dir}/error_analysis.json", 'w') as f:
                json.dump(error_analysis, f, indent=2)
            
            print(f"\nFull report saved to {save_dir}/")
        else:
            self.plot_confusion_matrix()
            self.plot_roc_curve()
            self.plot_precision_recall_curve()
            self.plot_prediction_distribution()
        
        return metrics


def evaluate_from_files(
    y_test_path: str,
    y_pred_path: str,
    y_prob_path: str,
    save_dir: str = None
):
    """
    Load predictions and generate evaluation report.
    
    Args:
        y_test_path: Path to true labels
        y_pred_path: Path to predictions
        y_prob_path: Path to probabilities
        save_dir: Directory to save report
    """
    y_true = np.load(y_test_path)
    y_pred = np.load(y_pred_path)
    y_prob = np.load(y_prob_path)
    
    evaluator = ModelEvaluator(y_true, y_pred, y_prob)
    metrics = evaluator.generate_full_report(save_dir)
    
    return metrics


if __name__ == '__main__':
    # Example: Generate evaluation report
    print("This module is meant to be imported.")
    print("Use ModelEvaluator class for evaluation.")

