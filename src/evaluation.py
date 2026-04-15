"""
Evaluation Module for Model Analysis
Provides comprehensive evaluation metrics, visualizations, and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import os


def compute_predictions(model, X_val, y_val, label_encoder: LabelEncoder):
    """
    Compute model predictions on validation set.
    
    Args:
        model: Trained Keras model
        X_val: Validation images (preprocessed)
        y_val: Validation labels (encoded integers)
        label_encoder: LabelEncoder for decoding labels
        
    Returns:
        Tuple of (y_true, y_pred, y_pred_labels, confidence_scores)
    """
    # Get predictions from model
    y_pred_probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    confidence_scores = np.max(y_pred_probs, axis=1)
    
    # Decode predictions to label names
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    return y_val, y_pred, y_pred_labels, confidence_scores


def generate_confusion_matrix(y_true, y_pred, label_encoder: LabelEncoder, save_path: str = None):
    """
    Generate and visualize confusion matrix.
    
    Args:
        y_true: True labels (encoded integers)
        y_pred: Predicted labels (encoded integers)
        label_encoder: LabelEncoder for label names
        save_path: Path to save confusion matrix plot
        
    Returns:
        Confusion matrix array
    """
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get label names
    label_names = label_encoder.classes_
    
    # Create visualization
    plt.figure(figsize=(14, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'},
        annot_kws={'size': 8}
    )
    
    plt.title('Confusion Matrix - Model Prediction Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm


def generate_classification_report(y_true, y_pred, label_encoder: LabelEncoder):
    """
    Generate and print classification report with per-class metrics.
    
    Args:
        y_true: True labels (encoded integers)
        y_pred: Predicted labels (encoded integers)
        label_encoder: LabelEncoder for label names
        
    Returns:
        Dictionary with classification report
    """
    label_names = label_encoder.classes_
    
    # Generate report
    report = classification_report(
        y_true, y_pred,
        target_names=label_names,
        digits=4,
        output_dict=True
    )
    
    # Print formatted report
    print("\n" + "="*80)
    print("CLASSIFICATION REPORT - Per-Class Performance")
    print("="*80)
    print(f"\n{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    for label in label_names:
        precision = report[label]['precision']
        recall = report[label]['recall']
        f1 = report[label]['f1-score']
        support = int(report[label]['support'])
        
        print(f"{label:<25} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}")
    
    print("-"*80)
    print(f"{'Accuracy':<25} {report['accuracy']:<12.4f}")
    print(f"{'Macro Average':<25} {report['macro avg']['precision']:<12.4f} "
          f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f}")
    print(f"{'Weighted Average':<25} {report['weighted avg']['precision']:<12.4f} "
          f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f}")
    print("="*80 + "\n")
    
    return report


def compute_per_class_accuracy(y_true, y_pred, label_encoder: LabelEncoder):
    """
    Compute per-class accuracy.
    
    Args:
        y_true: True labels (encoded integers)
        y_pred: Predicted labels (encoded integers)
        label_encoder: LabelEncoder for label names
        
    Returns:
        Dictionary with per-class accuracies
    """
    label_names = label_encoder.classes_
    per_class_acc = {}
    
    print("\n" + "="*50)
    print("PER-CLASS ACCURACY")
    print("="*50)
    
    for idx, label in enumerate(label_names):
        mask = y_true == idx
        if mask.sum() > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask])
            per_class_acc[label] = acc
            status = "✅" if acc > 0.8 else "⚠️" if acc > 0.6 else "❌"
            print(f"{status} {label:<25} {acc*100:>6.2f}%")
    
    print("="*50 + "\n")
    
    return per_class_acc


def plot_training_history(history_stage1, history_stage2, save_path: str = None):
    """
    Plot training vs validation accuracy and loss curves.
    Shows overfitting analysis clearly.
    
    Args:
        history_stage1: Training history from Stage 1
        history_stage2: Training history from Stage 2
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History & Overfitting Analysis', fontsize=16, fontweight='bold', y=1.00)
    
    # ==================== STAGE 1: ACCURACY ====================
    ax = axes[0, 0]
    ax.plot(history_stage1.history['accuracy'], 'b-', linewidth=2, label='Training Accuracy')
    ax.plot(history_stage1.history['val_accuracy'], 'r-', linewidth=2, label='Validation Accuracy')
    ax.set_title('Stage 1: Accuracy (Frozen Base)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add overfitting indicator
    final_train = history_stage1.history['accuracy'][-1]
    final_val = history_stage1.history['val_accuracy'][-1]
    gap = final_train - final_val
    status = "Good" if gap < 0.1 else "Moderate" if gap < 0.2 else "High Overfitting"
    ax.text(0.02, 0.98, f"Overfitting: {status} (Gap: {gap:.3f})", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== STAGE 1: LOSS ====================
    ax = axes[0, 1]
    ax.plot(history_stage1.history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax.plot(history_stage1.history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax.set_title('Stage 1: Loss (Frozen Base)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # ==================== STAGE 2: ACCURACY ====================
    ax = axes[1, 0]
    offset = len(history_stage1.history['accuracy'])
    epochs_s2 = range(len(history_stage2.history['accuracy']))
    
    ax.plot(epochs_s2, history_stage2.history['accuracy'], 'g-', linewidth=2, label='Training Accuracy')
    ax.plot(epochs_s2, history_stage2.history['val_accuracy'], 'orange', linewidth=2, label='Validation Accuracy')
    ax.set_title('Stage 2: Accuracy (Fine-tuned Base)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add overfitting indicator
    final_train = history_stage2.history['accuracy'][-1]
    final_val = history_stage2.history['val_accuracy'][-1]
    gap = final_train - final_val
    status = "Good" if gap < 0.1 else "Moderate" if gap < 0.2 else "High Overfitting"
    ax.text(0.02, 0.98, f"Overfitting: {status} (Gap: {gap:.3f})", 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ==================== STAGE 2: LOSS ====================
    ax = axes[1, 1]
    ax.plot(epochs_s2, history_stage2.history['loss'], 'g-', linewidth=2, label='Training Loss')
    ax.plot(epochs_s2, history_stage2.history['val_loss'], 'orange', linewidth=2, label='Validation Loss')
    ax.set_title('Stage 2: Loss (Fine-tuned Base)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to: {save_path}")
    
    plt.show()
    
    # Print interpretation guide
    print("\n" + "="*70)
    print("OVERFITTING INTERPRETATION GUIDE")
    print("="*70)
    print("""
    📊 How to Read the Graphs:
    
    ✅ GOOD FIT (Small Gap):
       - Training and validation curves follow closely
       - Gap between curves < 0.1 (10%)
       - Model generalizes well
    
    ⚠️  MODERATE OVERFITTING (Medium Gap):
       - Training curve above validation by 10-20%
       - Model is learning dataset-specific patterns
       - May need more regularization or data
    
    ❌ HIGH OVERFITTING (Large Gap):
       - Training curve significantly above validation (>20% gap)
       - Model memorizing training data, not generalizing
       - Validation curves plateau while training continues
    
    🎯 Key Points:
       - Validation loss should eventually level off
       - If validation loss increases while training loss decreases = overfitting
       - Dropout + Data Augmentation help reduce overfitting
    """)
    print("="*70)


def generate_evaluation_report(model, X_val, y_val, history_stage1, history_stage2, 
                               label_encoder: LabelEncoder, output_dir: str = "output"):
    """
    Generate complete evaluation report with all metrics and visualizations.
    
    Args:
        model: Trained Keras model
        X_val: Validation images
        y_val: Validation labels (encoded)
        history_stage1: Training history from Stage 1
        history_stage2: Training history from Stage 2
        label_encoder: LabelEncoder for labels
        output_dir: Directory to save visualizations
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Compute predictions
    print("\n[1/4] Computing predictions...")
    y_true, y_pred, y_pred_labels, confidence = compute_predictions(model, X_val, y_val, label_encoder)
    
    # 2. Generate confusion matrix
    print("\n[2/4] Generating confusion matrix...")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    confusion_matrix_result = generate_confusion_matrix(y_true, y_pred, label_encoder, cm_path)
    
    # 3. Classification report
    print("\n[3/4] Generating classification report...")
    classification_report_result = generate_classification_report(y_true, y_pred, label_encoder)
    
    # 4. Per-class accuracy
    print("\n[4/4] Computing per-class accuracy...")
    per_class_accuracy = compute_per_class_accuracy(y_true, y_pred, label_encoder)
    
    # 5. Training history plots
    print("\nGenerating training history plots...")
    history_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(history_stage1, history_stage2, history_path)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
    print(f"📁 Output saved to: {output_dir}/")
    
    return {
        'confusion_matrix': confusion_matrix_result,
        'classification_report': classification_report_result,
        'per_class_accuracy': per_class_accuracy,
        'confidence_scores': confidence
    }
