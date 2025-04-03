# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

import matplotlib.pyplot as plt

def plot_training_history(history, figsize=(6, 2.5)):
    """
    Plot training and validation metrics/loss over epochs based on available metrics in history.
    
    Overfitting:
     It happens when a model learns training data too well, including noise and specific patterns, leading to poor
     performance on unseen data. It often occurs when training metrics improve significantly while validation
     metrics plateau or worsen, showing the model memorizes rather than generalizes.
    - Signs of Overfitting:
      - Training metrics improve, but validation metrics plateau or decline.
      - A large gap between training and validation performance.
      - Strong performance on training data but poor results on test data.
    - Causes:
      - Complex Model: Too many parameters (e.g., deep U-Net) with a small dataset.
      - Small Dataset: Limited data leads to memorization instead of generalization.
      - Lack of Regularization: Missing techniques like dropout, weight decay, or data augmentation. 

    Underfitting:
      It happens when a model fails to learn the training data, leading to poor performance on both training
      and validation/test sets. This can occur if the model is too simple or insufficiently trained to capture the
      data’s complexity. 
    - Signs of Underfitting:
        - Low training and validation metrics that don’t improve significantly.
        - Training metrics improve slowly or plateau at suboptimal values.
        - Poor performance on the training set itself.
    - Causes: 
        - Simple Model: Insufficient capacity (e.g., too few layers or filters).
        - Insufficient Training: Too few epochs or a high learning rate.
        - Small/Imbalanced Dataset: Lack of diversity limits generalizable learning. 

    Precision:
    - It measures the proportion of predicted positive instances (foreground pixels in your case) that are actually correct.
    - High precision means the model is good at avoiding false positives—when it predicts a pixel as foreground, it’s usually correct.
    - Low precision indicates the model is predicting too many false positives (e.g., predicting background pixels as foreground).
    - True Positives (TP): Pixels correctly predicted as foreground (1).
    - False Positives (FP): Pixels incorrectly predicted as foreground (predicted 1, but actually 0). 
      
    Recall (also called sensitivity):
    - It measures the proportion of actual positive instances (foreground pixels) that the model correctly identifies.
    - High recall means the model is good at detecting most of the foreground pixels, minimizing false negatives.
    - Low recall indicates the model is missing many foreground pixels (e.g., predicting them as background).
    - False Negatives (FN): Pixels that are actually foreground (1) but predicted as background (0).
      
    Args:
    - history: History object from model.fit()
    """
    # Get metrics with both training and validation versions
    metrics_to_plot = [m for m in history.history if not m.startswith('val_') and f'val_{m}' in history.history]
   
    # If no metrics found, raise an error
    if not metrics_to_plot: raise ValueError("No valid metrics found in history to plot.")
    
    # Define epochs range
    epochs = range(1, len(history.history['loss']) + 1)

    # Create subplots dynamically based on number of metrics
    n_metrics = len(metrics_to_plot)
    n_rows = (n_metrics + 1) // 2  # Ceiling division to determine rows
    n_cols = min(n_metrics, 2)     # Max 2 columns
    
    plt.figure(figsize=figsize)  # Adjust height based on number of rows
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(n_rows, n_cols, i)
        plt.plot(epochs, history.history[metric], label=f'Training {metric}')
        plt.plot(epochs, history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.ylabel(metric.capitalize(), fontsize=10)
        if i == 1: plt.legend(fontsize=9)
        if i == n_metrics - 1 or i == n_metrics: plt.xlabel('Epoch', fontsize=10)
        plt.grid(True)
  
    plt.tight_layout()
    plt.show()