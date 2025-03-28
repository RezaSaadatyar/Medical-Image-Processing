import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plot training and validation metrics/loss over epochs based on available metrics in history.
    
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
    
    plt.figure(figsize=(8, 1.8 * n_rows))  # Adjust height based on number of rows
    
    for i, metric in enumerate(metrics_to_plot, 1):
        plt.subplot(n_rows, n_cols, i)
        plt.plot(epochs, history.history[metric], label=f'Training {metric}')
        plt.plot(epochs, history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.ylabel(metric.capitalize(), fontsize=10)
        if i == 1: plt.legend(fontsize=9)
        plt.grid(True)
    
    # Add xlabel only to the bottom row
    for i in range(n_rows * n_cols - n_cols + 1, n_rows * n_cols + 1):
        plt.subplot(n_rows, n_cols, i)
        plt.xlabel('Epoch', fontsize=10)
    
    plt.tight_layout()
    plt.show()