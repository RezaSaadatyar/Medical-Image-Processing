import numpy as np
import matplotlib.pyplot as plt

def plot_test_results(results):
    """
    Plot test set evaluation results as a bar chart based on available metrics in results.
    
    Args:
    - results: Dictionary from model.evaluate(return_dict=True)
    """
    # Extract all metrics from the results dictionary
    metrics_to_plot = list(results.keys())
    
    # If no metrics found, raise an error
    if not metrics_to_plot: raise ValueError("No metrics found in results to plot.")
    
    # Create bar plot with dynamic width and assigned colors
    plt.figure(figsize=(max(4, len(metrics_to_plot) * 0.8), 3))
    
    # Get corresponding values
    values = [results[metric] for metric in metrics_to_plot]
    
    # Define a color map and assign colors to each bar
    colors = plt.cm.summer(np.linspace(0, 1, len(metrics_to_plot)))
    bars = plt.bar(metrics_to_plot, values, color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3f}', 
                 ha='center', va='bottom' if yval < 0.5 else 'top', color='black', rotation=0)
    
    plt.margins(x=0.1)
    plt.ylabel('Value', fontsize=10)
    plt.xticks(rotation=15, ha='right')
    plt.ylim(0, max(1.0, max(values) * 1.1))  # Adjust y-axis for visibility
    plt.xlabel('Metrics', fontsize=10, labelpad=5)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.title('Test Set Evaluation Metrics', fontsize=10)
    plt.tight_layout()
    plt.show()