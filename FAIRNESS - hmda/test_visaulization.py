"""
Test script for visualization module
Run this script to see examples of all visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization import FairnessVisualizer

def main():
    # Create sample data for testing
    data = {
        'Model': ['Unfair RandomForest', 'Fair Disparate Impact', 'Fair Equalized Odds', 'Unfair LogisticRegression'],
        'Accuracy': [0.95, 0.92, 0.93, 0.91],
        'Balanced Accuracy': [0.94, 0.91, 0.92, 0.90],
        'Precision': [0.96, 0.93, 0.94, 0.92],
        'Recall': [0.97, 0.94, 0.95, 0.93],
        'Disparate Impact': [0.76, 0.98, 0.95, 0.83],
        'Equal Opportunity Difference': [-0.15, -0.04, -0.02, -0.11],
        'Statistical Parity Difference': [-0.18, -0.05, -0.03, -0.12],
        'Average Odds Difference': [-0.16, -0.05, -0.03, -0.11],
        'True Positive': [800, 780, 790, 770],
        'True Negative': [1200, 1180, 1190, 1170],
        'False Positive': [100, 120, 110, 130],
        'False Negative': [80, 100, 90, 110]
    }
    
    comparison_df = pd.DataFrame(data)
    
    # Simulate confusion matrix data
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1])
    
    print("Testing Fairness Visualization Module")
    print("=" * 40)
    
    # Create visualizer
    visualizer = FairnessVisualizer()
    
    # Test individual visualizations
    print("Creating bar chart for Disparate Impact...")
    fig1 = visualizer.create_bar_chart(comparison_df, 'Disparate Impact', 
                                 title="Disparate Impact Comparison",
                                 save_path="test_disparate_impact.png")
    
    print("Creating fairness vs performance tradeoff plot...")
    fig2 = visualizer.create_fairness_tradeoff(comparison_df, 
                                        save_path="test_fairness_tradeoff.png")
    
    print("Creating metrics heatmap...")
    fig3 = visualizer.create_heatmap(comparison_df, 
                              save_path="test_metrics_heatmap.png")
    
    print("Creating confusion matrix...")
    fig4 = visualizer.create_confusion_matrix(y_true, y_pred, 
                                       model_name="Test Model",
                                       save_path="test_confusion_matrix.png")
    
    print("Creating all visualizations at once...")
    all_figs = visualizer.create_all_visualizations(comparison_df, 
                                           output_dir="./test_visuals",
                                           prefix="test_")
    
    print("=" * 40)
    print("All visualizations created successfully!")
    print("Individual plots saved to current directory")
    print("Complete set saved to ./test_visuals/")
    print("=" * 40)
    
    # Show the plots (comment this out if running headless)
    print("Showing plots... (close windows to continue)")
    visualizer.show_plots()
    
    return 0

if __name__ == "__main__":
    main()