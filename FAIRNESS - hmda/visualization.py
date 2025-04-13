"""
Visualization tools for fairness metrics analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import warnings
import os

class FairnessVisualizer:
    """Class for creating visualizations of fairness metrics"""
    
    def __init__(self, style='darkgrid'):
        """
        Initialize the visualizer with styling settings
        
        Args:
            style: Seaborn style ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks')
        """
        # Set plot style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['figure.dpi'] = 100
        
        # Use a visually appealing color palette
        self.colors = sns.color_palette("viridis", 10)
        
        # Suppress matplotlib warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    
    def create_bar_chart(self, comparison_df, metric, title=None, save_path=None):
        """
        Create a bar chart for a specific metric
        
        Args:
            comparison_df: DataFrame with model comparison data
            metric: Column name to plot
            title: Chart title (optional)
            save_path: Path to save the figure (optional)
            
        Returns:
            The matplotlib figure
        """
        # Check if the metric exists in the dataframe
        if metric not in comparison_df.columns:
            print(f"Warning: Metric '{metric}' not found in data")
            return None
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar chart for better readability of model names
        ax = sns.barplot(x=metric, y='Model', data=comparison_df, palette='viridis')
        
        # Add reference lines for fairness metrics
        if metric == 'Disparate Impact':
            plt.axvline(x=1.0, color='red', linestyle='--', label='Fair value (1.0)')
            plt.axvspan(0.8, 1.2, alpha=0.2, color='green', label='Fair zone (0.8-1.2)')
            
        elif 'Difference' in metric:
            plt.axvline(x=0.0, color='red', linestyle='--', label='Fair value (0.0)')
            plt.axvspan(-0.1, 0.1, alpha=0.2, color='green', label='Fair zone (±0.1)')
        
        # Add value labels to the bars
        for i, v in enumerate(comparison_df[metric]):
            plt.text(v + 0.01, i, f'{v:.3f}', va='center')
            
        # Set title and labels
        if title:
            plt.title(title)
        else:
            plt.title(f'{metric} Comparison')
            
        plt.xlabel(metric)
        plt.ylabel('Model')
        plt.legend()
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
    
    def create_fairness_tradeoff(self, comparison_df, save_path=None):
        """
        Create a scatter plot showing fairness vs performance tradeoff
        
        Args:
            comparison_df: DataFrame with model comparison data
            save_path: Path to save the figure (optional)
            
        Returns:
            The matplotlib figure
        """
        # Check required columns
        required_cols = ['Disparate Impact', 'Accuracy']
        missing = [col for col in required_cols if col not in comparison_df.columns]
        if missing:
            print(f"Warning: Missing columns for tradeoff plot: {missing}")
            return None
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        ax = sns.scatterplot(
            x='Disparate Impact',
            y='Accuracy',
            hue='Model',
            s=150,  # Larger points
            data=comparison_df
        )
        
        # Add fairness reference line and zone
        plt.axvline(x=1.0, color='red', linestyle='--', label='Fair DI (1.0)')
        plt.axvspan(0.8, 1.2, alpha=0.1, color='green', label='Fair zone')
        
        # Add annotations for each model
        for i, row in comparison_df.iterrows():
            plt.annotate(
                row['Model'],
                (row['Disparate Impact'], row['Accuracy']),
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        # Set title and labels
        plt.title('Fairness vs Performance Tradeoff')
        plt.xlabel('Disparate Impact (1.0 = Fair)')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Models')
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
    
    def create_heatmap(self, comparison_df, save_path=None):
        """
        Create a heatmap of all metrics
        
        Args:
            comparison_df: DataFrame with model comparison data
            save_path: Path to save the figure (optional)
            
        Returns:
            The matplotlib figure
        """
        if len(comparison_df) == 0:
            print("Warning: Empty data for heatmap")
            return None
            
        # Set Model as index
        heatmap_df = comparison_df.set_index('Model') if 'Model' in comparison_df.columns else comparison_df
        
        # Get numeric columns only
        numeric_cols = heatmap_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            print("Warning: No numeric columns for heatmap")
            return None
            
        heatmap_df = heatmap_df[numeric_cols]
        
        # Create normalized DataFrame for coloring
        normalized_df = pd.DataFrame()
        for col in heatmap_df.columns:
            if col == 'Disparate Impact':
                # For DI, 1.0 is ideal
                normalized_df[col] = 1 - abs(heatmap_df[col] - 1.0)
            elif 'Difference' in col:
                # For difference metrics, 0.0 is ideal
                normalized_df[col] = 1 - abs(heatmap_df[col])
            else:
                # For other metrics, higher is better
                min_val, max_val = heatmap_df[col].min(), heatmap_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (heatmap_df[col] - min_val) / (max_val - min_val)
                else:
                    normalized_df[col] = heatmap_df[col]
        
        # Create figure
        plt.figure(figsize=(12, len(heatmap_df) * 1.2))
        
        # Red-Yellow-Green colormap
        cmap = LinearSegmentedColormap.from_list(
            "custom_cmap",
            ["#ff9999", "#ffff99", "#99ff99"]
        )
        
        # Create heatmap
        ax = sns.heatmap(
            normalized_df,
            annot=heatmap_df,
            fmt='.3f',
            cmap=cmap,
            linewidths=0.5,
            cbar_kws={'label': 'Normalized Score'}
        )
        
        plt.title('Metrics Comparison Across Models')
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
    
    def create_confusion_matrix(self, y_true, y_pred, model_name="Model", save_path=None):
        """
        Create a confusion matrix visualization
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Path to save the figure (optional)
            
        Returns:
            The matplotlib figure
        """
        from sklearn.metrics import confusion_matrix
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            xticklabels=['Denied (0)', 'Approved (1)'],
            yticklabels=['Denied (0)', 'Approved (1)']
        )
        
        # Add labels
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        
        # Extract values from confusion matrix
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Calculate metrics
            total = tn + fp + fn + tp
            accuracy = (tp + tn) / total if total > 0 else 0
            
            # Add metrics text
            metrics_text = (
                f"Accuracy: {accuracy:.3f}\n"
                f"True Negatives (TN): {tn}\n"
                f"False Positives (FP): {fp}\n"
                f"False Negatives (FN): {fn}\n"
                f"True Positives (TP): {tp}"
            )
            
            plt.figtext(0.6, 0.2, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
        
    def create_all_visualizations(self, comparison_df, output_dir=None, prefix=""):
        """
        Create and save all visualizations
        
        Args:
            comparison_df: DataFrame with model comparison data
            output_dir: Directory to save figures (optional)
            prefix: Prefix for filenames (optional)
            
        Returns:
            Dictionary of created figures
        """
        figures = {}
        
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Bar charts for key metrics
        for metric in ['Disparate Impact', 'Equal Opportunity Difference', 'Accuracy']:
            if metric in comparison_df.columns:
                save_path = os.path.join(output_dir, f"{prefix}{metric.replace(' ', '_')}.png") if output_dir else None
                figures[f"{metric}_bar"] = self.create_bar_chart(comparison_df, metric, save_path=save_path)
        
        # Fairness tradeoff
        save_path = os.path.join(output_dir, f"{prefix}fairness_tradeoff.png") if output_dir else None
        figures["tradeoff"] = self.create_fairness_tradeoff(comparison_df, save_path=save_path)
        
        # Heatmap
        save_path = os.path.join(output_dir, f"{prefix}metrics_heatmap.png") if output_dir else None
        figures["heatmap"] = self.create_heatmap(comparison_df, save_path=save_path)
        
        return figures
        
    def show_plots(self):
        """Show all current plots"""
        plt.show()