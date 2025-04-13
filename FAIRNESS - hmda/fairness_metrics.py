"""
Fairness metrics calculation module
"""

import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

class FairnessMetrics:
    """
    Class for calculating fairness metrics using AIF360
    """
    
    def __init__(self, protected_attribute='applicant_sex', privileged_value=1, unprivileged_value=2):
        """
        Initialize fairness metrics calculator
        
        Args:
            protected_attribute: Name of protected attribute column
            privileged_value: Value representing privileged group
            unprivileged_value: Value representing unprivileged group
        """
        self.protected_attribute = protected_attribute
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.privileged_groups = [{protected_attribute: privileged_value}]
        self.unprivileged_groups = [{protected_attribute: unprivileged_value}]
    
    def create_aif_dataset(self, df, label_column, favorable_label=1, unfavorable_label=0):
        """
        Create an AIF360 dataset from pandas DataFrame
        
        Args:
            df: Pandas DataFrame
            label_column: Name of target column
            favorable_label: Value considered favorable outcome (typically 1)
            unfavorable_label: Value considered unfavorable outcome (typically 0)
            
        Returns:
            AIF360 BinaryLabelDataset
        """
        # Ensure protected attribute is in the DataFrame
        if self.protected_attribute not in df.columns:
            raise ValueError(f"Protected attribute '{self.protected_attribute}' not in DataFrame")
            
        # Ensure target column is in the DataFrame
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not in DataFrame")
            
        # Create a copy to avoid modifying original
        df_copy = df.copy()
        
        # Ensure protected attribute and label are numeric
        df_copy[self.protected_attribute] = df_copy[self.protected_attribute].astype(int)
        df_copy[label_column] = df_copy[label_column].astype(int)
        
        # Create AIF360 dataset
        try:
            dataset = BinaryLabelDataset(
                df=df_copy,
                label_names=[label_column],
                protected_attribute_names=[self.protected_attribute],
                favorable_label=favorable_label,
                unfavorable_label=unfavorable_label
            )
            return dataset
        except Exception as e:
            print(f"Error creating AIF360 dataset: {e}")
            raise
    
    def create_prediction_dataset(self, dataset_true, y_pred):
        """
        Create dataset with predictions
        
        Args:
            dataset_true: AIF360 dataset with true structure
            y_pred: Numpy array of predictions
            
        Returns:
            AIF360 dataset with predictions
        """
        # Create a copy of the original dataset
        pred_dataset = dataset_true.copy()
        
        # Ensure y_pred is properly shaped
        y_pred_reshaped = np.asarray(y_pred).reshape(-1, 1)
        
        # Set predictions as labels
        pred_dataset.labels = y_pred_reshaped
        
        return pred_dataset
    
    def evaluate_fairness(self, dataset_true, dataset_pred):
        """
        Calculate fairness metrics
        
        Args:
            dataset_true: AIF360 dataset with true labels
            dataset_pred: AIF360 dataset with predicted labels
            
        Returns:
            Dictionary of fairness and classification metrics
        """
        try:
            # Create classification metric
            metric = ClassificationMetric(
                dataset_true, dataset_pred,
                privileged_groups=self.privileged_groups,
                unprivileged_groups=self.unprivileged_groups
            )
            
            # Calculate basic classification metrics
            results = {
                'accuracy': metric.accuracy(),
                'balanced_accuracy': 0.5 * (metric.true_positive_rate() + metric.true_negative_rate()),
                'precision': metric.precision(),
                'recall': metric.recall(),
                'f1_score': 2 * metric.precision() * metric.recall() / max(metric.precision() + metric.recall(), 1e-10),
            }
            
            # Add confusion matrix components
            results['true_positive'] = int(metric.true_positive_rate() * metric.num_positives())
            results['true_negative'] = int(metric.true_negative_rate() * metric.num_negatives())
            results['false_positive'] = int(metric.false_positive_rate() * metric.num_negatives())
            results['false_negative'] = int(metric.false_negative_rate() * metric.num_positives())
            
            # Add fairness metrics with error handling
            try:
                results['disparate_impact'] = metric.disparate_impact()
            except:
                results['disparate_impact'] = np.nan
                
            try:
                results['statistical_parity_difference'] = metric.statistical_parity_difference()
            except:
                results['statistical_parity_difference'] = np.nan
                
            try:
                results['equal_opportunity_difference'] = metric.equal_opportunity_difference()
            except:
                results['equal_opportunity_difference'] = np.nan
                
            try:
                results['average_odds_difference'] = metric.average_odds_difference()
            except:
                results['average_odds_difference'] = np.nan
                
            return results
            
        except Exception as e:
            print(f"Error calculating fairness metrics: {e}")
            # Return NaNs for all metrics on error
            return {
                'accuracy': np.nan,
                'balanced_accuracy': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'f1_score': np.nan,
                'true_positive': 0,
                'true_negative': 0,
                'false_positive': 0,
                'false_negative': 0,
                'disparate_impact': np.nan,
                'statistical_parity_difference': np.nan,
                'equal_opportunity_difference': np.nan,
                'average_odds_difference': np.nan
            }
    
    def explain_metrics(self):
        """
        Return explanation of fairness metrics
        
        Returns:
            Dictionary with metric explanations
        """
        return {
            'disparate_impact': 'Ratio of favorable outcome rate for unprivileged group to that of privileged group. Ideal value is 1.0.',
            'statistical_parity_difference': 'Difference in probability of favorable outcome between unprivileged and privileged groups. Ideal value is 0.0.',
            'equal_opportunity_difference': 'Difference in true positive rates between unprivileged and privileged groups. Ideal value is 0.0.',
            'average_odds_difference': 'Average of difference in FPR and TPR between unprivileged and privileged groups. Ideal value is 0.0.',
            'accuracy': 'Fraction of correctly classified instances',
            'balanced_accuracy': 'Average of true positive rate and true negative rate',
            'precision': 'Fraction of positive predictions that are correct',
            'recall': 'Fraction of actual positives correctly identified',
            'f1_score': 'Harmonic mean of precision and recall'
        }
    
    def print_metrics_explanation(self):
        """Print explanations of all metrics"""
        explanations = self.explain_metrics()
        print("\nFairness Metrics Explanation:")
        print("=" * 50)
        print("\nFairness Metrics:")
        for metric in ['disparate_impact', 'statistical_parity_difference', 'equal_opportunity_difference', 'average_odds_difference']:
            print(f"• {metric}: {explanations[metric]}")
        
        print("\nClassification Metrics:")
        for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1_score']:
            print(f"• {metric}: {explanations[metric]}")