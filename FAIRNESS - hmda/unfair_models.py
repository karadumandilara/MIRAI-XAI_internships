# ------------------------------
# unfair_models.py
# ------------------------------

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score
import time

class UnfairModels:
    """
    Class for training baseline unfair models
    """
    def __init__(self):
        """Initialize unfair model class"""
        pass
        
    def train_random_forest(self, X_train, y_train, X_test, n_estimators=100, max_depth=None):
        """
        Train a RandomForest classifier without fairness constraints
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            
        Returns:
            model: Trained model
            y_pred: Predictions on test set
        """
        print("\nTraining unfair RandomForest model...")
        start_time = time.time()
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Convert to numpy arrays for speed if needed
        if hasattr(X_train, 'values'):
            X_train_array = X_train.values
        else:
            X_train_array = X_train
            
        if hasattr(y_train, 'values'):
            y_train_array = y_train.values
        else:
            y_train_array = y_train
        
        # Fit model
        model.fit(X_train_array, y_train_array)
        
        # Make predictions
        if hasattr(X_test, 'values'):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
            
        y_pred = model.predict(X_test_array)
        
        print(f"RandomForest model completed in {time.time() - start_time:.2f} seconds")
        return model, y_pred
        
    def train_logistic_regression(self, X_train, y_train, X_test):
        """
        Train a LogisticRegression classifier without fairness constraints
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            
        Returns:
            model: Trained model
            y_pred: Predictions on test set
        """
        print("\nTraining unfair LogisticRegression model...")
        start_time = time.time()
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Convert to numpy arrays for speed if needed
        if hasattr(X_train, 'values'):
            X_train_array = X_train.values
        else:
            X_train_array = X_train
            
        if hasattr(y_train, 'values'):
            y_train_array = y_train.values
        else:
            y_train_array = y_train
        
        # Fit model
        model.fit(X_train_array, y_train_array)
        
        # Make predictions
        if hasattr(X_test, 'values'):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
            
        y_pred = model.predict(X_test_array)
        
        print(f"LogisticRegression model completed in {time.time() - start_time:.2f} seconds")
        return model, y_pred
    
    def evaluate_model(self, y_true, y_pred):
        """
        Calculate standard ML metrics (without fairness)
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        # Convert to numpy arrays if needed
        if hasattr(y_true, 'values'):
            y_true_array = y_true.values
        else:
            y_true_array = y_true
            
        if hasattr(y_pred, 'values'):
            y_pred_array = y_pred.values
        else:
            y_pred_array = y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true_array, y_pred_array),
            'balanced_accuracy': balanced_accuracy_score(y_true_array, y_pred_array),
            'precision': precision_score(y_true_array, y_pred_array, zero_division=0),
            'recall': recall_score(y_true_array, y_pred_array, zero_division=0)
        }
        
        return metrics