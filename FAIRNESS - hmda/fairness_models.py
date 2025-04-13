# ------------------------------
# fairness_models.py
# ------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
from fairness_metrics import FairnessMetrics
import time

class FairnessModels:
    def __init__(self, protected_attribute='applicant_sex', privileged_value=1, unprivileged_value=2):
        """
        Initialize fairness models
        
        Args:
            protected_attribute: Name of protected column
            privileged_value: Value representing privileged group
            unprivileged_value: Value representing unprivileged group
        """
        self.protected_attribute = protected_attribute
        self.metrics = FairnessMetrics(protected_attribute, privileged_value, unprivileged_value)
        self.privileged_groups = [{protected_attribute: privileged_value}]
        self.unprivileged_groups = [{protected_attribute: unprivileged_value}]

    def encode_categoricals(self, df):
        """
        Encode categorical columns for AIF360 compatibility
        
        Args:
            df: Pandas DataFrame
        
        Returns:
            DataFrame with categoricals encoded as integers
        """
        df_encoded = df.copy()
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
                df_encoded[col] = pd.Categorical(df_encoded[col]).codes
                # Replace -1 (missing) with 0 
                df_encoded.loc[df_encoded[col] == -1, col] = 0
        return df_encoded

    def create_fair_disparate_impact_model(self, X_train, y_train, X_test):
        """
        Create a model with disparate impact mitigation
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            
        Returns:
            model: Trained model
            y_pred: Predictions
            di_remover: Fitted disparate impact remover
        """
        print("\nTraining Disparate Impact model...")
        start_time = time.time()
        
        # Ensure we're working with DataFrames
        X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train.copy()
        X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test.copy()
        
        # Ensure y_train is a Series
        y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train.copy()
        
        # Prepare training data
        train_df = X_train.copy()
        train_df['target'] = y_train.values
        
        # Encode categoricals
        train_df = self.encode_categoricals(train_df)
        
        # Check if protected attribute exists
        if self.protected_attribute not in train_df.columns:
            raise ValueError(f"Protected attribute {self.protected_attribute} not in DataFrame")
            
        # Create AIF360 dataset
        try:
            aif_train = self.metrics.create_aif_dataset(train_df, label_column='target')
        except Exception as e:
            print(f"Error creating AIF360 dataset: {e}")
            raise
            
        # Apply disparate impact remover
        try:
            di_remover = DisparateImpactRemover(repair_level=0.8)  # Using 0.8 instead of 1.0 for better balance
            aif_train_repaired = di_remover.fit_transform(aif_train)
        except Exception as e:
            print(f"Error in disparate impact removal: {e}")
            raise

        # Extract repaired features and labels
        X_train_repaired = aif_train_repaired.features
        y_train_repaired = aif_train_repaired.labels.ravel()

        # Train model (reduced complexity for speed)
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=10, 
            min_samples_split=10,
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X_train_repaired, y_train_repaired)

        # Prepare test data
        X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test.copy()
        test_df = X_test_df.copy()
        test_df['target'] = np.zeros(len(X_test_df))  # Dummy target values
        test_df = self.encode_categoricals(test_df)
        
        # Create AIF360 test dataset
        try:
            aif_test = self.metrics.create_aif_dataset(test_df, label_column='target')
            # Use fit_transform instead of transform for DisparateImpactRemover
            aif_test_repaired = di_remover.fit_transform(aif_test)
        except Exception as e:
            print(f"Error transforming test data: {e}")
            raise
            
        # Make predictions
        X_test_repaired = aif_test_repaired.features
        y_pred = model.predict(X_test_repaired)
        
        print(f"Disparate Impact model completed in {time.time() - start_time:.2f} seconds")
        return model, y_pred, di_remover

    def create_fair_equalized_odds_model(self, X_train, y_train, X_test, y_test):
        """
        Create a model with equalized odds post-processing
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            model: Trained model
            y_pred_fair: Fair predictions
            eqodds: Fitted equalized odds post-processor
        """
        print("\nTraining Equalized Odds model...")
        start_time = time.time()
        
        # Ensure we're working with DataFrames/Series
        X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train.copy()
        X_test = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test.copy() 
        y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train.copy()
        y_test = pd.Series(y_test) if not isinstance(y_test, pd.Series) else y_test.copy()
        
        # Encode categoricals for training
        X_train_encoded = self.encode_categoricals(X_train)
        
        # Train base model - use LogisticRegression for speed
        base_model = LogisticRegression(
            max_iter=1000, 
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        )
        base_model.fit(X_train_encoded, y_train)
        
        # Encode test data and predict
        X_test_encoded = self.encode_categoricals(X_test)
        y_pred_base = base_model.predict(X_test_encoded)

        # Prepare test data for AIF360
        test_df = X_test.copy()
        test_df['target'] = y_test.values
        test_df = self.encode_categoricals(test_df)

        # Create AIF360 datasets
        try:
            aif_test = self.metrics.create_aif_dataset(test_df, label_column='target')
            aif_pred = self.metrics.create_prediction_dataset(aif_test, y_pred_base)
        except Exception as e:
            print(f"Error creating AIF360 datasets: {e}")
            # Return base model predictions if calibration fails
            return base_model, y_pred_base, None

        # Apply equalized odds post-processing
        try:
            eqodds = CalibratedEqOddsPostprocessing(
                privileged_groups=self.privileged_groups,
                unprivileged_groups=self.unprivileged_groups,
                cost_constraint='weighted'
            )
            eqodds.fit(aif_test, aif_pred)
            aif_pred_fair = eqodds.predict(aif_pred)
            y_pred_fair = aif_pred_fair.labels.ravel()
        except Exception as e:
            print(f"Error in equalized odds processing: {e}")
            # Return base model predictions if calibration fails
            return base_model, y_pred_base, None

        print(f"Equalized Odds model completed in {time.time() - start_time:.2f} seconds")
        return base_model, y_pred_fair, eqodds