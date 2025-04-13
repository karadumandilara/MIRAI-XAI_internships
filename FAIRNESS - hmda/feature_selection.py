import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import time

class HMDAFeatureSelector:
    """
    Fast feature selector using ExtraTreesClassifier
    """
    def __init__(self):
        self.feature_importances = None
        self.selected_features = None
        self.model = None
        
    def _prepare_data(self, X):
        """Prepare data for feature selection by encoding categoricals"""
        X_prepared = X.copy()
        
        # Fast categorical encoding
        for col in X_prepared.select_dtypes(include=['object', 'category']).columns:
            # Handle potential errors when trying to convert
            try:
                X_prepared[col] = pd.Categorical(X_prepared[col]).codes
                # Replace -1 (missing) with 0
                X_prepared.loc[X_prepared[col] == -1, col] = 0
            except:
                # If conversion fails, drop the column
                print(f"Warning: Could not encode column {col}, dropping it")
                X_prepared = X_prepared.drop(columns=[col])
                
        return X_prepared

    def select_features(self, df, target_col='target', protected_col='applicant_sex', n_features=30):
        """
        Select the most important features using ExtraTrees
        
        Args:
            df: Pandas DataFrame with both features and target
            target_col: Name of the target column
            protected_col: Name of the protected attribute
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        start_time = time.time()
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Prepare data - encode categoricals
        X_prepared = self._prepare_data(X)
        
        # Use ExtraTrees for fast feature importance (much faster than RandomForest)
        self.model = ExtraTreesClassifier(
            n_estimators=50,  # Reduced from original 100 for speed
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1,  # Use all cores
            class_weight='balanced'  # Handle class imbalance
        )
        
        # Fit the model to get feature importances
        try:
            self.model.fit(X_prepared, y)
            self.feature_importances = self.model.feature_importances_
        except Exception as e:
            print(f"Error in feature selection: {e}")
            # Return all features if selection fails
            return X.columns.tolist()

        # Create feature importance DataFrame
        feat_df = pd.DataFrame({
            'feature': X.columns,
            'importance': self.feature_importances
        }).sort_values(by='importance', ascending=False)

        # Select top n features
        selected_features = feat_df.head(n_features)['feature'].tolist()

        # Always include protected attribute
        if protected_col not in selected_features and protected_col in X.columns:
            selected_features.append(protected_col)

        self.selected_features = selected_features
        
        print(f"Feature selection completed in {time.time() - start_time:.2f} seconds")
        print(f"Selected {len(selected_features)} features")
        
        return selected_features

    def get_feature_importance_df(self):
        """
        Returns a DataFrame with feature importances
        """
        if self.feature_importances is None or self.model is None:
            raise ValueError("Call select_features() first")
            
        try:
            return pd.DataFrame({
                'Feature': self.selected_features,
                'Importance': [self.feature_importances[list(self.model.feature_names_in_).index(f)] 
                              if f in self.model.feature_names_in_ else 0 
                              for f in self.selected_features]
            }).sort_values(by='Importance', ascending=False)
        except:
            # Fallback if there's an error in the fancy approach
            return pd.DataFrame({
                'Feature': self.selected_features,
                'Importance': np.ones(len(self.selected_features))
            })