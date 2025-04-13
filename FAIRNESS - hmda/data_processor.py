import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class HMDADataProcessor:
    """
    Efficient HMDA data processing and preparation for fairness-aware ML
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_imputer = SimpleImputer(strategy='median')
        # We'll import feature selector only when needed to avoid circular imports
        self.feature_selector = None

    def process_dataset(self, df):
        """Process the HMDA dataset for modeling"""
        print("Processing dataset...")
        df = df.copy()

        # Drop columns with >95% missing (more efficient)
        missing_threshold = 0.95 * len(df)
        df = df.loc[:, df.isnull().sum() < missing_threshold]

        # Drop unnecessary identifiers
        drop_cols = ['Unnamed: 0', 'lei', 'activity_year']
        df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

        # Keep only approved (1) and denied (3)
        mask = df['action_taken'].isin([1, 3])
        df = df[mask]
        
        # Create target column
        df['target'] = (df['action_taken'] == 1).astype(int)
        df = df.drop(columns=['action_taken'])

        # Keep valid applicant_sex values (1=male, 2=female)
        df = df[df['applicant_sex'].isin([1, 2])]

        # Get column types
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        # Fast imputation for numerics (all at once)
        if len(numeric_cols) > 0:
            df[numeric_cols] = self.num_imputer.fit_transform(df[numeric_cols])

        # Fast imputation for categoricals
        for col in categorical_cols:
            mode_val = df[col].mode()[0] if not df[col].empty else "UNKNOWN"
            df[col].fillna(mode_val, inplace=True)
            
        print(f"Processed dataset shape: {df.shape}")
        return df

    def prepare_data(self, df, target_col='target', protected_col='applicant_sex',
                     test_size=0.3, random_state=42, n_features=30):
        """Prepare data for fairness modeling"""
        # Import here to avoid circular imports
        if self.feature_selector is None:
            from feature_selection import HMDAFeatureSelector
            self.feature_selector = HMDAFeatureSelector()
            
        # Process dataset
        df_processed = self.process_dataset(df)
        
        # Check if we have enough data
        if len(df_processed) < 10:
            raise ValueError(f"Not enough data after processing: {len(df_processed)} rows")
            
        # Check if we have both classes in target
        target_values = df_processed[target_col].value_counts()
        if len(target_values) < 2:
            print(f"Warning: Target column has only one value: {target_values.index[0]}")
            print("Attempting to create variation in target...")
            
            # If we only have one class, try to create artificial variation
            if 'target' in df_processed.columns:
                # Find a numeric column to use for artificial target
                numeric_cols = df_processed.select_dtypes(include=['number']).columns
                numeric_cols = [col for col in numeric_cols if col != protected_col]
                
                if len(numeric_cols) > 0:
                    # Use first numeric column for artificial split
                    artificial_col = numeric_cols[0]
                    median_val = df_processed[artificial_col].median()
                    df_processed['target'] = (df_processed[artificial_col] > median_val).astype(int)
                    print(f"Created artificial target based on {artificial_col} > {median_val}")
                    
                    # Check if we now have two classes
                    if len(df_processed['target'].unique()) < 2:
                        raise ValueError("Could not create variation in target column")
                else:
                    raise ValueError("No numeric columns available for artificial target")
            else:
                raise ValueError("Target column must have at least 2 unique values")
        
        # Feature selection
        print("Selecting features...")
        selected_features = self.feature_selector.select_features(
            df_processed,
            target_col=target_col,
            protected_col=protected_col,
            n_features=n_features
        )

        X = df_processed[selected_features]
        y = df_processed[target_col]
        
        # Save protected attribute values
        protected_values = X[protected_col].copy()

        # Train-test split with stratification
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )

        # Get protected values for test set
        protected_values_test = protected_values.loc[X_test.index]

        # Scale numeric features except protected attribute
        numeric_cols = X.select_dtypes(include='number').columns.tolist()
        if protected_col in numeric_cols:
            numeric_cols.remove(protected_col)

        if len(numeric_cols) > 0:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()
            
            X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        else:
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()

        print(f"Train set: {X_train_scaled.shape}, Test set: {X_test_scaled.shape}")
        return (X_train_scaled, X_test_scaled, y_train, y_test, 
                selected_features, X.columns.tolist(), protected_values_test)