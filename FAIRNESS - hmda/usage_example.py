"""
HMDA Fairness Analysis Pipeline - Structured by data groups and model fairness
"""

import os
import time
import warnings
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Import custom modules
from data_processor import HMDADataProcessor
from fairness_metrics import FairnessMetrics
from visualization import FairnessVisualizer

# Suppress warnings
warnings.filterwarnings('ignore')

def preprocess_dataframe(df):
    """Convert object columns to category codes"""
    df_processed = df.copy()
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = pd.Categorical(df_processed[col]).codes
    return df_processed

def evaluate_model(test_df, y_pred, model_name, visualizer, output_dir=None):
    """
    Evaluate a model using fairness metrics and create visualizations
    
    Args:
        test_df: Test DataFrame with features and target
        y_pred: Model predictions
        model_name: Name of the model
        visualizer: FairnessVisualizer instance
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary of metrics
    """
    # Create a copy of test data
    test_df = test_df.copy()
    
    # Ensure binary classification format (0, 1)
    if 'target' in test_df.columns:
        test_df['target'] = test_df['target'].replace({3: 0, 1: 1}).astype(int)
    
    # Ensure predictions are binary (0, 1)
    y_pred = np.array(y_pred)
    if len(np.unique(y_pred)) > 2:
        y_pred = np.where(y_pred == 1, 1, 0)
    
    # Check if we have both classes and protected groups
    if len(np.unique(test_df['target'])) < 2:
        print(f"⚠️ Skipping evaluation for '{model_name}' - only one class in target")
        return None
        
    if len(np.unique(test_df['applicant_sex'])) < 2:
        print(f"⚠️ Skipping evaluation for '{model_name}' - need both protected groups")
        return None
    
    # Calculate fairness metrics
    try:
        metrics = FairnessMetrics(protected_attribute='applicant_sex')
        aif_test = metrics.create_aif_dataset(test_df, label_column='target')
        aif_pred = metrics.create_prediction_dataset(aif_test, y_pred)
        results = metrics.evaluate_fairness(aif_test, aif_pred)
        
        # Add model name to results
        results['Model'] = model_name
        
        # Print metrics summary
        print(f"\n🔹 Results for {model_name}:")
        print("Classification Metrics:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {results['balanced_accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"  True Positives: {results['true_positive']}")
        print(f"  True Negatives: {results['true_negative']}")
        print(f"  False Positives: {results['false_positive']}")
        print(f"  False Negatives: {results['false_negative']}")
        
        print("\nFairness Metrics:")
        print(f"  Disparate Impact: {results['disparate_impact']:.4f}")
        print(f"  Statistical Parity Diff: {results['statistical_parity_difference']:.4f}")
        print(f"  Equal Opportunity Diff: {results['equal_opportunity_difference']:.4f}")
        print(f"  Average Odds Diff: {results['average_odds_difference']:.4f}")
        
        # Create visualizations
        if visualizer and output_dir:
            model_dir = os.path.join(output_dir, model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_"))
            os.makedirs(model_dir, exist_ok=True)
            
            # Create DataFrame for visualization
            df_viz = pd.DataFrame([results])
            
            # Create and save all visualizations
            visualizer.create_all_visualizations(df_viz, output_dir=model_dir)
            
            # Create confusion matrix visualization
            visualizer.create_confusion_matrix(
                test_df['target'], y_pred, 
                model_name=model_name,
                save_path=os.path.join(model_dir, "confusion_matrix.png")
            )
            
        return results
        
    except Exception as e:
        print(f"⚠️ Error evaluating {model_name}: {e}")
        return None

def train_unfair_models(X_train, X_test, y_train, y_test, data_group, output_dir=None):
    """
    Train and evaluate unfair models
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        data_group: Name of the data group
        output_dir: Directory to save results
        
    Returns:
        List of result dictionaries for unfair models
    """
    print(f"\n🔹 Training Unfair Models for {data_group}...")
    
    # Prepare test dataframe for evaluation
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    # Initialize visualizer
    visualizer = FairnessVisualizer()
    
    # Store unfair model results
    unfair_results = []
    
    # Create model-specific output directory
    if output_dir:
        unfair_dir = os.path.join(output_dir, f"{data_group}/Unfair_Models")
        os.makedirs(unfair_dir, exist_ok=True)
    else:
        unfair_dir = None
    
    # 1. Train Random Forest (Unfair)
    print("\n🔸 Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Evaluate Random Forest
    rf_results = evaluate_model(
        test_df, rf_pred, 
        f"RandomForest/{data_group}", 
        visualizer, unfair_dir
    )
    if rf_results:
        unfair_results.append(rf_results)
    
    # 2. Train Logistic Regression (Unfair)
    print("\n🔸 Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Evaluate Logistic Regression
    lr_results = evaluate_model(
        test_df, lr_pred, 
        f"LogisticRegression/{data_group}", 
        visualizer, unfair_dir
    )
    if lr_results:
        unfair_results.append(lr_results)
    
    # Create comparison visualizations for unfair models
    if len(unfair_results) > 1 and unfair_dir:
        print("\n🔹 Creating unfair model comparison visualizations...")
        
        # Create DataFrame for comparison
        comparison_df = pd.DataFrame(unfair_results)
        
        # Save comparison data
        comparison_df.to_csv(os.path.join(unfair_dir, "unfair_models_comparison.csv"), index=False)
        
        # Create comparison visualizations
        visualizer.create_all_visualizations(
            comparison_df, 
            output_dir=unfair_dir,
            prefix="unfair_comparison_"
        )
    
    return unfair_results

def train_fair_models(X_train, X_test, y_train, y_test, data_group, output_dir=None):
    """
    Train and evaluate fairness-aware models
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        data_group: Name of the data group
        output_dir: Directory to save results
        
    Returns:
        List of result dictionaries for fair models
    """
    # Import here to avoid circular imports
    from fairness_models import FairnessModels
    
    print(f"\n🔹 Training Fair Models for {data_group}...")
    
    # Prepare test dataframe for evaluation
    test_df = X_test.copy()
    test_df['target'] = y_test
    
    # Initialize visualizer
    visualizer = FairnessVisualizer()
    
    # Store fair model results
    fair_results = []
    
    # Create model-specific output directory
    if output_dir:
        fair_dir = os.path.join(output_dir, f"{data_group}/Fair_Models")
        os.makedirs(fair_dir, exist_ok=True)
    else:
        fair_dir = None
    
    # Initialize fairness models
    try:
        fairness_model = FairnessModels(protected_attribute='applicant_sex')
        
        # 1. Disparate Impact Remover
        print("\n🔸 Training Disparate Impact model...")
        try:
            di_model, di_pred, _ = fairness_model.create_fair_disparate_impact_model(
                X_train, y_train, X_test
            )
            
            # Evaluate Disparate Impact model
            di_results = evaluate_model(
                test_df, di_pred, 
                f"DisparateImpact/{data_group}", 
                visualizer, fair_dir
            )
            if di_results:
                fair_results.append(di_results)
                
        except Exception as e:
            print(f"⚠️ Error in Disparate Impact model: {e}")
        
        # 2. Equalized Odds Postprocessor
        print("\n🔸 Training Equalized Odds model...")
        try:
            eq_model, eq_pred, _ = fairness_model.create_fair_equalized_odds_model(
                X_train, y_train, X_test, y_test
            )
            
            # Evaluate Equalized Odds model
            eq_results = evaluate_model(
                test_df, eq_pred, 
                f"EqualizedOdds/{data_group}", 
                visualizer, fair_dir
            )
            if eq_results:
                fair_results.append(eq_results)
                
        except Exception as e:
            print(f"⚠️ Error in Equalized Odds model: {e}")
        
        # Create comparison visualizations for fair models
        if len(fair_results) > 1 and fair_dir:
            print("\n🔹 Creating fair model comparison visualizations...")
            
            # Create DataFrame for comparison
            comparison_df = pd.DataFrame(fair_results)
            
            # Save comparison data
            comparison_df.to_csv(os.path.join(fair_dir, "fair_models_comparison.csv"), index=False)
            
            # Create comparison visualizations
            visualizer.create_all_visualizations(
                comparison_df, 
                output_dir=fair_dir,
                prefix="fair_comparison_"
            )
        
        return fair_results
        
    except Exception as e:
        print(f"⚠️ Error training fair models: {e}")
        return []

def compare_fair_unfair(fair_results, unfair_results, data_group, output_dir=None):
    """
    Create comparisons between fair and unfair models
    
    Args:
        fair_results: Results from fair models
        unfair_results: Results from unfair models
        data_group: Name of the data group
        output_dir: Directory to save results
    """
    if not fair_results or not unfair_results:
        print(f"⚠️ Insufficient results for fair-unfair comparison for {data_group}")
        return
    
    print(f"\n🔹 Comparing Fair vs Unfair Models for {data_group}...")
    
    # Initialize visualizer
    visualizer = FairnessVisualizer()
    
    # Create comparison directory
    if output_dir:
        comparison_dir = os.path.join(output_dir, f"{data_group}/Fair_vs_Unfair")
        os.makedirs(comparison_dir, exist_ok=True)
    else:
        return
    
    # Combine results
    all_results = fair_results + unfair_results
    
    # Create DataFrame for comparison
    comparison_df = pd.DataFrame(all_results)
    
    # Save comparison data
    comparison_df.to_csv(os.path.join(comparison_dir, "fair_vs_unfair_comparison.csv"), index=False)
    
    # Create comparison visualizations
    visualizer.create_all_visualizations(
        comparison_df, 
        output_dir=comparison_dir,
        prefix="fair_vs_unfair_"
    )
    
    # Create specific comparisons for fairness metrics
    for metric in ['Disparate Impact', 'Equal Opportunity Difference', 'Statistical Parity Difference']:
        metric_key = metric.lower().replace(' ', '_')
        if metric_key in comparison_df.columns:
            try:
                fig = visualizer.create_bar_chart(
                    comparison_df, 
                    metric_key, 
                    title=f"{metric} Comparison - {data_group}",
                    save_path=os.path.join(comparison_dir, f"{metric.lower().replace(' ', '_')}_comparison.png")
                )
            except:
                pass

def analyze_data_group(df, group_name, output_dir=None):
    """
    Analyze a specific data group with both fair and unfair models
    
    Args:
        df: DataFrame with the data
        group_name: Name of the data group
        output_dir: Directory to save results
        
    Returns:
        Dictionary with fair and unfair results
    """
    print(f"\n{'='*80}")
    print(f"🔸 Analyzing Data Group: {group_name}")
    print(f"{'='*80}")
    
    # Create group-specific output directory
    if output_dir:
        group_dir = os.path.join(output_dir, group_name.replace(" ", "_"))
        os.makedirs(group_dir, exist_ok=True)
    else:
        group_dir = None
    
    # Initialize processor
    processor = HMDADataProcessor()
    
    try:
        # Process and prepare data
        X_train, X_test, y_train, y_test, _, _, _ = processor.prepare_data(
            df, target_col='target', protected_col='applicant_sex', test_size=0.3
        )
        
        # Preprocess categorical features
        X_train = preprocess_dataframe(X_train)
        X_test = preprocess_dataframe(X_test)
        
        # Train unfair models
        unfair_results = train_unfair_models(
            X_train, X_test, y_train, y_test, 
            group_name, group_dir
        )
        
        # Train fair models
        fair_results = train_fair_models(
            X_train, X_test, y_train, y_test, 
            group_name, group_dir
        )
        
        # Compare fair and unfair models
        compare_fair_unfair(
            fair_results, unfair_results, 
            group_name, group_dir
        )
        
        return {
            'unfair': unfair_results,
            'fair': fair_results
        }
        
    except Exception as e:
        print(f"⚠️ Error analyzing {group_name}: {e}")
        return None

def run_complete_analysis(df, output_dir=None):
    """
    Run complete analysis on all data groups
    
    Args:
        df: HMDA DataFrame
        output_dir: Directory to save results
        
    Returns:
        Dictionary of results for each group
    """
    # Initialize results dictionary
    all_results = {}
    
    # 1. Analyze all data
    print("\n🔹 Analyzing complete dataset...")
    all_results['all_data'] = analyze_data_group(df, "All_Data", output_dir)
    
    # 2. Analyze approved loans with artificial target
    if 'action_taken' in df.columns:
        print("\n🔹 Creating approved loans subset with artificial target...")
        
        # Get approved loans
        df_approved = df[df['action_taken'] == 1].copy()
        
        if len(df_approved) > 100:
            # Create artificial target based on loan amount or other numeric feature
            numeric_cols = df_approved.select_dtypes(include=['number']).columns
            
            # Try to find a suitable feature
            target_col = None
            for col in ['loan_amount', 'income', 'property_value']:
                if col in numeric_cols:
                    target_col = col
                    break
            
            # If no specific feature found, use first numeric column
            if not target_col and len(numeric_cols) > 0:
                for col in numeric_cols:
                    if col not in ['applicant_sex', 'action_taken', 'target']:
                        target_col = col
                        break
            
            if target_col:
                # Create target based on median value
                median_val = df_approved[target_col].median()
                df_approved['target'] = (df_approved[target_col] > median_val).astype(int)
                
                print(f"Created artificial target from {target_col} > {median_val}")
                
                # Check if we have both classes
                if len(df_approved['target'].unique()) > 1:
                    print(f"Target distribution: {df_approved['target'].value_counts().to_dict()}")
                    
                    # Analyze approved loans
                    all_results['approved_loans'] = analyze_data_group(
                        df_approved, "Approved_Loans", output_dir
                    )
                else:
                    print("⚠️ Could not create artificial target with both classes")
            else:
                print("⚠️ No suitable numeric columns found for artificial target")
    
    # 3. Analyze denied loans with artificial target
    if 'action_taken' in df.columns:
        print("\n🔹 Creating denied loans subset with artificial target...")
        
        # Get denied loans
        df_denied = df[df['action_taken'] == 3].copy()
        
        if len(df_denied) > 100:
            # Create artificial target based on numeric feature
            numeric_cols = df_denied.select_dtypes(include=['number']).columns
            
            # Try to find a suitable feature
            target_col = None
            for col in ['loan_amount', 'income', 'property_value']:
                if col in numeric_cols:
                    target_col = col
                    break
            
            # If no specific feature found, use first numeric column
            if not target_col and len(numeric_cols) > 0:
                for col in numeric_cols:
                    if col not in ['applicant_sex', 'action_taken', 'target']:
                        target_col = col
                        break
            
            if target_col:
                # Create target based on median value
                median_val = df_denied[target_col].median()
                df_denied['target'] = (df_denied[target_col] > median_val).astype(int)
                
                print(f"Created artificial target from {target_col} > {median_val}")
                
                # Check if we have both classes
                if len(df_denied['target'].unique()) > 1:
                    print(f"Target distribution: {df_denied['target'].value_counts().to_dict()}")
                    
                    # Analyze denied loans
                    all_results['denied_loans'] = analyze_data_group(
                        df_denied, "Denied_Loans", output_dir
                    )
                else:
                    print("⚠️ Could not create artificial target with both classes")
            else:
                print("⚠️ No suitable numeric columns found for artificial target")
    
    # Create cross-dataset comparisons
    if output_dir and len(all_results) > 1:
        # Create comparison directory
        cross_dir = os.path.join(output_dir, "Cross_Dataset_Comparison")
        os.makedirs(cross_dir, exist_ok=True)
        
        # Compare unfair models across datasets
        unfair_models = []
        for group, results in all_results.items():
            if results and 'unfair' in results and results['unfair']:
                for model in results['unfair']:
                    model_copy = model.copy()
                    model_copy['Data_Group'] = group
                    unfair_models.append(model_copy)
        
        if unfair_models:
            # Create DataFrame
            unfair_df = pd.DataFrame(unfair_models)
            
            # Save comparison data
            unfair_df.to_csv(os.path.join(cross_dir, "unfair_models_across_datasets.csv"), index=False)
            
            # Create visualizer
            visualizer = FairnessVisualizer()
            
            # Create visualizations
            visualizer.create_all_visualizations(
                unfair_df, 
                output_dir=cross_dir,
                prefix="unfair_cross_dataset_"
            )
        
        # Compare fair models across datasets
        fair_models = []
        for group, results in all_results.items():
            if results and 'fair' in results and results['fair']:
                for model in results['fair']:
                    model_copy = model.copy()
                    model_copy['Data_Group'] = group
                    fair_models.append(model_copy)
        
        if fair_models:
            # Create DataFrame
            fair_df = pd.DataFrame(fair_models)
            
            # Save comparison data
            fair_df.to_csv(os.path.join(cross_dir, "fair_models_across_datasets.csv"), index=False)
            
            # Create visualizer
            visualizer = FairnessVisualizer()
            
            # Create visualizations
            visualizer.create_all_visualizations(
                fair_df, 
                output_dir=cross_dir,
                prefix="fair_cross_dataset_"
            )
    
    return all_results

def main():
    """Main function to run the HMDA fairness analysis pipeline"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='HMDA Fairness Analysis Pipeline')
    parser.add_argument('--data', type=str, default='/Users/dilara/Desktop/SHAP/hmda_NC_2023_cleaned.csv',
                       help='Path to HMDA CSV file')
    parser.add_argument('--output', type=str, default='/Users/dilara/Desktop/fairness_results',
                       help='Directory to save results')
    parser.add_argument('--sample', type=int, default=0,
                       help='Use a random sample of data (specify size, 0 = use all)')
    
    args = parser.parse_args()
    
    # Start timer
    start_time = time.time()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"⚠️ Error: Data file not found at {args.data}")
        return 1
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🔸 HMDA Fairness Analysis Pipeline 🔸")
    print(f"Data file: {args.data}")
    print(f"Output directory: {args.output}")
    print("="*80)
    
    try:
        # Load data
        print("\n🔹 Loading data...")
        df = pd.read_csv(args.data, low_memory=False)
        print(f"Loaded {len(df)} rows and {df.shape[1]} columns")
        
        # Use sample if specified
        if args.sample > 0 and args.sample < len(df):
            print(f"Taking random sample of {args.sample} rows")
            df = df.sample(args.sample, random_state=42)
        
        # Ensure we have target column
        if 'action_taken' in df.columns and 'target' not in df.columns:
            df['target'] = (df['action_taken'] == 1).astype(int)
            print(f"Created target column from action_taken")
        
        # Run complete analysis
        results = run_complete_analysis(df, args.output)
        
        # Save metrics explanation
        metrics = FairnessMetrics()
        with open(os.path.join(args.output, "metrics_explanation.txt"), "w") as f:
            for metric, explanation in metrics.explain_metrics().items():
                f.write(f"{metric}: {explanation}\n\n")
        
        # Final results
        print("\n" + "="*80)
        print(f"✅ Analysis completed in {(time.time() - start_time)/60:.2f} minutes")
        print(f"Results saved to: {args.output}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"⚠️ Error in analysis: {e}")
        return 1

if __name__ == "__main__":
    exit(main())