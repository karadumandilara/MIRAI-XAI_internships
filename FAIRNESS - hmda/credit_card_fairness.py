def main():
    """Main function to run the credit card fairness analysis pipeline"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Credit Card Fairness Analysis Pipeline')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to credit card CSV file')
    parser.add_argument('--output', type=str, default='./credit_card_fairness_results',
                       help='Directory to save results')
    parser.add_argument('--protected', type=str, default='Gender',
                       help='Protected attribute column name')
    parser.add_argument('--target', type=str, default='Approved',
                       help='Target column name')
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
    print(f"🔸 Credit Card Fairness Analysis Pipeline 🔸")
    print(f"Data file: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Protected attribute: {args.protected}")
    print(f"Target column: {args.target}")
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
        
        # Run credit card analysis
        results = run_credit_card_analysis(
            df, 
            output_dir=args.output,
            protected_col=args.protected,
            target_col=args.target
        )
        
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

def detect_columns(df):
    """
    Attempt to automatically detect protected attribute and target columns
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        (protected_col, target_col): Best guesses for column names
    """
    # Look for common protected attribute column names
    protected_candidates = [
        'Gender', 'Sex', 'gender', 'sex', 'gender_code', 'sex_code', 
        'male', 'female', 'is_male', 'is_female', 'protected_attribute'
    ]
    
    # Look for common target column names
    target_candidates = [
        'Approved', 'approved', 'approval', 'target', 'result', 'decision',
        'card_approved', 'is_approved', 'Application_Status', 'application_status',
        'Accept', 'accept', 'accepted', 'status', 'approved_status'
    ]
    
    # Find protected attribute column
    protected_col = None
    for col in protected_candidates:
        if col in df.columns:
            # Verify it has at least two unique values
            if len(df[col].dropna().unique()) >= 2:
                protected_col = col
                break
    
    # Find target column
    target_col = None
    for col in target_candidates:
        if col in df.columns:
            # Verify it has at least two unique values
            if len(df[col].dropna().unique()) >= 2:
                target_col = col
                break
    
    return protected_col, target_col

if __name__ == "__main__":
    exit(main())