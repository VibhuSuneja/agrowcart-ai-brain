import pandas as pd
import os

def run_audit():
    print("🛡️ [AgrowCart Research Engine] Starting Pre-Submission Feature Audit...")
    
    csv_path = 'master_training_data_v2.csv'
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    all_columns = df.columns.tolist()
    
    # These are the meta/categorical columns dropped by the training engine
    excluded_cols = ['Date', 'Market_Name', 'Commodity', 'District']
    
    # The actual features used as input to the LSTM/Transformer
    feature_cols = [c for c in all_columns if c not in excluded_cols]
    
    print(f"\n✅ Data Source Verified: {csv_path}")
    print(f"✅ Total Raw Dimensions: {len(all_columns)}")
    print(f"✅ Excluded Metadata: {len(excluded_cols)} ({', '.join(excluded_cols)})")
    
    print("-" * 50)
    print(f"🚀 FINAL FEATURE COUNT: {len(feature_cols)}")
    print("-" * 50)
    
    print("\nFeature Taxonomy:")
    for i, col in enumerate(feature_cols, 1):
        print(f"{i:2d}. {col}")
        
    print("\n--- AUDIT VERDICT ---")
    if len(feature_cols) == 15:
        print("🟢 SUCCESS: Feature count matches 'Research Paper Draft.md' (15 features).")
    else:
        print(f"🔴 MISMATCH: Found {len(feature_cols)} features, paper expected 15.")

if __name__ == "__main__":
    run_audit()
