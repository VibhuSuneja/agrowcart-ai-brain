import pandas as pd
import requests
import json
import os

def validate_live_accuracy():
    print("📈 AI vs real-world Market Comparison...")
    
    # 1. Get the latest real price from our massive dataset
    csv_path = 'datasets/bajra_massive_dataset.csv'
    if not os.path.exists(csv_path):
        print("❌ Error: massive_dataset.csv not found. Run fetch_millet_data.py first!")
        return
    
    real_df = pd.read_csv(csv_path)
    real_avg_price = real_df['modal_price'].mean()
    print(f"✅ Real-world Average (India today): ₹{real_avg_price:.2f}")

    # 2. Get the latest AI prediction
    url = "http://localhost:8000/predict"
    # We use some dummy historical context for the test
    import numpy as np
    dummy_context = np.random.rand(14, 11).tolist()
    
    try:
        response = requests.post(url, json={"data": dummy_context}, timeout=5)
        if response.status_code == 200:
            pred_price = response.json().get('predicted_price')
            print(f"🤖 AI Prediction for today/tomorrow: ₹{pred_price:.2f}")
            
            # 3. Calculate Variance
            diff = abs(real_avg_price - pred_price)
            variance_pct = (diff / real_avg_price) * 100
            
            print("-" * 40)
            print(f"📊 Accuracy Variance: {variance_pct:.2f}%")
            
            if variance_pct < 5:
                print("💎 STATUS: HIGH ACCURACY (Stable for Research)")
            elif variance_pct < 10:
                print("⚖️ STATUS: MODERATE (Normal Market Volatility)")
            else:
                print("⚠️ STATUS: LOW ACCURACY (Model needs retraining with more real history)")
        else:
            print("❌ Server error. Ensure api_server.py is running.")
    except Exception as e:
        print(f"❌ Error connecting to AI: {e}")

if __name__ == "__main__":
    validate_live_accuracy()
