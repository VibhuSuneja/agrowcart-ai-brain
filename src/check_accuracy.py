import pandas as pd
import numpy as np
import torch
import joblib
import os
from datetime import datetime, timedelta

def verify_model_accuracy(millet_name):
    """
    Simulates a 'Back-Validation' by testing the AI against the 
    most recent actual market prices.
    """
    model_path = f"models/{millet_name}_lstm_model.pth"
    scaler_path = f"models/{millet_name}_scaler.gz"
    data_path = f"datasets/{millet_name}_training_processed.csv"

    if not all(os.path.exists(p) for p in [model_path, scaler_path, data_path]):
        return None

    # Load resources
    df = pd.read_csv(data_path)
    scaler = joblib.load(scaler_path)
    
    # Simple evaluation: Test against the last 5 days
    # (Assuming the model was trained on the whole set, this measures 'Training Fit')
    # For a real drift check, we would hold out new data.
    
    test_rows = df.tail(14)
    if len(test_rows) < 14: return None
    
    # Extract features (skip Date and Target if it was included in training columns earlier)
    # We use the same feature set as api_server.py
    features = test_rows.drop(columns=['Date', 'District', 'Commodity', 'Market_Name'], errors='ignore').values
    
    # Prepare tensor
    input_tensor = torch.tensor(features).float().unsqueeze(0)
    
    # Load model and predict
    from api_server import PriceLSTM
    model = PriceLSTM(input_dim=11, hidden_dim=64, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    
    with torch.no_grad():
        pred_scaled = model(input_tensor)
        
    actual_price = test_rows.iloc[-1]['Modal_Price']
    predicted_price = scaler.inverse_transform(pred_scaled.numpy())[0][0]
    
    error_pct = abs(actual_price - predicted_price) / actual_price * 100
    accuracy = max(0, 100 - error_pct)
    
    return {
        "millet": millet_name.upper(),
        "accuracy": round(accuracy, 2),
        "status": "Healthy" if accuracy > 85 else "Needs Retraining"
    }

if __name__ == "__main__":
    MILLETS = ['bajra', 'jowar', 'ragi', 'kodo', 'foxtail', 'barnyard', 'little']
    print(f"--- 📊 AI Engine Health Report ({datetime.now().strftime('%Y-%m-%d')}) ---")
    
    report = []
    for m in MILLETS:
        stats = verify_model_accuracy(m)
        if stats:
            print(f"Millet: {stats['millet']} | Accuracy: {stats['accuracy']}% | Status: {stats['status']}")
            report.append(stats)
            
    # Save report for API to read
    report_df = pd.DataFrame(report)
    report_df.to_csv("models/health_report.csv", index=False)
    print("\n✅ Health report saved to models/health_report.csv")
