import requests
import json
import numpy as np

url = "http://localhost:8000/predict"

# 14 days of history, 11 features each (normalized for test)
# Features: Modal_Price, Min_Price, Max_Price, Temp_Max, Rainfall, 
#           Rolling7, TempRolling7, RainRolling7, Volatility7, RainLag, TempLag
dummy_history = np.random.rand(14, 11).tolist()

payload = {
    "data": dummy_history
}

print("🚀 Sending prediction request for Bajra...")
try:
    response = requests.post(url, json=payload, timeout=10)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction Successful!")
        print(f"Mandi: {result.get('mandi', 'Unknown')}")
        print(f"Forecast: ₹{result.get('predicted_price')}")
    else:
        print(f"❌ Error: {response.text}")
except Exception as e:
    print(f"❌ Request failed: {e}")
