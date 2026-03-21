import pandas as pd
import os
import numpy as np

def fetch_agmarknet_tomato_haryana():
    print("🌾 Generating Simulated Tomato Mandi Prices for Haryana...")
    
    date_range = pd.date_range(start='2024-01-01', end='2026-02-28', freq='D')
    
    # Simple simulation of price volatility
    np.random.seed(42)
    base_price = 2000
    prices = base_price + np.cumsum(np.random.normal(0, 50, len(date_range)))
    prices = np.clip(prices, 500, 8000)
    
    df = pd.DataFrame({
        'Date': date_range,
        'Market_Name': 'Kurukshetra',
        'Commodity': 'Tomato',
        'Modal_Price': prices,
        'Min_Price': prices * 0.9,
        'Max_Price': prices * 1.1
    })
    
    os.makedirs('datasets', exist_ok=True)
    filepath = 'datasets/agmarknet_tomato_haryana.csv'
    df.to_csv(filepath, index=False)
    
    print(f"🚀 SUCCESS! Mandi price data saved to {filepath}")
    print(df.head())

if __name__ == "__main__":
    fetch_agmarknet_tomato_haryana()
