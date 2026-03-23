import pandas as pd
import numpy as np
import os

def synthesize_millet_history(raw_api_csv, weather_csv, millet_name):
    """
    Synthesizes historical training data for any millet.
    """
    print(f"🛠️ Generating feature-engineered dataset for {millet_name}...")
    
    # 1. Load weather
    weather_df = pd.read_csv(weather_csv)
    date_range = weather_df['Date'].unique()
    
    # 2. Get latest price
    api_df = pd.read_csv(raw_api_csv)
    current_avg_price = api_df['modal_price'].mean()
    
    # 3. Simulate Price Dynamics
    prices = [current_avg_price]
    for _ in range(len(date_range) - 1):
        noise = np.random.normal(0, 15)
        seasonal_factor = 100 * np.sin(2 * np.pi * len(prices) / 365)
        new_price = prices[-1] + noise + (current_avg_price - prices[-1])*0.05 + seasonal_factor*0.1
        prices.append(new_price)
    
    prices = np.array(prices[::-1])
    
    # 4. Create the training DataFrame
    final_df = weather_df.copy()
    final_df['Modal_Price'] = prices
    final_df['Min_Price'] = prices * 0.94
    final_df['Max_Price'] = prices * 1.06
    final_df['Commodity'] = millet_name
    
    # 5. Feature Engineering
    final_df['Modal_Price_Rolling_7'] = final_df['Modal_Price'].rolling(7).mean().bfill()
    final_df['Temp_Max_Rolling_7'] = final_df['Temp_Max_C'].rolling(7).mean().bfill()
    final_df['Rainfall_Rolling_7'] = final_df['Rainfall_mm'].rolling(7).mean().bfill()
    final_df['Price_Volatility_7'] = final_df['Modal_Price'].rolling(7).std().bfill()
    final_df['Rainfall_Lag_1'] = final_df['Rainfall_mm'].shift(1).bfill()
    final_df['Temp_Lag_1'] = final_df['Temp_Max_C'].shift(1).bfill()
    
    final_df['District'] = 'Kurukshetra' # Placeholder for demo purposes
    final_df['Market_Name'] = 'Kurukshetra'
    
    os.makedirs('datasets', exist_ok=True)
    output_path = f'datasets/{millet_name.lower()}_training_processed.csv'
    final_df.to_csv(output_path, index=False)
    print(f"✅ Success! Training-ready dataset created: {output_path}")

if __name__ == "__main__":
    import sys
    millet = sys.argv[1] if len(sys.argv) > 1 else 'Jowar'
    synthesize_millet_history(
        f'datasets/{millet.lower()}_massive_dataset.csv', 
        'datasets/kurukshetra_weather_24_26.csv',
        millet
    )
