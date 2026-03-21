import pandas as pd
import numpy as np
import os

def preprocess_data():
    print("🧹 Starting Data Preprocessing...")
    
    # 1. Load Datasets
    weather_path = 'datasets/kurukshetra_weather_24_26.csv'
    price_path = 'datasets/agmarknet_tomato_haryana.csv'
    
    if not os.path.exists(weather_path) or not os.path.exists(price_path):
        print("❌ Error: One or more datasets are missing.")
        return

    df_weather = pd.read_csv(weather_path)
    df_price = pd.read_csv(price_path)
    
    # 2. Standardize Date Formats
    df_weather['Date'] = pd.to_datetime(df_weather['Date'])
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    
    # 3. Handle Missing Values via Interpolation
    # First, ensure we have a continuous date range to identify gaps
    min_date = max(df_weather['Date'].min(), df_price['Date'].min())
    max_date = min(df_weather['Date'].max(), df_price['Date'].max())
    
    date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Reindex and Interpolate Weather
    df_weather = df_weather.set_index('Date').reindex(date_range).rename_axis('Date').reset_index()
    df_weather['Temp_Max_C'] = df_weather['Temp_Max_C'].interpolate(method='linear')
    df_weather['Rainfall_mm'] = df_weather['Rainfall_mm'].interpolate(method='linear')
    
    # Reindex and Interpolate Price
    df_price = df_price.set_index('Date').reindex(date_range).rename_axis('Date').reset_index()
    df_price['Modal_Price'] = df_price['Modal_Price'].interpolate(method='linear')
    df_price['Min_Price'] = df_price['Min_Price'].interpolate(method='linear')
    df_price['Max_Price'] = df_price['Max_Price'].interpolate(method='linear')
    
    # 4. Merge on Date
    df_master = pd.merge(df_price, df_weather, on='Date', how='inner')
    
    # 5. Engineer Time-Series Features
    print("⚙️ Engineering features...")
    
    # 7-day Rolling Averages
    df_master['Modal_Price_Rolling_7'] = df_master['Modal_Price'].rolling(window=7).mean()
    df_master['Temp_Max_Rolling_7'] = df_master['Temp_Max_C'].rolling(window=7).mean()
    df_master['Rainfall_Rolling_7'] = df_master['Rainfall_mm'].rolling(window=7).mean()
    
    # Price Volatility (7-day standard deviation)
    df_master['Price_Volatility_7'] = df_master['Modal_Price'].rolling(window=7).std()
    
    # Lags (Weather shocks)
    df_master['Rainfall_Lag_1'] = df_master['Rainfall_mm'].shift(1)
    df_master['Temp_Lag_1'] = df_master['Temp_Max_C'].shift(1)
    
    # 5.5 Cyclical Time Features (Seasonal Context)
    df_master['Day_Sin'] = np.sin(2 * np.pi * df_master['Date'].dt.dayofweek / 7)
    df_master['Day_Cos'] = np.cos(2 * np.pi * df_master['Date'].dt.dayofweek / 7)
    df_master['Month_Sin'] = np.sin(2 * np.pi * df_master['Date'].dt.month / 12)
    df_master['Month_Cos'] = np.cos(2 * np.pi * df_master['Date'].dt.month / 12)

    # 6. Drop initial NaN rows created by rolling/lag operations
    df_master.dropna(inplace=True)
    
    # 7. Export Master Training Data
    output_path = 'master_training_data_v2.csv'
    df_master.to_csv(output_path, index=False)
    
    print(f"🚀 SUCCESS! Master training data saved to {output_path}")
    print(f"Total Rows: {len(df_master)}")
    print(df_master.head())

if __name__ == "__main__":
    preprocess_data()
