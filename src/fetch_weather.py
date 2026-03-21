import requests
import pandas as pd
import os

# --- CONFIGURATION ---
# Coordinates for Kurukshetra, Haryana
LATITUDE = 29.9695
LONGITUDE = 76.8226

# Timeframe for our research
START_DATE = "20240101"
END_DATE = "20260228" # Up to last month

# NASA POWER API Endpoint for daily agricultural data
# T2M_MAX = Max Temperature at 2 meters (C)
# PRECTOTCORR = Corrected Total Precipitation (mm)
URL = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=T2M_MAX,PRECTOTCORR&community=AG&longitude={LONGITUDE}&latitude={LATITUDE}&start={START_DATE}&end={END_DATE}&format=JSON"

def fetch_kurukshetra_weather():
    print("🌍 Initiating connection to NASA POWER API...")
    
    try:
        response = requests.get(URL)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Data received. Parsing JSON...")
        
        # Extract the specific parameter dictionaries
        temp_data = data['properties']['parameter']['T2M_MAX']
        rain_data = data['properties']['parameter']['PRECTOTCORR']
        
        # Convert to Pandas DataFrame
        df = pd.DataFrame({
            'Date': list(temp_data.keys()),
            'Temp_Max_C': list(temp_data.values()),
            'Rainfall_mm': list(rain_data.values())
        })
        
        # Format the Date column to standard YYYY-MM-DD
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        df['District'] = 'Kurukshetra' # Add district for future merging
        
        # Clean up -999.0 values (NASA's code for missing data)
        df.replace(-999.0, pd.NA, inplace=True)
        df.fillna(method='ffill', inplace=True) # Forward fill any missing gaps
        
        # Save to CSV
        os.makedirs('datasets', exist_ok=True)
        filepath = 'datasets/kurukshetra_weather_24_26.csv'
        df.to_csv(filepath, index=False)
        
        print(f"🚀 SUCCESS! Weather data saved to {filepath}")
        print(df.head())
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except KeyError as e:
        print(f"❌ Data Parsing Error. Unexpected JSON structure: {e}")

if __name__ == "__main__":
    fetch_kurukshetra_weather()