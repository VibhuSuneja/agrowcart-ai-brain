import requests
import pandas as pd
import os
import time

# --- CONFIGURATION ---
# Register at data.gov.in to get your own API Key
API_KEY = "579b464db66ec23bdd0000019d9dbdc537724906531135f2350df022" 
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

# All 7 major Indian millets with exact Agmarknet commodity labels
MILLETS = {
    'Bajra':    'Bajra(Pearl Millet/Cumbu)',
    'Jowar':    'Jowar(Sorghum)',
    'Ragi':     'Ragi(Finger Millet)',
    'Kodo':     'Kodo Millet',
    'Foxtail':  'Foxtail Millet/Italian Millet/Navane',
    'Barnyard': 'Sawa/Shamwa(Barnyard Millet)',
    'Little':   'Little Millet',
}

# Top-producing state per millet for maximum data coverage
MILLET_STATES = {
    'Bajra':    'Rajasthan',
    'Jowar':    'Maharashtra',
    'Ragi':     'Karnataka',
    'Kodo':     'Madhya Pradesh',
    'Foxtail':  'Andhra Pradesh',
    'Barnyard': 'Uttarakhand',
    'Little':   'Madhya Pradesh',
}

def fetch_millet_prices(commodity_name, state="Rajasthan", district=None):
    """
    Fetches historical and daily mandi prices from OGD India API.
    Loops through pages to build a larger dataset.
    """
    print(f"🌾 Querying Agmarknet API for {commodity_name} in {state}...")
    
    all_records = []
    limit = 1000
    offset = 0
    
    while True:
        base_url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
        params = {
            "api-key": API_KEY,
            "format": "json",
            "offset": offset,
            "limit": limit,
            "filters[state]": state,
            "filters[commodity]": commodity_name
        }
        
        if district:
            params["filters[district]"] = district
        
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json()
                records = data.get('records', [])
                
                if not records:
                    break
                
                all_records.extend(records)
                print(f"📦 Fetched {len(all_records)} records so far...")
                
                # If we got fewer records than the limit, we've reached the end
                if len(records) < limit:
                    break
                
                offset += limit
                time.sleep(1) # Be polite to the Gov API
            else:
                print(f"❌ Gov API Error: {response.status_code}")
                break
        except Exception as e:
            print(f"❌ Connection Error: {e}")
            break

    if not all_records:
        print(f"❌ No records found for {commodity_name} in {state}.")
        return None
    
    df = pd.DataFrame(all_records)
    # Convert arrival date to datetime objects
    df['arrival_date'] = pd.to_datetime(df['arrival_date'], dayfirst=True)
    return df

def process_and_save_millet(millet_key):
    commodity_label = MILLETS.get(millet_key)
    if not commodity_label:
        print(f"❌ Unknown millet: {millet_key}. Valid options: {list(MILLETS.keys())}")
        return
    hub_state = MILLET_STATES.get(millet_key, 'Rajasthan')
    df = fetch_millet_prices(commodity_label, state=hub_state)
    
    if df is not None:
        os.makedirs('datasets', exist_ok=True)
        filename = f"datasets/{millet_key.lower()}_massive_dataset.csv"
        df.to_csv(filename, index=False)
        print(f"✅ {millet_key.upper()} SUCCESS! {len(df)} records saved to {filename}")
        print(df.head())
    else:
        print(f"❌ No data for {millet_key} in {hub_state}. Trying fallback: All India...")
        df_fb = fetch_millet_prices(commodity_label, state=None)
        if df_fb is not None:
            filename = f"datasets/{millet_key.lower()}_massive_dataset.csv"
            df_fb.to_csv(filename, index=False)
            print(f"✅ {millet_key.upper()} FALLBACK SUCCESS! {len(df_fb)} records saved.")

if __name__ == "__main__":
    import sys
    millet = sys.argv[1] if len(sys.argv) > 1 else 'Bajra'
    process_and_save_millet(millet)
