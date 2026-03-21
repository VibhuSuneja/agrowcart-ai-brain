# Agrowcart AI: Expansion Plan for "All Millets of India"

To scale the Agrowcart AI Brain from just Tomatoes to **all Indian Millets**, we must transition from manual CSV downloads to an automated pipeline that pulls directly from verified Government of India databases. 

The United Nations declared 2023 the "International Year of Millets" (IYM), and India is the largest producer globally. Tracking these prices accurately is a high-impact thesis topic.

## 1. The 9 Target Millets in India
To create a comprehensive "Millet Forecasting Engine", the AI needs to be trained on these specific crops as categorized by the Ministry of Agriculture:

**Major Millets:**
1. **Sorghum (Jowar)** - *High volume, primarily Maharashtra, Karnataka, Andhra Pradesh.*
2. **Pearl Millet (Bajra)** - *Highest volume, primarily Rajasthan, UP, Haryana.*
3. **Finger Millet (Ragi/Mandua)** - *High volume, primarily Karnataka, Tamil Nadu.*

**Minor (Shree Anna) Millets:**
4. **Foxtail Millet (Kangni/Kakun)**
5. **Proso Millet (Cheena)**
6. **Kodo Millet (Kodo)**
7. **Barnyard Millet (Sawa/Sanwa/Jhangora)**
8. **Little Millet (Kutki)**
9. **Brown top millet (Korale)**

---

## 2. Reliable Government Data Sources (API Integration)

For a B.Tech thesis and a production-grade application, you must cite reliable, official sources. 

### A. Primary Source: OGD India API (data.gov.in)
The **Open Government Data (OGD) Platform India** provides a direct streaming API for the **Agmarknet** database. This is the ultimate "source of truth" for APMC Mandi prices.
*   **What it provides:** Daily Arrivals, Minimum Price, Maximum Price, and Modal Price (the most common price) for over 200 commodities.
*   **Millet Coverage:** Extensive coverage for Bajra, Jowar, and Ragi.
*   **API Access:** You can register for a free API key at [data.gov.in](https://data.gov.in/).
*   **Endpoint Example:** `https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070`

### B. Secondary Source: e-NAM Database
The **National Agriculture Market (e-NAM)** portal links mandis across India. 
*   **What it provides:** Real-time auction data, trade volumes, and electronic bidding prices.
*   **Use Case:** Excellent for validating Agmarknet data or if you specifically want to forecast "digital auction" prices.

### C. Macro Data: Directorate of Economics & Statistics (DES)
*   **What it provides:** MSP (Minimum Support Prices), crop estimation, and Farm Harvest Prices (FHP).
*   **Use Case:** Your AI can use MSP as a "Floor Feature" (a baseline below which prices rarely sustain for long). If the government sets the Bajra MSP at ₹2500, the AI learns this is a psychological support level.

---

## 3. Data Extraction Architecture (How to code it)

To build this, you need a new Python script that runs daily, queries `data.gov.in`, and updates your local datasets before retraining.

Here is the exact architectural blueprint for your `fetch_millet_data.py` script:

```python
import requests
import pandas as pd
import time

# Get this free from data.gov.in
API_KEY = "YOUR_GOVT_API_KEY"
# The specific Agmarknet Resource ID on the portal
RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"

MILLETS_TO_TRACK = ['Bajra(Pearl Millet)', 'Jowar(Sorghum)', 'Ragi (Finger Millet)']

def fetch_millet_prices(state, commodity):
    """
    Queries the official Gov India API for daily mandi prices.
    """
    base_url = f"https://api.data.gov.in/resource/{RESOURCE_ID}"
    params = {
        "api-key": API_KEY,
        "format": "json",
        "offset": 0,
        "limit": 1000,
        "filters[state]": state,
        "filters[commodity]": commodity
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data['records'])
    else:
        print(f"Gov API Error: {response.status_code}")
        return None

# Execution Flow:
# 1. Loop through MILLETS_TO_TRACK
# 2. Call fetch_millet_prices()
# 3. Clean the 'modal_price' column
# 4. Merge with fetch_weather.py data for the corresponding districts
# 5. Save as 'datasets/{millet_name}_training_data.csv'
```

---

## 4. Next Steps for Implementation

If you want to transition the project completely to Millets for your final presentation, here is your path:

1.  **Get the API Key:** Register at `data.gov.in` to get your secret API key.
2.  **Pull the Data:** Implement the script above to build a massive dataset for Bajra, Jowar, and Ragi.
3.  **Train 3 Models:** Run our `train_models.py` system three separate times on the newly downloaded data to generate 3 brains:
    *   `bajra_lstm_model.pth`
    *   `jowar_lstm_model.pth`
    *   `ragi_lstm_model.pth`
4.  **Update FastAPI:** Modify `api_server.py` to accept the crop name in the request, and route it to the correct `pth` file.
