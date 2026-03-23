from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
import numpy as np
import torch.nn as nn
import os
from pydantic import BaseModel
from typing import List, Dict

# --- 1. MODEL ARCHITECTURE ---
class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

app = FastAPI(title="Agrowcart Multi-Millet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. MULTI-MODEL ASSET LOADING ---
MODELS: Dict[str, nn.Module] = {}
SCALERS: Dict[str, object] = {}

def load_all_millet_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(models_dir):
        print("⚠️ Models directory not found.")
        return

    print("🚀 Scanning for Millet Assets...")
    for file in os.listdir(models_dir):
        if file.endswith("_lstm_model.pth"):
            crop = file.split("_")[0]
            try:
                # Load LSTM
                m = PriceLSTM(input_dim=11, hidden_dim=64, num_layers=2, output_dim=1)
                m.load_state_dict(torch.load(os.path.join(models_dir, file), map_location=torch.device('cpu')))
                m.eval()
                MODELS[crop] = m
                
                # Load Scaler
                scaler_path = os.path.join(models_dir, f"{crop}_scaler.gz")
                if os.path.exists(scaler_path):
                    SCALERS[crop] = joblib.load(scaler_path)
                
                print(f"✅ Loaded: {crop.upper()}")
            except Exception as e:
                print(f"❌ Error loading {crop}: {e}")

@app.on_event("startup")
def startup_event():
    load_all_millet_assets()

# --- 3. ENDPOINTS ---
@app.get("/")
def health_check():
    return {
        "status": "online",
        "active_models": list(MODELS.keys()),
        "target": "Agrowcart Indian Millet Forecasting"
    }

@app.get("/crops")
def get_available_crops():
    """List all crops currently supported by the AI engine."""
    return {"supported_crops": list(MODELS.keys())}

class PredictionRequest(BaseModel):
    crop: str = "bajra"
    data: List[List[float]] # 14 days of historical data

@app.post("/predict")
async def predict_price(request: PredictionRequest):
    crop = request.crop.lower()
    if crop not in MODELS:
        return {"error": f"Model for '{crop}' not found. Available: {list(MODELS.keys())}"}

    try:
        # data should be [[feat1...feat11], ... 14 times]
        input_array = np.array(request.data, dtype=np.float32)
        if input_array.shape != (14, 11):
            return {"error": f"Invalid data shape. Expected (14, 11)."}
            
        # Get the latest price from input for comparison (index 2 is modal_price)
        current_price = input_array[-1, 2]
        
        input_tensor = torch.tensor(input_array).unsqueeze(0)
        
        with torch.no_grad():
            prediction = MODELS[crop](input_tensor)
        
        # Scale back to Real Rupees
        # Type ignored for MinMaxScaler call
        real_price = SCALERS[crop].inverse_transform(prediction.numpy())[0][0] # type: ignore
        
        # --- FARMER FRIENDLY LOGIC ---
        price_diff = real_price - current_price
        sentiment = "Neutral"
        advice = "Wait and monitor mandi volume."
        
        if price_diff > 100:
            sentiment = "📈 BULLISH (Increasing)"
            advice = "Recommendation: HOLD. Prices are likely to rise in the next 24-48 hours. Secure your stock."
        elif price_diff < -100:
            sentiment = "📉 BEARISH (Decreasing)"
            advice = "Recommendation: SELL NOW. Market trend is cooling. Sell before further price drops."
        else:
            sentiment = "⚖️ STABLE"
            advice = "Recommendation: MARKET STABLE. Current prices are fair. No immediate risk in selling or holding."

        return {
            "crop": crop.upper(),
            "predicted_price": round(float(real_price), 2),
            "expected_range": f"₹{round(real_price-20, 2)} - ₹{round(real_price+20, 2)}",
            "market_sentiment": sentiment,
            "farmer_advice": advice,
            "currency": "INR",
            "mandi_region": "Primary Indian Hub"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
