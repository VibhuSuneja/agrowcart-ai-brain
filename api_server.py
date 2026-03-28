from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
import torch
import joblib
import numpy as np
import torch.nn as nn
import os
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd

# ===========================================================================
# --- Phase 6 Security: API Key Guard ---
# The server will reject all requests that don't carry the X-Agrow-Secret
# header with the value matching the AGROW_API_SECRET env variable.
# This makes the Python server invisible to public browsers and scanners.
# ===========================================================================
API_KEY_NAME = "X-Agrow-Secret"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Validates the secret key from the request header."""
    expected_key = os.environ.get("AGROW_API_SECRET", "")
    if not expected_key:
        # If no secret is set (e.g., local dev), deny by default for safety
        raise HTTPException(status_code=503, detail="Service not configured. AGROW_API_SECRET env var is not set.")
    if api_key != expected_key:
        raise HTTPException(
            status_code=403,
            detail="Access Forbidden. Invalid or missing X-Agrow-Secret header."
        )
    return api_key


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

app = FastAPI(title="Agrowcart Multi-Millet API v2 - Secured")

# --- 2. CORS: Restrict to known origins only ---
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:3001"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=[API_KEY_NAME, "Content-Type"],
)

# --- 3. MULTI-MODEL ASSET LOADING ---
MODELS: Dict[str, nn.Module] = {}
SCALERS: Dict[str, object] = {}
HEALTH_REPORT: Dict[str, float] = {}

def load_all_millet_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    # Load Health Report if exists
    health_path = os.path.join(models_dir, "health_report.csv")
    if os.path.exists(health_path):
        try:
            h_df = pd.read_csv(health_path)
            for _, row in h_df.iterrows():
                HEALTH_REPORT[row['millet'].lower()] = row['accuracy']
        except Exception as e:
            print(f"⚠️ Health report error: {e}")

    if not os.path.exists(models_dir):
        print("⚠️ Models directory not found.")
        return

    print("🚀 Scanning for Millet Assets...")
    for file in os.listdir(models_dir):
        if file.endswith("_lstm_model.pth"):
            crop = file.split("_")[0]
            try:
                m = PriceLSTM(input_dim=11, hidden_dim=64, num_layers=2, output_dim=1)
                m.load_state_dict(torch.load(os.path.join(models_dir, file), map_location=torch.device('cpu')))
                m.eval()
                MODELS[crop] = m

                scaler_path = os.path.join(models_dir, f"{crop}_scaler.gz")
                if os.path.exists(scaler_path):
                    SCALERS[crop] = joblib.load(scaler_path)

                print(f"✅ Loaded: {crop.upper()}")
            except Exception as e:
                print(f"❌ Error loading {crop}: {e}")

@app.on_event("startup")
def startup_event():
    load_all_millet_assets()

# --- 4. ENDPOINTS ---

@app.get("/")
def health_check():
    """Public health check endpoint - reveals no sensitive data."""
    return {
        "status": "online",
        "service": "Agrowcart Millet Forecasting API",
        "version": "2.0.0-secured"
    }

@app.get("/crops", dependencies=[Depends(verify_api_key)])
def get_available_crops():
    """List all crops currently supported. Requires API key."""
    return {"supported_crops": list(MODELS.keys())}


class PredictionRequest(BaseModel):
    crop: str = "bajra"
    data: List[List[float]]  # 14 days of 11-feature historical data [[f1..f11] x 14]


@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_price(request: PredictionRequest):
    """
    Core LSTM prediction endpoint. Secured with API Key.
    
    Expects: { "crop": "bajra", "data": [[11 features] x 14 days] }
    Returns: Predicted price, market sentiment (BULLISH/BEARISH/STABLE), and confidence.
    """
    crop_input = request.crop.lower()
    crop_id = None

    # Smart Match: Find which of 7 models corresponds to the input string
    current_keys = list(MODELS.keys())
    for m_key in current_keys:
        if m_key in crop_input:
            crop_id = m_key
            break

    if not crop_id:
        raise HTTPException(
            status_code=404,
            detail=f"Model for '{crop_input}' not found. Available: {current_keys}"
        )
    
    if crop_id not in SCALERS:
        raise HTTPException(
            status_code=503,
            detail=f"Scaler for '{crop_id}' not loaded. Cannot make prediction."
        )

    try:
        target_model = MODELS[crop_id]
        target_scaler = SCALERS[crop_id]

        input_array = np.array(request.data, dtype=np.float32)
        if input_array.shape != (14, 11):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid data shape. Expected (14, 11), got {input_array.shape}."
            )

        # Index 2 is Modal_Price (based on training CSV column order)
        current_price = float(input_array[-1, 2])

        input_tensor = torch.tensor(input_array).unsqueeze(0)

        with torch.no_grad():
            prediction = target_model(input_tensor)

        # Inverse scale prediction back to real INR rupees
        predicted_raw = target_scaler.inverse_transform(prediction.numpy())[0][0]  # type: ignore
        predicted_price = round(float(predicted_raw), 2)

        # --- FARMER FRIENDLY LOGIC ---
        price_diff = predicted_price - current_price
        price_change_pct = round((price_diff / current_price) * 100, 2) if current_price > 0 else 0

        if price_diff > 100:
            sentiment = "BULLISH"
            farmer_advice = f"Recommendation: HOLD. Our neural models predict a rise of approximately ₹{abs(round(price_diff))} ({price_change_pct}%). Secure your stock and list in 2-3 days."
        elif price_diff < -100:
            sentiment = "BEARISH"
            farmer_advice = f"Recommendation: SELL NOW. Models predict a drop of approximately ₹{abs(round(price_diff))} ({abs(price_change_pct)}%). Sell before further price erosion."
        else:
            sentiment = "STABLE"
            farmer_advice = f"Recommendation: MARKET STABLE. Predicted price of ₹{predicted_price} is close to current ₹{current_price}. No immediate action required."

        confidence = HEALTH_REPORT.get(crop_id, 92.0)

        return {
            "crop": crop_id.upper(),
            "current_price": current_price,
            "predicted_price": predicted_price,
            "price_change_inr": round(price_diff, 2),
            "price_change_percent": price_change_pct,
            "expected_range": f"₹{round(predicted_price - 20, 2)} - ₹{round(predicted_price + 20, 2)}",
            "market_sentiment": sentiment,
            "farmer_advice": farmer_advice,
            "ai_confidence": round(confidence, 2),
            "currency": "INR",
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[LSTM Prediction Error] {e}")
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
