from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import joblib
import numpy as np
import torch.nn as nn
import os
from pydantic import BaseModel
from typing import List

# --- REPLICATING MODEL ARCHITECTURE (Required for loading) ---
class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

app = FastAPI(title="Agrowcart Price Prediction API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None

@app.on_event("startup")
def load_assets():
    global model, scaler
    print("🚀 Loading AI Assets...")
    # Parameters must match the trained model (Now 15 features)
    model = PriceLSTM(input_dim=15, hidden_dim=64, num_layers=2, output_dim=1)
    
    # Use absolute paths to avoid ambiguity during container startup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "price_lstm_model.pth")
    scaler_path = os.path.join(base_dir, "models", "price_scaler.gz")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    scaler = joblib.load(scaler_path)
    print("✅ Assets Loaded Successfully.")

@app.get("/")
def health_check():
    return {"status": "online", "model": "LSTM", "target": "Tomato Prices (Haryana)"}

class PredictionRequest(BaseModel):
    data: List[List[float]]

@app.post("/predict")
async def predict_price(request: PredictionRequest):
    try:
        # data should be [[feat1...feat15], [feat1...feat15], ... 14 times]
        input_array = np.array(request.data, dtype=np.float32)
        # Reshape to (BatchSize=1, SequenceLength=14, Features=15)
        input_tensor = torch.tensor(input_array).unsqueeze(0)
        
        with torch.no_grad():
            prediction = model(input_tensor)
        
        # Convert back to real Rupees
        real_price = scaler.inverse_transform(prediction.numpy())
        
        return {
            "predicted_price": round(float(real_price[0][0]), 2),
            "currency": "INR",
            "mandi": "Kurukshetra"
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
