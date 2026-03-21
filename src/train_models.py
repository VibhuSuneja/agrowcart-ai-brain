import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math

# --- 1. CONFIGURATION & HYPERPARAMETERS ---
LOOKBACK_WINDOW = 14  # Use 14 days of history...
FORECAST_HORIZON = 1  # ...to predict 1 day ahead
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 0.0005

# --- 2. DATA PREPARATION (The Sliding Window) ---
def load_and_window_data(filepath):
    print("Loading data and creating sequences...")
    df = pd.read_csv(filepath)
    
    # We predict 'Modal_Price', but we use all numeric features as input
    # Drop non-numeric columns: Date, Market_Name, Commodity, District
    features = df.drop(columns=['Date', 'Market_Name', 'Commodity', 'District']).values
    target = df['Modal_Price'].values.reshape(-1, 1)
    
    # Scale the data (Neural networks hate large numbers)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(features)
    scaled_target = target_scaler.fit_transform(target)
    
    X, y = [], []
    for i in range(len(scaled_features) - LOOKBACK_WINDOW - FORECAST_HORIZON + 1):
        X.append(scaled_features[i : i + LOOKBACK_WINDOW])
        y.append(scaled_target[i + LOOKBACK_WINDOW : i + LOOKBACK_WINDOW + FORECAST_HORIZON])
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).squeeze(-1)
    
    # Split: 80% Train, 20% Test chronologically (NEVER shuffle time-series data)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    return X_train, y_train, X_test, y_test, target_scaler, scaled_features.shape[1]

# --- 3. THE ARCHITECTURES ---

# Model A: The Baseline LSTM
class PriceLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        out, (hn, cn) = self.lstm(x)
        # We only want the output from the last time step
        out = self.fc(out[:, -1, :]) 
        return out

# Model B: The State-of-the-Art Transformer
class PositionalEncoding(nn.Module):
    # Transformers don't understand time inherently. We must inject "time stamps".
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class PriceTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, output_dim):
        super(PriceTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x)
        # Average pooling across the sequence to get final prediction
        out = out.mean(dim=1) 
        out = self.fc(out)
        return out

# --- 4. THE TRAINING ENGINE ---
def train_model(model, X_train, y_train, model_name="Model"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n--- Training {model_name} ---")
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}')
            
    return model

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --- 4.5 EVALUATION & VISUALIZATION ---
def evaluate_and_plot(model, X_test, y_test, scaler, model_name="Model"):
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        predictions = model(X_test).numpy()
    
    actuals = y_test.numpy()
    
    # Inverse transform to get actual Rupee (₹) values back!
    predictions_real = scaler.inverse_transform(predictions)
    actuals_real = scaler.inverse_transform(actuals.reshape(-1, 1))
    
    # Calculate Metrics
    rmse = np.sqrt(mean_squared_error(actuals_real, predictions_real))
    mae = mean_absolute_error(actuals_real, predictions_real)
    
    print(f"\n📊 {model_name} Results:")
    print(f"RMSE: ₹{rmse:.2f}")
    print(f"MAE:  ₹{mae:.2f} (Average error in prediction)")
    
    return predictions_real, actuals_real, rmse, mae

# --- 5. EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    X_train, y_train, X_test, y_test, target_scaler, num_features = load_and_window_data('master_training_data.csv')
    
    # 2. Initialize Models
    lstm_model = PriceLSTM(input_dim=num_features, hidden_dim=64, num_layers=2, output_dim=FORECAST_HORIZON)
    transformer_model = PriceTransformer(input_dim=num_features, d_model=32, nhead=2, num_layers=2, output_dim=FORECAST_HORIZON)
    
    # 3. Train Both Models
    print("\n--- Phase A: Training LSTM ---")
    trained_lstm = train_model(lstm_model, X_train, y_train, "LSTM")
    
    print("\n--- Phase B: Training Transformer ---")
    trained_transformer = train_model(transformer_model, X_train, y_train, "Transformer")
    
    # 4. Evaluate and Get Real Values
    lstm_preds, actuals, lstm_rmse, lstm_mae = evaluate_and_plot(trained_lstm, X_test, y_test, target_scaler, "LSTM")
    trans_preds, _, trans_rmse, trans_mae = evaluate_and_plot(trained_transformer, X_test, y_test, target_scaler, "Transformer")
    
    # 5. The Moment of Truth: Visualization
    plt.figure(figsize=(14, 6))
    plt.plot(actuals, label='Actual Tomato Price (₹)', color='black', linewidth=2)
    plt.plot(lstm_preds, label=f'LSTM Prediction (MAE: ₹{lstm_mae:.2f})', color='blue', linestyle='dashed')
    plt.plot(trans_preds, label=f'Transformer Prediction (MAE: ₹{trans_mae:.2f})', color='red', linestyle='dotted')
    
    plt.title('AI Model Comparison: Real-time Agricultural Price Forecasting')
    plt.xlabel('Days (Test Dataset)')
    plt.ylabel('Modal Price (₹)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("\n📈 Generating comparison graph... Close the graph window to end the script.")
    plt.show()

    # --- 6. EXPORT ASSETS FOR DEPLOYMENT ---
    import joblib
    # 1. Save PyTorch Model
    torch.save(lstm_model.state_dict(), "price_lstm_model.pth")
    # 2. Save the Scaler (Crucial for the frontend to process data)
    joblib.dump(target_scaler, "price_scaler.gz")
    print("\n🚀 Assets exported: price_lstm_model.pth & price_scaler.gz")