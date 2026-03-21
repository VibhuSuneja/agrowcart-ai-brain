Research Project: AgrowCart AI Price Forecasting Engine

Author: Vibhu Suneja | 3rd Year B.Tech CSE, JMIT Kurukshetra
Domain: Applied Machine Learning, Time-Series Forecasting, Hyper-Local E-Commerce
Target Venue: IEEE/Springer Undergraduate Conference or ArXiv preprint.

1. The Core Objective

To conduct a comparative analysis of deep learning models (LSTM, GRU, and Temporal Fusion Transformers) for predicting the real-time, hyper-local price volatility of agricultural commodities (Tomato, Onion, Potato) in the Haryana/Punjab region.

The "Why": Standard agricultural apps only show current prices. AgrowCart will predict future prices using historical mandi data and local weather patterns, solving the information asymmetry for farmers.

2. Technical Architecture

Data Source: Agmarknet (GoI) & OpenWeather API.

Data Processing: Python, Pandas, NumPy, Scikit-learn.

ML Framework: TensorFlow/Keras or PyTorch.

Backend Interface: FastAPI or Flask (Serving the model).

Frontend Integration: AgrowCart MERN Stack (React frontend calls the Python API).

3. Phase-by-Phase Roadmap (Action-Oriented)

Phase 1: The Data Foundation (Days 1-4)

Goal: Secure and clean the raw material.

[ ] Acquire Market Data: Download 2024-2026 daily prices for target crops in Kurukshetra/Ambala mandis.

[ ] Acquire Weather Data: Download historical temperature and rainfall data for the same dates.

[ ] Data Merging: Write a Python script to merge these datasets on the Date column.

[ ] Data Cleaning: Handle missing values (NaNs) using interpolation. Remove extreme outliers caused by manual data entry errors.

[ ] Feature Engineering: Create new columns like 7_day_moving_average or price_momentum.

Phase 2: Baseline Models (Days 5-10)

Goal: Prove that the data can be predicted using standard methods.

[ ] Train-Test Split: Split data chronologically (e.g., train on 2024-2025, test on 2026).

[ ] Build LSTM: Implement a basic Long Short-Term Memory network.

[ ] Build GRU: Implement a Gated Recurrent Unit network.

[ ] Evaluate: Record the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) for both.

Phase 3: The State-of-the-Art (Days 11-18)

Goal: Implement the Transformer model to handle sudden spikes (weather shocks).

[ ] Build Transformer: Implement a Time-Series Transformer (e.g., using HuggingFace's Time Series library or PyTorch).

[ ] Hyperparameter Tuning: Adjust learning rates, batch sizes, and attention heads.

[ ] The Ultimate Comparison: Graph the predictions of LSTM, GRU, and Transformers against the actual historical prices. This graph is the centerpiece of your paper.

Phase 4: System Integration (Days 19-24)

Goal: Move from Jupyter Notebook to a real-world application.

[ ] Export Model: Save the best-performing model as a .h5 or .pt file.

[ ] Create API: Wrap the model in a FastAPI endpoint (/predict-price?crop=tomato&mandi=kurukshetra).

[ ] MERN Connection: Have your AgrowCart Node.js backend ping this Python API and display the predicted price on the React frontend.

Phase 5: Paper Drafting (Days 25-30)

Goal: Translate code into academic literature.

[ ] Abstract: 250 words summarizing the problem, method, and results.

[ ] Introduction: The socio-economic problem of price volatility in India.

## Phase 2: Model Research & Comparative Analysis

### Methodology
For this research, we implemented two distinct neural network architectures to predict tomato price volatility:
1. **Long Short-Term Memory (LSTM)**: Chosen for its ability to handle sequential dependencies and maintain a hidden state over time.
2. **Time-Series Transformer**: Utilized multi-head attention mechanisms to capture global dependencies across a 14-day window.

**Data Configuration**:
- **Lookback Window**: 14 days
- **Forecast Horizon**: 1 day
- **Features**: 11 numeric features (Prices, Weather, Rolling Averages, Volatility, Lags)
- **Train/Test Split**: 80/20 Chronological split

### Results & Discussion
| Metric | LSTM (Tuned) | Transformer (Tuned) |
| :--- | :--- | :--- |
| **MAE (Mean Absolute Error)** | **₹79.38** | **₹84.33** |
| **Response Type** | Smooth / Trend-following | Jagged / Reactive |

**Key Findings**:
- **Data Efficiency**: The LSTM performed better on the restricted dataset (~800 rows), demonstrating higher stability in trend-following.
- **Model Reactivity**: The Transformer, after hyperparameter tuning (150 epochs, lower learning rate), showed a higher sensitivity to price shocks compared to its initial "flatline" state.
- **The "Shock" Hypothesis**: While the LSTM is more accurate on average, the Transformer's attention mechanism provides better qualitative insights into price directions during weather-induced volatility.

## Phase 3: System Integration (FastAPI Microservice)
**STATUS: COMPLETED & DEPLOYED (March 2026)**
- **API Construction**: Successfully wrapped the PyTorch LSTM model in a fully asynchronous Python FastAPI microservice (`api_server.py`).
- **Cloud Deployment**: Deployed the microservice to Render (`agrowcart-ai-brain.onrender.com`) using a standalone GitHub repository to maintain a clean Microservice Architecture.
- **Frontend Integration**: Successfully wired the live Vercel Next.js frontend (`usePricePrediction.ts`) to send real-time user requests to the Render API, achieving full end-to-end functionality.

## Phase 4: The "All-India Millet" Expansion (Current Focus)
**STATUS: IN PROGRESS**
- Transitioning the core engine from a single crop (Tomato) to a nationwide Millet Forecasting Engine (covering Jowar, Bajra, Ragi, etc.).
- Designing a direct data pipeline to the Open Government Data (OGD) Platform (data.gov.in) to ingest official Agmarknet APMC Mandi prices, making the research thesis highly credible and scalable.

## Phase 5: Paper Drafting
- [ ] Abstract & Introduction
- [ ] Results: The comparison tables (MAE, Latency) and the prediction graphs.
- [ ] Conclusion: Why LSTMs perform optimally for hyper-local Indian datasets, and the architectural benefits of decoupling the AI Brain from the Node.js frontend.

4. Anti-Procrastination Rules (Personal Protocol)

No New Ideas: Until Phase 3 is complete, no starting new side projects or pivoting to different tech stacks.

The 25-Minute Rule: When stuck on data cleaning (the most boring part), use a strict Pomodoro timer. Just 25 minutes of pure Python, then walk away.

Visual Proof First: Don't obsess over perfect code. Focus on getting a visual graph output as fast as possible to keep the intellectual momentum high.