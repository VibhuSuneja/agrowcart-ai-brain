# AgrowCart AI Brain: Hyper-Local Agricultural Price Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.lions/PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)

An end-to-end, deep learning-based forecasting engine tailored for the **Kurukshetra Mandi**, focusing on highly perishable commodities such as tomatoes. This project fuses daily APMC market prices with exogenous NASA meteorological vectors to provide precision forecasts for smallholder farmers.

## 🚀 Key Features
- **15-Dimensional Feature Set:** Integrates Market Dynamics (Price), Climate Vectors (NASA POWER), and Cyclical Seasonality (Sin/Cos Encodings).
- **Comparative Architecture Study:** Evaluates **LSTM** vs. **Time-Series Transformer** performance in low-resource data regimes.
- **FastAPI Microservice:** Scalable, asynchronous inference engine ready for production.
- **Academic Rigor:** Accompanied by a full research paper detailing the "Mean Collapse" phenomenon in Transformers.

## 📊 The Moment of Truth (Results)
| Model | RMSE (₹) | MAE (₹) |
| :--- | :--- | :--- |
| **LSTM (Optimized)** | **91.11** | **₹72.89** |
| Transformer (Optimized) | 106.50 | ₹85.20 |

Our findings demonstrate that **LSTMs** remain superior for hyper-local forecasting (~780 time steps) due to their strong sequential inductive bias, whereas Transformers suffer from "data hunger" when applied to micro-datasets.

## 📂 Repository Structure
- `research/`: Contains the final academic manuscript.
- `src/`: Core Python engine for preprocessing, training, and inference.
- `models/`: Optimized weights and scalers for deployment.
- `datasets/`: Data acquisition scripts (Agmarknet & NASA POWER).

## 🛠️ Tech Stack
- **Languages:** Python 3.10
- **DL Frameworks:** PyTorch, TensorFlow
- **Backend:** FastAPI, Uvicorn
- **Deployment:** Render (Backend), Vercel (Frontend)

## 📖 Citation
See the full research paper in `research/Comparative_Analysis_Agri_Price_Forecasting.pdf` for a detailed analysis of the inductive bias vs. data hunger trade-off.

---
**Developed by Vibhu Suneja**  
*Empowering Farmers through Data-Driven Intelligence.*
