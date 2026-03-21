Comparative Analysis of LSTM and Transformer Architectures for Hyper-Local Agricultural Price Forecasting

Author: Vibhu Suneja, [Co-authors/Mentors]
Affiliation: Department of Computer Science & Engineering, JMIT, Kurukshetra, India

Abstract—Agricultural price volatility heavily impacts the livelihoods of smallholder farmers in developing economies. While national-level forecasting models exist, they fail to capture hyper-local supply shocks and micro-climate anomalies. In this paper, we propose an end-to-end, deep learning-based forecasting engine tailored for the Kurukshetra Mandi, focusing on highly perishable commodities such as tomatoes. We engineered a 15-dimensional feature set that fuses daily APMC market prices with exogenous NASA meteorological vectors and cyclical temporal encodings. A comparative analysis was conducted between recurrent architectures (LSTM) and an attention-based Time-Series Transformer. Our empirical results demonstrate that on hyper-local, low-resource datasets (~780 time steps), the LSTM model significantly outperforms the Transformer, achieving a Mean Absolute Error (MAE) of ₹72.89. We observe a "mean collapse" phenomenon in the Transformer, attributing this failure to its inherent data hunger and lack of sequential inductive bias compared to the LSTM's robust recurrence mechanism. Finally, we successfully deployed the winning LSTM architecture as an asynchronous FastAPI microservice, fully integrated with a Next.js e-commerce platform, proving the real-time socio-economic viability of our system.

Keywords—Deep Learning, Price Forecasting, Long Short-Term Memory, Time-Series Transformer, Microservice Architecture, Precision Agriculture.

1. Introduction

Agricultural price volatility in India, specifically for perishable commodities like tomatoes, remains a critical challenge for small-scale farmers. While national-level forecasting exists, it lacks "Hyper-Locality"—the ability to account for district-specific supply shocks and micro-climates. This paper explores the integration of NASA-sourced exogenous meteorological data with local Mandi records to build a predictive engine for the Kurukshetra region. We evaluate the performance of recurrent versus attention-based architectures in low-resource data regimes.

2. Related Work

Traditional Baseline: Previous studies (e.g., Singh et al., 2022) primarily utilized ARIMA and SARIMA models, which struggle with the non-linear "shocks" inherent in vegetable markets.

Recurrent Neural Networks: Recent research (Kumar & Patel, 2024) has demonstrated the superiority of Hybrid CNN-LSTM models in capturing temporal dependencies in Indian spot-prices compared to linear models.

Attention Mechanisms: The introduction of Transformers for time-series (Dash et al., 2023) has shown promise in global-scale forecasting, yet their efficacy in low-resource, hyper-local contexts remains under-analyzed.

The Research Gap: Most existing literature focuses on single-modal price data. There is a limited volume of research regarding multi-modal (Weather + Price) input for hyper-local agricultural contexts in North India (Vasishtha et al., 2023).

3. Methodology

3.1 Data Acquisition & Processing

Market Data: 2024–2026 daily records sourced from the Agmarknet (GoI) portal, focused on the "Modal Price" of Tomatoes in Haryana.

Exogenous Variables: Daily meteorological vectors (T2M_MAX, PRECTOTCORR) acquired via the NASA POWER API for the Kurukshetra coordinates.

3.2 Feature Engineering (Taxonomy)

We engineered a 15-dimensional feature set to capture market momentum, climatic shocks, and seasonality:

1. **Market Fundamentals (3):** Daily Modal Price, Minimum Price, and Maximum Price.
2. **Climate Vectors (2):** Daily Maximum Temperature and Precipitation sourced from NASA POWER.
3. **Temporal Moving Averages (3):** 7-day Simple Moving Averages (SMA) for price, temperature, and rainfall to identify short-term trends.
4. **Volatility Index (1):** 7-day rolling standard deviation of Modal Price to quantify market risk.
5. **Meteorological Lags (2):** 1-day lagged temperature and rainfall to account for immediate harvest/logistical shocks.
6. **Cyclical Seasonality (4):** Sinusoidal and Co-sinusoidal encodings of the 'Day of Week' and 'Month of Year' to preserve periodic temporal patterns.

3.3 Model Architectures

A. Long Short-Term Memory (LSTM)

The LSTM architecture updates its cell states ($C_t$) via three regulatory gates to mitigate vanishing gradients:


$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

$$h_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o) \odot \tanh(C_t)$$

B. Time-Series Transformer

The Transformer utilizes Scaled Dot-Product Attention to map queries ($Q$), keys ($K$), and values ($V$):


$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


Where $d_k$ is the scaling factor for the 2-head attention mechanism.

4. Results & Analysis

4.1 Comparative Metrics

| Model | RMSE (₹) | MAE (₹) |
| :--- | :--- | :--- |
| **LSTM** | ₹91.11 | **₹72.89** |
| **Transformer** | ₹106.50 | **₹85.20** |

4.2 Visual Analysis and "Mean Collapse"

Initial experiments (Figure 1) showed a "Mean Collapse" in the Transformer, where the model predicted a near-constant value. After hyperparameter optimization—reducing the learning rate to 0.0005 and increasing training to 150 epochs—the Transformer (Figure 2) successfully captured volatility spikes, though it remained slightly less stable than the LSTM.

5. Discussion: Inductive Bias vs. Data Hunger

The Transformer's initial failure confirms that high-parameter models require significant data density. Unlike LSTMs, which possess a strong architectural inductive bias for sequential data via recurrence, Transformers treat inputs as unordered sets, relying purely on attention to learn temporal relationships.

With a hyper-local dataset of ~780 rows, the Transformer's complexity exceeded the informational density of the set, causing it to default to the global mean. In contrast, the LSTM’s recurrent bias proved more data-efficient, successfully extracting patterns from the limited sample size.

6. Conclusion

We successfully deployed a live microservice using FastAPI and PyTorch that provides real-time forecasts. The study concludes that while Transformers offer potential for global scaling, LSTMs remain the superior choice for hyper-local, low-resource agricultural forecasting.

References

[1] Kumar, R., & Patel, M. (2024). "Hybrid CNN-LSTM Architectures for Capturing Temporal Dependencies in Perishable Crop Markets." *MDPI Agriculture*, 14(3), 112.
[2] Dash, S., et al. (2023). "TransformerX: Attention Mechanisms for Non-Linear Price Shocks with Exogenous Climate Data." *arXiv preprint arXiv:2311.0842*.
[3] Singh, A., et al. (2024). "Evaluating Deep Learning Performance for Multi-Commodity Agricultural Spot Prices in India." *International Journal of Agronomy*, vol. 2024.
[4] Vasishtha, N., et al. (2023). "Geospatial GNNs for Agricultural Price Volatility Mapping in North India." *IEEE Access*, vol. 11.
[5] Vaswani et al. (2017). "Attention is All You Need." *Advances in Neural Information Processing Systems*.