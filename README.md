# Stock Price Prediction using LSTM

An end-to-end deep learning system for predicting daily stock closing prices from historical OHLCV (Open, High, Low, Close, Volume) data using stacked **LSTM (Long Short-Term Memory)** recurrent neural networks in TensorFlow 2.x and Keras, accompanied by an interactive **Streamlit Web Dashboard**.

---

## 🌟 Key Features

- **Modular Architecture**: Clean, reusable functions for data preprocessing, windowed sequence generation, model training, evaluation, forecasting, and plotting.
- **Interactive Streamlit Web Dashboard (`app.py`)**:
  - Live exploration of historical OHLCV data with interactive **Plotly Candlestick** charts and technical moving averages (20/50-day SMA).
  - Hyperparameter tuning via sidebar (Lookback window, epochs, batch size, learning rate, LSTM units, dropout).
  - Real-time training progress and loss curves.
  - Multi-step forward price forecasting (1–30 days).
- **Leakage-Free Preprocessing**: Chronological splitting without data shuffling and `MinMaxScaler` normalization fitted strictly on training data.
- **Honest Naive Baseline Comparison**: Computes a $y(t) = y(t-1)$ benchmark to verify whether the model has learned authentic market signals or merely learned a trivial 1-day lag.

---

## 📂 Project Structure

```
Stock-Price-Prediction-using-LSTM/
├── README.md               <- Project documentation & guide
├── requirements.txt        <- Python dependencies
├── stock_prediction.py     <- Modular core training pipeline and CLI
├── app.py                  <- Interactive Streamlit web application
├── data/
│   └── ge.us.txt           <- Bundled General Electric historical data
├── outputs/                <- Automatically generated training runs
│   └── <timestamp>/
│       ├── best_model.keras
│       ├── training_log.csv
│       ├── train_vs_val_loss.png
│       ├── pred_vs_real.png
│       └── metrics.txt
└── .gitignore              <- Standard git ignore rules
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/<your-username>/Stock-Price-Prediction-using-LSTM.git
cd Stock-Price-Prediction-using-LSTM

# (Optional) Create and activate virtual environment
python -m venv venv
# On Windows PowerShell:
venv\Scripts\Activate.ps1
# On macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

### 2. Launch the Streamlit Web App

To view interactive charts, adjust parameters, and run predictions in your browser:

```bash
streamlit run app.py
```

The web dashboard will automatically launch at `http://localhost:8501`.

---

### 3. Run via Command Line Interface (CLI)

You can also run training directly from the command line:

```bash
# Quick test run (5 epochs)
python stock_prediction.py --epochs 5

# Full training run
python stock_prediction.py --epochs 80 --batch-size 32 --time-steps 60

# Custom CSV data file
python stock_prediction.py --data data/your_stock.csv
```

#### Available CLI Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/ge.us.txt` | Path to CSV/TXT file with OHLCV columns |
| `--time-steps` | `60` | Lookback window (days of history per sample) |
| `--batch-size` | `32` | Training mini-batch size |
| `--epochs` | `80` | Maximum number of training epochs |
| `--lr` | `0.001` | Learning rate for optimizer |
| `--train-frac` | `0.8` | Proportion of data for training |
| `--seed` | `42` | Random seed for reproducibility |
| `--output-dir` | `None` | Custom output directory path |

---

## 📊 Evaluation & The Naive Baseline Trap

Stock prices closely mimic a **random walk**. Standard metrics (like high $R^2$ or tight visual fit on charts) can be misleading because models often learn to output the previous day's price:

$$\hat{y}_{naive}(t) = y(t-1)$$

This repository explicitly tests the model against this persistence baseline:
- **LSTM RMSE vs. Naive RMSE**
- **Directional Accuracy (%)** (predicting whether tomorrow's price goes up or down)

If the model beats the naive baseline, it has extracted true predictive signal beyond simple inertia.

---

## 🛠️ Pushing This Project to GitHub

To push your local repository to GitHub:

```bash
# 1. Initialize git (if not already initialized)
git init

# 2. Stage all files
git add .

# 3. Commit the changes
git commit -m "Initial commit: Modular LSTM stock prediction pipeline and Streamlit dashboard"

# 4. Set default branch to main
git branch -M main

# 5. Link to your GitHub repository
git remote add origin https://github.com/<YOUR-USERNAME>/<YOUR-REPO-NAME>.git

# 6. Push to GitHub
git push -u origin main
```

---

## 📜 License

This project is licensed under the Apache-2.0 License.
