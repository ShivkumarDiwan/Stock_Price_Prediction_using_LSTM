"""
Stock Price Prediction using an LSTM Neural Network
===================================================

Predicts the daily closing price of a stock from its historical OHLCV data
using a Keras/TensorFlow LSTM. Features modular functions, robust data scaling,
an honest naive baseline comparison, and future price forecasting.

Usage Examples (CLI):
--------------------
    # Default run on bundled GE data
    python stock_prediction.py

    # Quick test run
    python stock_prediction.py --epochs 5

    # Custom CSV file
    python stock_prediction.py --data data/ge.us.txt --epochs 50 --batch-size 32

Web Dashboard (Streamlit):
-------------------------
    streamlit run app.py
"""

import os
import time
import argparse
import random
from typing import Dict, Tuple, Any, Optional, List, Union

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Headless mode for server / terminal environments
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Suppress verbose TensorFlow startup logs
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from tensorflow.keras.optimizers import RMSprop, Adam
except ImportError:
    tf = None

# Default feature columns and target column
FEATURE_COLS: List[str] = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COL: str = "Close"


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across random, numpy, and tensorflow."""
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="LSTM Stock Price Prediction Pipeline")
    p.add_argument("--data", default="data/ge.us.txt",
                   help="Path to CSV/TXT file with Open, High, Low, Close, Volume columns")
    p.add_argument("--time-steps", type=int, default=60,
                   help="Lookback window (number of past days used for prediction)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Training batch size")
    p.add_argument("--epochs", type=int, default=80,
                   help="Maximum number of training epochs")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate for optimizer")
    p.add_argument("--train-frac", type=float, default=0.8,
                   help="Fraction of data used for training (remaining is validation/test)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Custom output directory to save results")
    return p.parse_args()


def load_dataset(source: Union[str, pd.DataFrame], feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load stock dataset from file path or existing DataFrame and validate required columns.

    Parameters
    ----------
    source : str or pd.DataFrame
        Path to CSV/TXT file or an existing pandas DataFrame.
    feature_cols : list of str, optional
        List of required feature columns (default: Open, High, Low, Close, Volume).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with sorted index and validated columns.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Data file not found at: {source}")
        df = pd.read_csv(source)

    # Normalize column names (strip whitespace)
    df.columns = [c.strip() for c in df.columns]

    # Validate required columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}. Available columns: {list(df.columns)}")

    # Parse Date column if available
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

    # Drop missing values in feature columns
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


def build_timeseries(mat: np.ndarray, target_index: int, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a 2D feature matrix (rows=days, cols=features) into supervised LSTM samples.

    Parameters
    ----------
    mat : np.ndarray
        2D array of shape (num_days, num_features).
    target_index : int
        Column index of the target feature (e.g., Close).
    time_steps : int
        Window size of consecutive days to use as input sequence.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: shape (num_samples, time_steps, num_features)
        y: shape (num_samples,)
    """
    num_samples = mat.shape[0] - time_steps
    if num_samples <= 0:
        raise ValueError(f"Not enough data rows ({mat.shape[0]}) for time_steps={time_steps}.")

    num_features = mat.shape[1]
    x = np.zeros((num_samples, time_steps, num_features), dtype=np.float32)
    y = np.zeros((num_samples,), dtype=np.float32)

    for i in range(num_samples):
        x[i] = mat[i:i + time_steps]
        y[i] = mat[i + time_steps, target_index]

    return x, y


def prepare_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = TARGET_COL,
    train_frac: float = 0.8,
    time_steps: int = 60
) -> Dict[str, Any]:
    """
    Perform chronological train/val/test splitting, MinMax scaling on train set only
    (preventing future lookahead leakage), and windowed sequence generation.

    Parameters
    ----------
    df : pd.DataFrame
        Stock DataFrame containing OHLCV features.
    feature_cols : list of str, optional
        Features to scale and feed into model.
    target_col : str
        Target column name to predict.
    train_frac : float
        Fraction of data for training (remainder split 50/50 between val and test).
    time_steps : int
        Sliding window size.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing train/val/test arrays, fitted scaler, DataFrames, and target index.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    target_index = feature_cols.index(target_col)

    # 1. Chronological Split (No random shuffling for time-series)
    total_len = len(df)
    split_idx = int(total_len * train_frac)
    df_train = df.iloc[:split_idx].copy()
    df_test_full = df.iloc[split_idx:].copy()

    # 2. MinMax Scaling (Fit ONLY on training set to prevent data leakage)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(df_train[feature_cols].values)
    scaled_test_full = scaler.transform(df_test_full[feature_cols].values)

    # 3. Create Supervised Sequences
    x_train, y_train = build_timeseries(scaled_train, target_index, time_steps)
    x_temp, y_temp = build_timeseries(scaled_test_full, target_index, time_steps)

    # Split test portion in half: validation set & final held-out test set
    half = len(x_temp) // 2
    x_val, x_test = x_temp[:half], x_temp[half:]
    y_val, y_test = y_temp[:half], y_temp[half:]

    # Corresponding dates for test set if available
    test_dates = None
    if "Date" in df_test_full.columns:
        test_dates = df_test_full["Date"].iloc[time_steps + half:].reset_index(drop=True)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "scaler": scaler,
        "target_index": target_index,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "df_train": df_train,
        "df_test_full": df_test_full,
        "test_dates": test_dates,
        "time_steps": time_steps,
    }


def create_model(
    time_steps: int,
    num_features: int,
    lr: float = 1e-3,
    lstm_units: Tuple[int, int] = (100, 60),
    dropout_rate: float = 0.4,
    dense_units: int = 20
) -> Any:
    """
    Construct and compile stacked LSTM neural network model.

    Parameters
    ----------
    time_steps : int
        Input sequence length.
    num_features : int
        Number of feature columns.
    lr : float
        Learning rate.
    lstm_units : tuple of int
        Units for (1st LSTM layer, 2nd LSTM layer).
    dropout_rate : float
        Dropout probability after LSTM layers.
    dense_units : int
        Units for dense hidden layer before output.

    Returns
    -------
    tf.keras.Model
        Compiled Keras sequential model.
    """
    if tf is None:
        raise ImportError("TensorFlow is required to build the LSTM model. Please install tensorflow.")

    model = Sequential([
        Input(shape=(time_steps, num_features)),
        LSTM(lstm_units[0], return_sequences=True, kernel_initializer="random_uniform"),
        Dropout(dropout_rate),
        LSTM(lstm_units[1]),
        Dropout(dropout_rate),
        Dense(dense_units, activation="relu"),
        Dense(1, activation="sigmoid"),  # Output scaled between 0 and 1
    ])

    optimizer = RMSprop(learning_rate=lr)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])
    return model


def train_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 80,
    batch_size: int = 32,
    callbacks: Optional[List[Any]] = None,
    verbose: int = 1
) -> Any:
    """
    Train the compiled LSTM model on train data with validation tracking.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled model.
    x_train, y_train : np.ndarray
        Training samples and targets.
    x_val, y_val : np.ndarray
        Validation samples and targets.
    epochs : int
        Max epochs.
    batch_size : int
        Batch size.
    callbacks : list, optional
        Keras callbacks (EarlyStopping, Checkpoints, etc.).
    verbose : int
        Logging verbosity (0=silent, 1=progress bar, 2=one line per epoch).

    Returns
    -------
    tf.keras.callbacks.History
        Training history object.
    """
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,  # Preserve sequential temporal order
        validation_data=(x_val, y_val),
        callbacks=callbacks or [],
        verbose=verbose,
    )
    return history


def predict_prices(
    model: Any,
    x_data: np.ndarray,
    y_scaled: np.ndarray,
    scaler: MinMaxScaler,
    target_index: int,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate model predictions and unscale both predicted and real prices to original currency scale.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    x_data : np.ndarray
        Input sequences.
    y_scaled : np.ndarray
        Normalized true target values.
    scaler : MinMaxScaler
        Fitted scaler used for inverse transformation.
    target_index : int
        Index of the target column in scaler.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (y_pred_price, y_true_price) in actual stock price units.
    """
    y_pred_scaled = model.predict(x_data, batch_size=batch_size, verbose=0).flatten()

    data_range = scaler.data_range_[target_index]
    data_min = scaler.data_min_[target_index]

    y_pred_price = y_pred_scaled * data_range + data_min
    y_true_price = y_scaled * data_range + data_min

    return y_pred_price, y_true_price


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Calculate performance metrics for LSTM model and compare against an honest Naive Baseline
    (Tomorrow = Today: y_pred_naive(t) = y_true(t-1)).

    Parameters
    ----------
    y_true : np.ndarray
        Actual stock prices.
    y_pred : np.ndarray
        Predicted stock prices.

    Returns
    -------
    Dict[str, Any]
        Dictionary of RMSE, MAE, MAPE, directional accuracy, baseline metrics, and verdict.
    """
    # LSTM Metrics
    lstm_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    lstm_mae = float(mean_absolute_error(y_true, y_pred))
    lstm_mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    # Directional Accuracy (Did the model predict the correct direction of change?)
    if len(y_true) > 1:
        actual_diff = np.diff(y_true)
        pred_diff = y_pred[1:] - y_true[:-1]
        directional_acc = float(np.mean((actual_diff * pred_diff) > 0) * 100)
    else:
        directional_acc = 0.0

    # Naive Baseline: Predict tomorrow's price is equal to today's price (y_t = y_{t-1})
    naive_pred = y_true[:-1]
    naive_true = y_true[1:]
    naive_rmse = float(np.sqrt(mean_squared_error(naive_true, naive_pred)))
    naive_mae = float(mean_absolute_error(naive_true, naive_pred))
    naive_mape = float(np.mean(np.abs((naive_true - naive_pred) / (naive_true + 1e-8))) * 100)

    beats_baseline = bool(lstm_rmse < naive_rmse)

    if beats_baseline:
        improvement_pct = ((naive_rmse - lstm_rmse) / naive_rmse) * 100
        verdict = f"SUCCESS: LSTM BEATS the naive baseline by {improvement_pct:.2f}% RMSE reduction."
    else:
        gap_pct = ((lstm_rmse - naive_rmse) / naive_rmse) * 100
        verdict = (
            f"CAUTION: LSTM does NOT beat the naive baseline (+{gap_pct:.2f}% RMSE). "
            "The model primarily mirrors the 1-day lag characteristic of financial random walks."
        )

    return {
        "lstm_rmse": lstm_rmse,
        "lstm_mae": lstm_mae,
        "lstm_mape": lstm_mape,
        "directional_accuracy_pct": directional_acc,
        "naive_rmse": naive_rmse,
        "naive_mae": naive_mae,
        "naive_mape": naive_mape,
        "beats_baseline": beats_baseline,
        "verdict": verdict,
    }


def plot_loss(history: Any, save_path: Optional[str] = None) -> plt.Figure:
    """Generate and optionally save training vs validation loss curve."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history.history["loss"], label="Train Loss (MSE)", color="#2563eb", lw=2)
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="Val Loss (MSE)", color="#ea580c", lw=2)
    ax.set_title("LSTM Model Loss During Training", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss (Mean Squared Error)", fontsize=11)
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: Optional[pd.Series] = None,
    title: str = "Actual vs Predicted Stock Price (Test Set)",
    save_path: Optional[str] = None
) -> plt.Figure:
    """Generate and optionally save actual vs predicted price chart."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x_axis = dates if dates is not None and len(dates) == len(y_true) else np.arange(len(y_true))

    ax.plot(x_axis, y_true, label="Actual Price", color="#059669", lw=2)
    ax.plot(x_axis, y_pred, label="LSTM Prediction", color="#dc2626", lw=1.8, linestyle="--")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date / Trading Days", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.legend(frameon=True, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def forecast_future(
    model: Any,
    last_window_scaled: np.ndarray,
    scaler: MinMaxScaler,
    target_index: int,
    num_days: int = 7
) -> List[float]:
    """
    Perform multi-step autoregressive forecasting for future N days.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    last_window_scaled : np.ndarray
        Last known window of shape (time_steps, num_features).
    scaler : MinMaxScaler
        Fitted scaler.
    target_index : int
        Index of target feature.
    num_days : int
        Number of forward days to project.

    Returns
    -------
    List[float]
        Projected stock prices for each future day.
    """
    current_window = last_window_scaled.copy()  # shape: (time_steps, num_features)
    data_range = scaler.data_range_[target_index]
    data_min = scaler.data_min_[target_index]

    future_prices: List[float] = []

    for _ in range(num_days):
        x_input = current_window[np.newaxis, :, :]  # (1, time_steps, num_features)
        pred_scaled = float(model.predict(x_input, verbose=0)[0, 0])
        pred_price = pred_scaled * data_range + data_min
        future_prices.append(pred_price)

        # Roll the window forward: create new row with predicted target and repeated other features
        new_row = current_window[-1].copy()
        new_row[target_index] = pred_scaled
        current_window = np.vstack([current_window[1:], new_row])

    return future_prices


def run_pipeline(
    data_path: str = "data/ge.us.txt",
    time_steps: int = 60,
    batch_size: int = 32,
    epochs: int = 80,
    lr: float = 1e-3,
    train_frac: float = 0.8,
    seed: int = 42,
    output_dir: Optional[str] = None,
    verbose: int = 1
) -> Dict[str, Any]:
    """
    Execute end-to-end LSTM training and evaluation pipeline.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing pipeline results, metrics, output paths, and history.
    """
    set_random_seed(seed)
    start_time = time.time()

    if output_dir is None:
        output_dir = os.path.join("outputs", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=======================================================")
    print(f" LSTM Stock Prediction Pipeline")
    print(f" Outputs directory: {output_dir}")
    print(f"=======================================================\n")

    # 1. Load Data
    print(f"[1/6] Loading data from: {data_path}")
    df = load_dataset(data_path, FEATURE_COLS)
    print(f"      Total rows: {len(df)} | Columns: {FEATURE_COLS}")

    # 2. Preprocess & Split
    print(f"[2/6] Preparing sequences (lookback={time_steps} days, train_frac={train_frac})...")
    data_dict = prepare_data(
        df,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        train_frac=train_frac,
        time_steps=time_steps
    )
    x_train, y_train = data_dict["x_train"], data_dict["y_train"]
    x_val, y_val = data_dict["x_val"], data_dict["y_val"]
    x_test, y_test = data_dict["x_test"], data_dict["y_test"]
    scaler = data_dict["scaler"]
    target_idx = data_dict["target_index"]

    print(f"      Train samples: {x_train.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")

    # 3. Build Model
    print(f"[3/6] Building LSTM architecture...")
    model = create_model(time_steps, x_train.shape[2], lr=lr)
    if verbose:
        model.summary()

    # 4. Train Model
    model_path = os.path.join(output_dir, "best_model.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=30,
                      min_delta=1e-4, verbose=1, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True,
                        mode="min", verbose=0),
        CSVLogger(os.path.join(output_dir, "training_log.csv"), append=False),
    ]

    print(f"[4/6] Training LSTM for up to {epochs} epochs (batch_size={batch_size})...")
    history = train_model(
        model, x_train, y_train, x_val, y_val,
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=verbose
    )

    # 5. Evaluate on Held-out Test Set
    print(f"[5/6] Evaluating best saved model on held-out test set...")
    if os.path.exists(model_path):
        best_model = load_model(model_path)
    else:
        best_model = model

    y_pred_price, y_true_price = predict_prices(
        best_model, x_test, y_test, scaler, target_idx, batch_size=batch_size
    )
    metrics = calculate_metrics(y_true_price, y_pred_price)

    metrics_text = (
        f"=======================================================\n"
        f" Performance Evaluation & Baseline Comparison\n"
        f"=======================================================\n"
        f"LSTM Model   -> RMSE: {metrics['lstm_rmse']:.4f} | MAE: {metrics['lstm_mae']:.4f} | MAPE: {metrics['lstm_mape']:.2f}%\n"
        f"Naive Model  -> RMSE: {metrics['naive_rmse']:.4f} | MAE: {metrics['naive_mae']:.4f} | MAPE: {metrics['naive_mape']:.2f}%\n"
        f"Directional Accuracy: {metrics['directional_accuracy_pct']:.2f}%\n"
        f"\nVerdict:\n{metrics['verdict']}\n"
        f"=======================================================\n"
    )
    print("\n" + metrics_text)

    metrics_file = os.path.join(output_dir, "metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(metrics_text)

    # 6. Save Plots
    print(f"[6/6] Generating and saving visualization plots...")
    loss_plot_path = os.path.join(output_dir, "train_vs_val_loss.png")
    plot_loss(history, save_path=loss_plot_path)
    plt.close()

    pred_plot_path = os.path.join(output_dir, "pred_vs_real.png")
    plot_predictions(
        y_true_price, y_pred_price,
        dates=data_dict["test_dates"],
        save_path=pred_plot_path
    )
    plt.close()

    elapsed = (time.time() - start_time) / 60
    print(f"Pipeline completed in {elapsed:.2f} minutes.")
    print(f"All artifacts saved to: {output_dir}\n")

    return {
        "model": best_model,
        "history": history,
        "metrics": metrics,
        "y_true_price": y_true_price,
        "y_pred_price": y_pred_price,
        "output_dir": output_dir,
        "data_dict": data_dict
    }


def main():
    """CLI execution entrypoint."""
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        time_steps=args.time_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        train_frac=args.train_frac,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=1
    )


if __name__ == "__main__":
    main()
