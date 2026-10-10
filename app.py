"""
Streamlit Web Application: Stock Price Prediction using LSTM Neural Network
===========================================================================
An interactive, modern web dashboard for training LSTM models on historical OHLCV
stock data, evaluating predictions against a realistic naive baseline, and
forecasting future price trajectories.
"""

import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import modular functions from stock_prediction.py
from stock_prediction import (
    load_dataset,
    prepare_data,
    create_model,
    train_model,
    predict_prices,
    calculate_metrics,
    forecast_future,
    FEATURE_COLS,
    TARGET_COL,
    set_random_seed,
)

# Set page config
st.set_page_config(
    page_title="Stock Price Prediction | LSTM Neural Network",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for modern dark/light glassmorphic UI
st.markdown("""
<style>
    .main-title-container {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.2rem;
    }
    .main-title-icon {
        font-size: 2.3rem;
        line-height: 1;
    }
    .main-title-text {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #10b981, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(226, 232, 240, 0.15);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #3b82f6;
    }
    .metric-lbl {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .verdict-box-good {
        background: rgba(16, 185, 129, 0.12);
        border-left: 5px solid #10b981;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .verdict-box-warn {
        background: rgba(245, 158, 11, 0.12);
        border-left: 5px solid #f59e0b;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Helper to load sample or uploaded dataset ---
@st.cache_data(show_spinner=False)
def get_dataset(source_type: str, uploaded_file: Optional[Any] = None) -> pd.DataFrame:
    """Load dataset based on sidebar user selection."""
    if source_type == "Bundled General Electric (GE)":
        default_path = os.path.join("data", "ge.us.txt")
        if not os.path.exists(default_path):
            st.error("Bundled dataset `data/ge.us.txt` not found.")
            return pd.DataFrame()
        return load_dataset(default_path)
    elif source_type == "Upload Custom CSV":
        if uploaded_file is not None:
            return load_dataset(uploaded_file)
        return pd.DataFrame()
    return pd.DataFrame()


def render_header():
    """Render top page header."""
    st.markdown(
        """
        <div class="main-title-container">
            <span class="main-title-icon">📈</span>
            <span class="main-title-text">Stock Price Prediction using LSTM</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="sub-title">Deep Learning Time-Series Forecasting with Honest Naive Baseline Comparison & Interactive Analysis</div>',
        unsafe_allow_html=True
    )


def sidebar_controls():
    """Build interactive sidebar parameters."""
    st.sidebar.header("⚙️ Data & Hyperparameters")

    data_source = st.sidebar.selectbox(
        "Data Source",
        ["Bundled General Electric (GE)", "Upload Custom CSV"],
        index=0
    )

    uploaded_file = None

    if data_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV / TXT",
            type=["csv", "txt"],
            help="Must include Open, High, Low, Close, Volume columns"
        )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧠 Model Architecture")

    time_steps = st.sidebar.slider(
        "Lookback Window (Days)",
        min_value=10,
        max_value=120,
        value=60,
        step=5,
        help="Number of previous consecutive trading days used to predict the next day's price."
    )

    train_frac = st.sidebar.slider(
        "Training Data Fraction",
        min_value=0.5,
        max_value=0.9,
        value=0.8,
        step=0.05,
        help="Proportion of data used for training. The remaining data is split equally for validation & test."
    )

    epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=5,
        max_value=120,
        value=30,
        step=5,
        help="Number of complete training passes over the dataset."
    )

    batch_size = st.sidebar.select_slider(
        "Batch Size",
        options=[16, 32, 64, 128],
        value=32
    )

    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001
    )

    lstm_units_1 = st.sidebar.slider("LSTM Layer 1 Units", 32, 128, 100, 16)
    lstm_units_2 = st.sidebar.slider("LSTM Layer 2 Units", 16, 96, 60, 16)
    dropout_rate = st.sidebar.slider("Dropout Rate", 0.0, 0.6, 0.4, 0.1)

    seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    train_button = st.sidebar.button("🚀 Train & Evaluate Model", type="primary", use_container_width=True)

    return {
        "data_source": data_source,
        "uploaded_file": uploaded_file,
        "time_steps": time_steps,
        "train_frac": train_frac,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "lstm_units": (lstm_units_1, lstm_units_2),
        "dropout_rate": dropout_rate,
        "seed": seed,
        "train_button": train_button,
    }


def main():
    render_header()
    controls = sidebar_controls()

    # Load selected data
    df = get_dataset(
        controls["data_source"],
        controls["uploaded_file"]
    )

    if df.empty:
        st.info("👈 Please select or upload a valid dataset from the sidebar to begin.")
        return

    # Create top level tabs
    tab_data, tab_training, tab_predictions, tab_forecast, tab_docs = st.tabs([
        "📊 Historical Data",
        "🧠 Model Training",
        "📈 Predictions & Baseline",
        "🔮 Future Forecast",
        "📖 Documentation"
    ])

    # -------------------------------------------------------------
    # TAB 1: Historical Data & Technicals
    # -------------------------------------------------------------
    with tab_data:
        st.subheader("Historical Stock Price & Volume Overview")

        # Summary Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        latest_close = float(df["Close"].iloc[-1])
        min_close = float(df["Close"].min())
        max_close = float(df["Close"].max())
        avg_vol = float(df["Volume"].mean())
        num_rows = len(df)

        m1.metric("Total Trading Days", f"{num_rows:,}")
        m2.metric("Latest Close", f"${latest_close:.2f}")
        m3.metric("All-Time High", f"${max_close:.2f}")
        m4.metric("All-Time Low", f"${min_close:.2f}")
        m5.metric("Avg Daily Volume", f"{avg_vol/1e6:.2f}M")

        # Plotly Candlestick / Line Chart
        fig_price = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Price & 20/50-Day Moving Averages", "Daily Trading Volume"),
            row_heights=[0.7, 0.3]
        )

        x_axis = df["Date"] if "Date" in df.columns else df.index

        # Candlestick or Line
        if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
            fig_price.add_trace(
                go.Candlestick(
                    x=x_axis,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="OHLC Price"
                ),
                row=1, col=1
            )
        else:
            fig_price.add_trace(
                go.Scatter(x=x_axis, y=df["Close"], name="Close Price", line=dict(color="#3b82f6", width=1.8)),
                row=1, col=1
            )

        # Add moving averages
        if len(df) >= 50:
            ma20 = df["Close"].rolling(20).mean()
            ma50 = df["Close"].rolling(50).mean()
            fig_price.add_trace(go.Scatter(x=x_axis, y=ma20, name="20-Day SMA", line=dict(color="#f59e0b", width=1.2)), row=1, col=1)
            fig_price.add_trace(go.Scatter(x=x_axis, y=ma50, name="50-Day SMA", line=dict(color="#8b5cf6", width=1.2)), row=1, col=1)

        # Volume Chart
        fig_price.add_trace(
            go.Bar(x=x_axis, y=df["Volume"], name="Volume", marker=dict(color="#64748b", opacity=0.6)),
            row=2, col=1
        )

        fig_price.update_layout(
            height=580,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=40, b=20),
            hovermode="x unified"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        with st.expander("📄 View Raw Dataset"):
            st.dataframe(df.tail(100), use_container_width=True)

    # State storage for trained model and results
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = None

    # Handle training trigger
    if controls["train_button"]:
        with st.spinner("Preparing sequences and training LSTM Neural Network..."):
            set_random_seed(controls["seed"])

            data_dict = prepare_data(
                df,
                feature_cols=FEATURE_COLS,
                target_col=TARGET_COL,
                train_frac=controls["train_frac"],
                time_steps=controls["time_steps"]
            )

            x_train, y_train = data_dict["x_train"], data_dict["y_train"]
            x_val, y_val = data_dict["x_val"], data_dict["y_val"]
            x_test, y_test = data_dict["x_test"], data_dict["y_test"]
            scaler = data_dict["scaler"]
            target_idx = data_dict["target_index"]

            # Build Model
            model = create_model(
                time_steps=controls["time_steps"],
                num_features=x_train.shape[2],
                lr=controls["learning_rate"],
                lstm_units=controls["lstm_units"],
                dropout_rate=controls["dropout_rate"]
            )

            # Custom progress callback
            progress_bar = st.progress(0, text="Training LSTM...")

            class StreamlitCallback:
                def __init__(self, total_epochs):
                    self.total_epochs = total_epochs

            # Train
            start_train = time.time()
            history = train_model(
                model, x_train, y_train, x_val, y_val,
                epochs=controls["epochs"],
                batch_size=controls["batch_size"],
                verbose=0
            )
            train_duration = time.time() - start_train
            progress_bar.progress(100, text=f"Training finished in {train_duration:.1f} seconds!")

            # Predict & Evaluate
            y_pred_price, y_true_price = predict_prices(
                model, x_test, y_test, scaler, target_idx, batch_size=controls["batch_size"]
            )
            metrics = calculate_metrics(y_true_price, y_pred_price)

            # Save in session state
            st.session_state.pipeline_results = {
                "model": model,
                "history": history.history,
                "metrics": metrics,
                "y_true_price": y_true_price,
                "y_pred_price": y_pred_price,
                "data_dict": data_dict,
                "train_duration": train_duration,
                "controls": controls,
            }
            st.success("✅ Model training and evaluation completed successfully!")

    results = st.session_state.pipeline_results

    # -------------------------------------------------------------
    # TAB 2: Model Training & Architecture
    # -------------------------------------------------------------
    with tab_training:
        if results is None:
            st.info("💡 Click **'🚀 Train & Evaluate Model'** in the sidebar to start training.")
        else:
            st.subheader("Model Training Curves & Loss History")
            history = results["history"]

            # Interactive Plotly Loss Curve
            fig_loss = go.Figure()
            epochs_range = list(range(1, len(history["loss"]) + 1))

            fig_loss.add_trace(
                go.Scatter(
                    x=epochs_range,
                    y=history["loss"],
                    mode="lines+markers",
                    name="Training Loss (MSE)",
                    line=dict(color="#3b82f6", width=2.5)
                )
            )

            if "val_loss" in history:
                fig_loss.add_trace(
                    go.Scatter(
                        x=epochs_range,
                        y=history["val_loss"],
                        mode="lines+markers",
                        name="Validation Loss (MSE)",
                        line=dict(color="#f97316", width=2.5)
                    )
                )

            fig_loss.update_layout(
                title="Mean Squared Error Loss Across Epochs",
                xaxis_title="Epoch",
                yaxis_title="Loss (MSE)",
                template="plotly_dark",
                height=420,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Final Training Loss", f"{history['loss'][-1]:.6f}")
            if "val_loss" in history:
                c2.metric("Best Validation Loss", f"{min(history['val_loss']):.6f}")
            c3.metric("Training Time", f"{results['train_duration']:.1f} sec")

    # -------------------------------------------------------------
    # TAB 3: Predictions & Naive Baseline
    # -------------------------------------------------------------
    with tab_predictions:
        if results is None:
            st.info("💡 Train the model from the sidebar to view evaluation predictions.")
        else:
            metrics = results["metrics"]
            y_true = results["y_true_price"]
            y_pred = results["y_pred_price"]
            data_dict = results["data_dict"]
            dates = data_dict["test_dates"]

            st.subheader("LSTM Model vs. Naive Random-Walk Baseline")

            # Comparative Metric Cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "LSTM RMSE",
                f"${metrics['lstm_rmse']:.4f}",
                delta=f"-{(metrics['naive_rmse'] - metrics['lstm_rmse']):.4f}" if metrics['beats_baseline'] else f"+{(metrics['lstm_rmse'] - metrics['naive_rmse']):.4f}",
                delta_color="inverse"
            )
            col2.metric("Naive Baseline RMSE", f"${metrics['naive_rmse']:.4f}")
            col3.metric("LSTM MAE", f"${metrics['lstm_mae']:.4f}")
            col4.metric("Directional Accuracy", f"{metrics['directional_accuracy_pct']:.2f}%")

            # Honest Educational Verdict Banner
            if metrics["beats_baseline"]:
                st.markdown(
                    f"""
                    <div class="verdict-box-good">
                        <h4>🟢 Baseline Comparison: LSTM Outperformed the Naive Baseline</h4>
                        <p>{metrics['verdict']}</p>
                        <small>The LSTM model captured non-trivial sequential patterns beyond simple lag memory.</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="verdict-box-warn">
                        <h4>⚠️ Baseline Comparison: LSTM Did Not Beat Naive Persistence</h4>
                        <p>{metrics['verdict']}</p>
                        <small>Stock price series resemble near random walks. The visual alignment in the chart is largely driven by a 1-day lag effect.</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Interactive Plotly Comparison Chart
            x_test_axis = dates if dates is not None and len(dates) == len(y_true) else np.arange(len(y_true))

            fig_pred = go.Figure()
            fig_pred.add_trace(
                go.Scatter(
                    x=x_test_axis,
                    y=y_true,
                    mode="lines",
                    name="Actual Real Price",
                    line=dict(color="#10b981", width=2.2)
                )
            )
            fig_pred.add_trace(
                go.Scatter(
                    x=x_test_axis,
                    y=y_pred,
                    mode="lines",
                    name="LSTM Prediction",
                    line=dict(color="#ef4444", width=2, dash="dot")
                )
            )

            fig_pred.update_layout(
                title="Test Set: Actual vs. LSTM Predicted Stock Prices",
                xaxis_title="Date / Trading Days",
                yaxis_title="Stock Price ($)",
                template="plotly_dark",
                height=500,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # Error Residuals Plot
            residuals = y_true - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(
                go.Scatter(
                    x=x_test_axis,
                    y=residuals,
                    mode="lines",
                    name="Residual Error ($)",
                    line=dict(color="#a855f7", width=1.5)
                )
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_res.update_layout(
                title="Prediction Residuals (Actual Price - Predicted Price)",
                xaxis_title="Date / Trading Days",
                yaxis_title="Residual ($)",
                template="plotly_dark",
                height=280,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_res, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: Future Price Forecasting
    # -------------------------------------------------------------
    with tab_forecast:
        if results is None:
            st.info("💡 Please train a model first before generating future forecasts.")
        else:
            st.subheader("🔮 Multi-Step Forward Forecasting")
            st.markdown(
                "Autoregressively project closing prices into the future by rolling the LSTM sequence forward."
            )

            forecast_days = st.slider("Number of Future Days to Forecast", 1, 30, 7)

            if st.button("Generate Future Forecast", type="secondary"):
                model = results["model"]
                data_dict = results["data_dict"]
                scaler = data_dict["scaler"]
                target_idx = data_dict["target_index"]
                time_steps = data_dict["time_steps"]

                # Extract last known window from full test data
                df_test_full = data_dict["df_test_full"]
                last_features = scaler.transform(df_test_full[data_dict["feature_cols"]].values[-time_steps:])

                future_preds = forecast_future(
                    model=model,
                    last_window_scaled=last_features,
                    scaler=scaler,
                    target_index=target_idx,
                    num_days=forecast_days
                )

                # Generate future dates
                if "Date" in df.columns:
                    last_date = df["Date"].iloc[-1]
                    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_days + 1)]
                else:
                    future_dates = [f"Day +{i}" for i in range(1, forecast_days + 1)]

                forecast_df = pd.DataFrame({
                    "Future Step": [f"Day +{i}" for i in range(1, forecast_days + 1)],
                    "Date": future_dates,
                    "Projected Price ($)": [f"${p:.2f}" for p in future_preds],
                    "Numeric Price": future_preds
                })

                st.dataframe(forecast_df[["Future Step", "Date", "Projected Price ($)"]], use_container_width=True)

                # Forecast Chart
                recent_history = df.tail(60)
                recent_x = recent_history["Date"] if "Date" in recent_history.columns else recent_history.index

                fig_future = go.Figure()
                fig_future.add_trace(
                    go.Scatter(
                        x=recent_x,
                        y=recent_history["Close"],
                        mode="lines",
                        name="Recent Actual History",
                        line=dict(color="#3b82f6", width=2)
                    )
                )
                fig_future.add_trace(
                    go.Scatter(
                        x=future_dates,
                        y=future_preds,
                        mode="lines+markers",
                        name="Future Projection",
                        line=dict(color="#f59e0b", width=2.5, dash="dash")
                    )
                )
                fig_future.update_layout(
                    title=f"Next {forecast_days}-Day Price Trajectory Forecast",
                    xaxis_title="Timeline",
                    yaxis_title="Stock Price ($)",
                    template="plotly_dark",
                    height=450,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_future, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 5: Documentation & Methodology
    # -------------------------------------------------------------
    with tab_docs:
        st.subheader("Methodology & Design Architecture")
        st.markdown("""
        ### 1. Data Pipeline & Leakage Prevention
        - **Chronological Split**: Financial time series data must never be randomly shuffled. Shuffling introduces lookahead bias and overstates predictive capability.
        - **Scaler Isolation**: `MinMaxScaler` is fitted strictly on the training partition and then applied to validation and test sets.

        ### 2. LSTM Neural Network Architecture
        - **Lookback Window**: Transforms 2D data into 3D tensors `(samples, time_steps, features)`.
        - **Stacked LSTM Layers**: Two sequential LSTM layers with intermediate Dropout (0.4) to mitigate overfitting.
        - **Dense Projection**: Hidden ReLU dense layer followed by a Sigmoid output layer mapped to normalized price target.

        ### 3. The Naive Baseline Criterion
        In financial machine learning, predicting stock prices is challenging because asset prices closely follow **martingale / random-walk processes**:
        $$y_{naive}(t) = y(t-1)$$
        If an LSTM model achieves an impressive chart simply because it lags 1 day behind, it does not beat this naive persistence baseline. This application calculates and displays this benchmark transparently.
        """)


if __name__ == "__main__":
    main()
