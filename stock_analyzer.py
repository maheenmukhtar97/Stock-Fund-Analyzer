import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import plotly.graph_objects as go

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="AI Stock Analyzer", layout="wide")

st.markdown("""
<style>

/* KEEP default dark theme */
body {
    background-color: #0e1117;
    color: white;
}

/* Glass card (subtle, dark-friendly) */
.glass {
    background: rgba(255, 255, 255, 0.05);  /* very light */
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border-radius: 12px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    margin-bottom: 15px;
}

/* Optional hover effect */
.glass:hover {
    transform: scale(1.01);
    transition: 0.2s ease;
}

/* Signal colors */
.buy {
    color: #00ff88;
    font-size: 28px;
    font-weight: bold;
}
.sell {
    color: #ff4b4b;
    font-size: 28px;
    font-weight: bold;
}
.hold {
    color: #f1c40f;
    font-size: 28px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.title("📈 AI Stock Analyzer (LSTM)")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")

ticker = st.sidebar.selectbox("Select Ticker", [
    "AAPL", "MSFT", "TSLA", "SPY", "QQQ", "DIA", "GOOGL", "AMZN", "NVDA", "GLD"
])

period = st.sidebar.selectbox("Data Period", ["1Y", "2Y", "5Y"])
window_size = st.sidebar.slider("Lookback Window", 10, 60, 20)
future_days = st.sidebar.slider("Forecast Horizon", 1, 10, 5)
threshold = st.sidebar.slider("Buy Threshold %", 0.5, 5.0, 1.0) / 100
lstm_units = st.sidebar.slider("LSTM Units", 32, 128, 64)
dropout_rate = st.sidebar.slider("Dropout", 0.1, 0.5, 0.2)

run = st.sidebar.button("🚀 Run Analysis")

# ---------------- FUNCTIONS ----------------
def get_data(ticker, period):
    end = datetime.today()
    start = end - timedelta(days=365 * int(period[0]))
    
    df = yf.download(ticker, start=start, end=end)

    # 🔥 FIX: flatten multi-level columns if exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = df.index.tz_localize(None)
    return df

def add_features(df):
    df['body'] = df['Close'] - df['Open']
    df['range'] = df['High'] - df['Low']
    df['upper_shadow'] = df['High'] - np.maximum(df['Close'], df['Open'])
    df['lower_shadow'] = np.minimum(df['Close'], df['Open']) - df['Low']
    df['direction'] = np.where(df['Close'] > df['Open'], 1, -1)

    df['return_1'] = df['Close'].pct_change()
    df['return_5'] = df['Close'].pct_change(5)
    df['return_10'] = df['Close'].pct_change(10)

    df['sma20'] = df['Close'].rolling(20).mean()
    df['sma50'] = df['Close'].rolling(50).mean()
    df['price_vs_sma20'] = df['Close'] / df['sma20']
    df['price_vs_sma50'] = df['Close'] / df['sma50']
    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26

    df['volatility'] = df['return_1'].rolling(20).std()

    df['volume_rel'] = df['Volume'] / df['Volume'].rolling(20).mean()
    

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 1 - (1 / (1 + rs))

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].bfill().ffill()
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    return df

def create_labels(df, future_days, threshold):
    
    df = df.copy()

    # Create future price
    df['future_price'] = df['Close'].shift(-future_days)

    # 🔥 Instead of dropna → keep valid rows only
    mask = df['future_price'].notna()
    df = df[mask]

    # Safety check
    if len(df) == 0:
        return df

    # Calculate returns
    returns = ((df['future_price'] - df['Close']) / df['Close']).squeeze()

    # Create target
    df['target'] = (returns > threshold).astype(int)
    if df.empty:
        st.error("❌ Dataset became empty after label creation. Try 2Y or 5Y period.")
        st.stop()

    return df


def create_sequences(data, target, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(target[i+window])
    return np.array(X), np.array(y)

# ---------------- MAIN ----------------
if run:
    df = get_data(ticker, period)
    df = add_features(df)
    df = create_labels(df, future_days, threshold)

    features = df.drop(columns=[col for col in ['target', 'future_price'] if col in df.columns])
    target = df['target']
    if len(features) < 100:
        st.error("❌ Not enough data after preprocessing. Try selecting longer time period (2Y or 5Y).")
        st.stop()

    split1 = int(len(features)*0.8)
    split2 = int(len(features)*0.9)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("### 📊 Data Split")

    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Train", split1)
    col2.metric("🟡 Validation", split2 - split1)
    col3.metric("🔴 Test", len(features) - split2)

    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)

    col1.metric("📊 Dataset Size", len(df))
    col2.metric("🧠 Features", len(features.columns))

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(features[:split1])
    val_scaled = scaler.transform(features[split1:split2])
    test_scaled = scaler.transform(features[split2:])

    X_train, y_train = create_sequences(train_scaled, target[:split1].values, window_size)
    X_val, y_val = create_sequences(val_scaled, target[split1:split2].values, window_size)
    X_test, y_test = create_sequences(test_scaled, target[split2:].values, window_size)

    # MODEL
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(window_size, X_train.shape[2])),
        Dropout(dropout_rate),
        LSTM(lstm_units),
        Dropout(dropout_rate),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    with st.expander("🧠 Model Architecture"):
        model.summary(print_fn=lambda x: st.text(x))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    es = EarlyStopping(patience=10, restore_best_weights=True)
    rl = ReduceLROnPlateau(patience=5)


    with st.spinner("🧠 Training LSTM Model..."):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=32,
            callbacks=[es, rl],
            verbose=0
        )
        # EVALUATION
    preds = model.predict(X_test)
    # Create signal column
    test_index = df.index[-len(preds):]

    signals = pd.DataFrame({
        "Date": test_index,
        "Price": df['Close'].iloc[-len(preds):],
        "Prob": preds.flatten()
    })

    signals['Signal'] = signals['Prob'].apply(
        lambda x: 1 if x > 0.5 else 0
    )

    buy_signals = signals[signals['Signal'] == 1]
    sell_signals = signals[signals['Signal'] == 0]
    preds_binary = (preds > 0.5).astype(int)


    # PLOTS
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close']
    ))
    # ✅ ADD SMA LINES
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sma20'],
        line=dict(color='blue'),
        name='SMA20'
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df['sma50'],
        line=dict(color='orange'),
        name='SMA50'
    ))

    # ✅ ADD VOLUME (secondary axis)
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))

    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False
        )
    )
    # ✅ ADD SIGNAL MARKERS

    fig.add_trace(go.Scatter(
        x=buy_signals["Date"],
        y=buy_signals["Price"],
        mode='markers',
        marker=dict(color='green', size=8),
        name='BUY'
    ))

    fig.add_trace(go.Scatter(
       x=sell_signals["Date"],
       y=sell_signals["Price"],
       mode='markers',
       marker=dict(color='red', size=8),
       name='SELL'
    ))
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 📈 Price Chart")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    with st.expander("📋 View Feature Table"):
       st.dataframe(df.tail(100))

    # LOSS CURVE
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=history.history['loss'], name='Train Loss'))
    fig2.add_trace(go.Scatter(y=history.history['val_loss'], name='Val Loss'))

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("### 📉 Training Loss")

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    acc = accuracy_score(y_test, preds_binary)
    roc = roc_auc_score(y_test, preds)
    
    # -------- METRICS CARDS --------
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("🎯 Accuracy", f"{acc:.2f}")
    col2.metric("📈 ROC-AUC", f"{roc:.2f}")

    st.markdown('</div>', unsafe_allow_html=True)


    # -------- CLASSIFICATION REPORT (ADD HERE) --------
    from sklearn.metrics import classification_report
    import pandas as pd

    report_dict = classification_report(y_test, preds_binary, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.markdown("### 📋 Classification Report")
    st.dataframe(
        report_df.style
        .format("{:.2f}")
        .background_gradient(cmap="Greens")
    )
    st.info(f"""
    📌 Model Performance Summary:

    - Accuracy shows overall correctness → {acc:.2f}
    - ROC-AUC shows prediction quality → {roc:.2f}

    ⚠️ Note:
    Small dataset may reduce accuracy. Increasing data period (2Y or 5Y) will improve results.
    """)

    st.markdown("### 📊 Prediction Probability")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=test_index,
        y=preds.flatten(),
        name="BUY Probability"
    ))

    st.plotly_chart(fig3, use_container_width=True)

    # FINAL SIGNAL
    latest = scaler.transform(features.tail(window_size))
    latest = np.expand_dims(latest, axis=0)
    prob = model.predict(latest)[0][0]

    if prob >= 0.65:
        signal = "STRONG BUY"
        cls = "buy"
    elif prob >= 0.52:
        signal = "BUY"
        cls = "buy"
    elif prob >= 0.35:
        signal = "HOLD"
        cls = "hold"
    elif prob >= 0.20:
        signal = "SELL"
        cls = "sell"
    else:
        signal = "STRONG SELL"
        cls = "sell"

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("### 🎯 Final Decision")

    st.markdown(f'<p class="{cls}">{signal} ({prob:.2f})</p>', unsafe_allow_html=True)



    # ✅ Dynamic reasoning
    if prob >= 0.65:
        reason = "Strong bullish trend detected with high momentum."
    elif prob >= 0.52:
        reason = "Moderate upward trend with positive indicators."
    elif prob >= 0.35:
        reason = "Mixed signals — no strong trend."
    elif prob >= 0.20:
        reason = "Weak momentum with bearish pressure."
    else:
        reason = "Strong downward trend detected."


    st.markdown(f"""
    📊 Model predicted probability: {prob:.2f}
    📌 Decision: **{signal}**

    🧠 Reasoning:
    {reason}
    """)  
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.warning("This is for educational purposes only. Not financial advice.")
    st.markdown("""
    ### 🧠 Why LSTM?

    LSTM (Long Short-Term Memory) is used because stock prices are sequential data.

    It learns patterns over time such as:
    - Trends
    - Momentum
    - Market cycles

    Unlike simple models, LSTM remembers past information which improves prediction accuracy.
    """)
    st.markdown("### ⚙️ Model Tuning Guide")

    guide = pd.DataFrame({
        "Parameter": ["Lookback Window", "Forecast Days", "LSTM Units", "Dropout"],
        "What it Does": [
            "How much past data model sees",
            "How far ahead prediction is made",
            "Model complexity (higher = more learning)",
            "Prevents overfitting"
        ]
    })

    st.table(guide)
