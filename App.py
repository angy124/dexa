# app.py
import os
import joblib
import time
import requests
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objs as go

from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------
# Config
# ---------------------------
st.set_page_config(page_title="Pro AI Trading System", layout="wide")
st.title("🚀 Pro AI Trading System — Signals · SMC · Fibonacci · Backtest · Broker Hooks")

MODEL_PATH = "ai_trading_model.pkl"

# ---------------------------
# Utilities & Data
# ---------------------------
@st.cache_data(ttl=30)
def fetch_data(symbol: str, period: str = "3mo", interval: str = "1h") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, threads=False)
        if df.empty:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['EMA_10'] = EMAIndicator(df['Close'], window=10).ema_indicator()
    df['EMA_30'] = EMAIndicator(df['Close'], window=30).ema_indicator()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()
    bb = BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    # returns & lags
    df['ret_1'] = df['Close'].pct_change(1)
    df['ret_3'] = df['Close'].pct_change(3)
    for lag in (1,2,3):
        df[f'close_lag_{lag}'] = df['Close'].shift(lag)
    df.dropna(inplace=True)
    return df

# ---------------------------
# Fibonacci levels
# ---------------------------
def fibonacci_levels(df: pd.DataFrame):
    """Return common Fibonacci horizontal levels calculated from the visible window (full df)."""
    high = df['High'].max()
    low = df['Low'].min()
    diff = high - low
    return {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '1.0': low
    }

# ---------------------------
# Support & Resistance (swing highs/lows)
# ---------------------------
def support_resistance(df: pd.DataFrame, window=5):
    levels = []
    for i in range(window, len(df)-window):
        is_support = df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window].min()
        is_resist = df['High'].iloc[i] == df['High'].iloc[i-window:i+window].max()
        if is_support:
            levels.append((df.index[i], df['Low'].iloc[i], 'support'))
        if is_resist:
            levels.append((df.index[i], df['High'].iloc[i], 'resistance'))
    return levels

# ---------------------------
# Smart Money Concepts (very simplified)
#   - Detect Break of Structure (BOS) and Change of Character (ChoCH)
# ---------------------------
def detect_smc(df: pd.DataFrame):
    signals = []
    # We'll use higher-high/higher-low for BOS up, lower-low/lower-high for BOS down
    for i in range(2, len(df)):
        prev = i-1
        if df['High'].iloc[i] > df['High'].iloc[prev] and df['Low'].iloc[i] > df['Low'].iloc[prev]:
            signals.append((df.index[i], df['Close'].iloc[i], 'BOS_UP'))
        if df['High'].iloc[i] < df['High'].iloc[prev] and df['Low'].iloc[i] < df['Low'].iloc[prev]:
            signals.append((df.index[i], df['Close'].iloc[i], 'BOS_DOWN'))
        # ChoCH: quick reversal candle detection (simple)
        if (df['Close'].iloc[i] > df['Open'].iloc[i] and df['Close'].iloc[prev] < df['Open'].iloc[prev]) or \
           (df['Close'].iloc[i] < df['Open'].iloc[i] and df['Close'].iloc[prev] > df['Open'].iloc[prev]):
            signals.append((df.index[i], df['Close'].iloc[i], 'CHOCH'))
    return signals

# ---------------------------
# ML Model helpers
# ---------------------------
def prepare_features_and_target(df: pd.DataFrame, horizon:int=1):
    df = df.copy()
    df['future_close'] = df['Close'].shift(-horizon)
    df.dropna(inplace=True)
    df['target'] = (df['future_close'] > df['Close']).astype(int)  # 1 = up next period
    feature_cols = ['EMA_10','EMA_30','MACD','MACD_signal','RSI_14','ret_1','ret_3','close_lag_1','close_lag_2']
    X = df[feature_cols].copy()
    y = df['target'].copy()
    return X, y, df

def train_model(X, y, save_path=MODEL_PATH):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    st.write(f"Model test accuracy: {acc:.4f}")
    st.text(classification_report(y_test, preds))
    joblib.dump(clf, save_path)
    return clf

def load_model(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            st.warning(f"Failed to load model: {e}")
            return None
    return None

def predict(model, X):
    if model is None or X.empty:
        return np.array([])
    return model.predict(X)

# ---------------------------
# Backtesting (simple)
# ---------------------------
def backtest_from_signals(df: pd.DataFrame, signals: np.ndarray, fee=0.0005):
    data = df.copy()
    # align to last len(signals)
    if len(signals) != len(data):
        # if signals shorter, align them to tail
        if len(signals) < len(data):
            data = data.iloc[-len(signals):].copy()
        else:
            raise ValueError("Signals longer than data")
    data['signal'] = signals
    # execute next bar open (approx by using close returns here)
    data['position'] = data['signal'].shift(1).fillna(0)
    data['market_ret'] = data['Close'].pct_change().fillna(0)
    data['strategy_ret'] = data['position'] * data['market_ret']
    trades = data['position'].diff().abs()
    data['strategy_ret'] = data['strategy_ret'] - trades * fee
    data['equity'] = (1 + data['strategy_ret']).cumprod()
    total_return = data['equity'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(data)) - 1 if len(data) > 0 else 0
    vol = data['strategy_ret'].std() * np.sqrt(252) if len(data) > 1 else 0
    sharpe = (annual_return / vol) if vol != 0 else np.nan
    data['cum_max'] = data['equity'].cummax()
    data['drawdown'] = data['equity'] / data['cum_max'] - 1
    max_dd = data['drawdown'].min()
    # win rate (very approximate)
    trade_points = data[data['position'].diff().abs()==1]
    wins = 0
    losses = 0
    # compute naive per-trade returns
    if not trade_points.empty:
        # simple: count positive next-period returns at entries
        for idx in trade_points.index:
            pos = data.index.get_loc(idx)
            # trade return from entry (next bar) to next exit (not perfect but ok)
            if pos+1 < len(data):
                r = data['strategy_ret'].iloc[pos+1]
                if r > 0: wins += 1
                else: losses += 1
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else np.nan
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate
    }
    return data, metrics

# ---------------------------
# Broker placeholders (simulation only)
# ---------------------------
def place_order_simulated(symbol, qty, side):
    return {"status": "simulated", "symbol": symbol, "qty": qty, "side": side, "time": datetime.utcnow().isoformat()}

def place_order_zerodha(api_key, access_token, symbol, qty, txn_type="BUY"):
    st.warning("Zerodha hook is placeholder — implement auth & secure key storage.")
    # Example pseudocode (DO NOT use as-is):
    # url = "https://api.kite.trade/orders/regular"
    # headers = {"Authorization": f"token {api_key}:{access_token}"}
    # payload = {...}
    # r = requests.post(url, headers=headers, data=payload)
    return {"status":"simulated_zerodha", "symbol":symbol, "qty":qty, "txn_type":txn_type}

def place_order_alpaca(key, secret, base_url, symbol, qty, side="buy"):
    st.warning("Alpaca hook is placeholder — implement auth & secure key storage.")
    # Placeholder pseudocode
    return {"status":"simulated_alpaca", "symbol":symbol, "qty":qty, "side":side}

# ---------------------------
# Plot helpers
# ---------------------------
def plot_candles_with_overlays(df: pd.DataFrame, signals=None, fibo=None, sr_levels=None, smc=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'))
    # overlays
    if 'EMA_10' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_10'], name='EMA10', line=dict(width=1)))
    if 'EMA_30' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA_30'], name='EMA30', line=dict(width=1)))
    # Fibonacci
    if fibo:
        for k, v in fibo.items():
            fig.add_hline(y=v, line_dash='dot', annotation_text=f"F {k}", annotation_position="top left")
    # support/resistance lines
    if sr_levels:
        for ts, lvl, typ in sr_levels:
            color = 'green' if typ=='support' else 'red'
            fig.add_hline(y=lvl, line_dash='dash', line_color=color, annotation_text=typ, annotation_position="bottom left")
    # smc markers
    if smc:
        for ts, price, tag in smc:
            fig.add_trace(go.Scatter(x=[ts], y=[price], mode='markers+text', text=[tag], textposition='top center',
                                     marker=dict(size=10, color='purple', symbol='star'), name=tag))
    # signals
    if signals is not None and len(signals)==len(df):
        buys = df[signals==1]
        sells = df[signals==0]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green'), name='AI Buy'))
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='AI Sell'))
    fig.update_layout(height=650, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Streamlit UI
# ---------------------------
# Sidebar controls
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Symbol (yfinance)", value="AAPL")
    period = st.selectbox("History Period", ["1mo","3mo","6mo","1y","2y"], index=1)
    interval = st.selectbox("Interval", ["15m","30m","1h","1d"], index=2)
    st.markdown("---")
    retrain = st.button("Train / Re-train Model")
    run_backtest_btn = st.button("Run Backtest")
    enable_live_sim = st.checkbox("Enable Live Simulation (alerts)", value=False)
    st.markdown("---")
    st.markdown("Broker (simulated placeholders)")
    broker_choice = st.selectbox("Broker", ["Simulated","Zerodha (placeholder)","Alpaca (placeholder)"])
    if broker_choice == "Zerodha (placeholder)":
        zerodha_api_key = st.text_input("Zerodha API Key", type="password")
        zerodha_access_token = st.text_input("Zerodha Access Token", type="password")
    elif broker_choice == "Alpaca (placeholder)":
        alpaca_key = st.text_input("Alpaca Key", type="password")
        alpaca_secret = st.text_input("Alpaca Secret", type="password")
        alpaca_base = st.text_input("Alpaca Base URL", value="https://paper-api.alpaca.markets")
    st.markdown("---")
    st.write("Model file:", MODEL_PATH)
    if st.button("Delete saved model"):
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            st.success("Model removed")
        else:
            st.info("No model file found")

# Fetch data
df_raw = fetch_data(symbol, period=period, interval=interval)
if df_raw.empty:
    st.error("No data available for this symbol/period/interval. Try another selection.")
    st.stop()

df = add_indicators(df_raw)

# Prepare features/target
X, y, df_labeled = prepare_features_and_target(df)
model = load_model()

# Train if requested
if retrain:
    with st.spinner("Training model..."):
        model = train_model(X, y)
        st.success("Model trained and saved.")

# Predict signals
signals = np.zeros(len(df_labeled), dtype=int)
if model is not None and not X.empty:
    preds = predict(model, X)
    # align preds to df_labeled index
    signals[-len(preds):] = preds
else:
    signals[:] = 0  # default hold

# Compute Fibonacci, SR, SMC
fibo = fibonacci_levels(df_labeled)
sr_levels = support_resistance(df_labeled, window=6)
smc = detect_smc(df_labeled)

# Layout: main + right column
col1, col2 = st.columns([3,1])

with col1:
    st.subheader(f"{symbol} — Chart with Indicators & Signals")
    plot_candles_with_overlays(df_labeled, signals=signals, fibo=fibo, sr_levels=sr_levels, smc=smc)

with col2:
    st.subheader("AI Signal & Latest Info")
    latest_price = df_labeled['Close'].iloc[-1]
    latest_time = df_labeled.index[-1]
    latest_signal = int(signals[-1]) if len(signals)>0 else 0
    st.metric("Latest Price", f"{latest_price:.4f}", delta=None)
    st.write(f"Timestamp: {latest_time}")
    if latest_signal == 1:
        st.success("AI Signal: BUY")
    else:
        st.info("AI Signal: HOLD / NO BUY")

    st.markdown("---")
    st.subheader("Latest Features")
    display_cols = ['Close','EMA_10','EMA_30','MACD','MACD_signal','RSI_14']
    st.dataframe(df_labeled.tail(6)[display_cols])

    st.markdown("---")
    st.subheader("Manual Trade (Simulated)")
    qty = st.number_input("Qty (simulated)", min_value=1, value=1, step=1)
    side = st.selectbox("Side", ["buy","sell"])
    if st.button("Place Manual Trade"):
        if broker_choice == "Simulated":
            resp = place_order_simulated(symbol, qty, side)
            st.write("Simulated order:", resp)
        elif broker_choice == "Zerodha (placeholder)":
            resp = place_order_zerodha(zerodha_api_key, zerodha_access_token, symbol, qty, txn_type=side.upper())
            st.write("Zerodha placeholder response:", resp)
        else:
            resp = place_order_alpaca(alpaca_key, alpaca_secret, alpaca_base, symbol, qty, side=side)
            st.write("Alpaca placeholder response:", resp)

# Alerts / live simulation
if enable_live_sim:
    if latest_signal == 1:
        st.sidebar.success(f"LIVE ALERT: BUY {symbol} @ {latest_price:.4f}")
    else:
        st.sidebar.info("LIVE: No buy signal right now")

# Backtest panel
st.markdown("---")
st.header("Backtest & Performance (simple)")

if run_backtest_btn:
    with st.spinner("Running backtest..."):
        bt_df, metrics = backtest_from_signals(df_labeled, signals, fee=0.0005)
        st.subheader("Key Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Return", f"{metrics['total_return']*100:.2f}%")
        c2.metric("Annualized Return", f"{metrics['annual_return']*100:.2f}%")
        c3.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        c4, c5 = st.columns(2)
        c4.metric("Volatility (ann)", f"{metrics['volatility']*100:.2f}%")
        c5.metric("Sharpe", f"{metrics['sharpe']:.2f}")
        st.metric("Win Rate", f"{(metrics['win_rate']*100) if not np.isnan(metrics['win_rate']) else 'N/A'}%")
        st.subheader("Equity Curve & Drawdown")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['equity'], name='Strategy Equity'))
        fig.add_trace(go.Scatter(x=bt_df.index, y=bt_df['cum_max'], name='Cum Max', opacity=0.3))
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bt_df.index, y=bt_df['drawdown'], name='Drawdown'))
        st.plotly_chart(fig2, use_container_width=True)
        st.subheader("Backtest Sample (tail)")
        st.dataframe(bt_df[['Close','signal','position','strategy_ret','equity','drawdown']].tail(50))
        csv = bt_df.to_csv().encode('utf-8')
        st.download_button("Download backtest CSV", csv, file_name=f"backtest_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv")

# Model & data management
st.markdown("---")
st.header("Model & Data Management")
colA, colB = st.columns(2)
with colA:
    if st.button("Export latest features CSV"):
        path = f"features_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.csv"
        df_labeled.to_csv(path)
        st.success(f"Saved features -> {path}")
with colB:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            st.download_button("Download model.pkl", f, file_name="ai_trading_model.pkl")

st.markdown("""
### Notes & Next Steps
- This app is an experiment/demo scaffold. For production trading you must:
  - Add secure secret management for API keys (don't store in plaintext).
  - Implement robust order-reconciliation, persistence, logging.
  - Use realistic slippage, order fills, and market-impact modelling in backtests.
  - Consider more advanced models (LSTM/Transformer, RL) with careful evaluation and walk-forward validation.
- Broker functions above are placeholders; consult broker docs (Zerodha Kite, Alpaca, IBKR) and implement properly.
""")