import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# Clear Streamlit cache to ensure fresh data on every run
st.cache_data.clear()

def recalculate_equity_curve(df, horizon_days=5):
    """
    Recalculate equity curve per ticker from signals.
    Each ticker gets its own equity curve based on its own trades.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])

    result_parts = []
    for ticker, g in df.groupby('ticker'):
        g = g.sort_values('date').copy()
        dates = g['date'].tolist()
        # Non-overlapping windows per ticker
        trading_dates = [dates[i] for i in range(0, len(dates), horizon_days)]
        trading_mask = g['date'].isin(trading_dates)
        g['strategy_return'] = 0.0
        g.loc[trading_mask, 'strategy_return'] = (
            g.loc[trading_mask, 'signal'] * g.loc[trading_mask, 'future_return']
        )
        # Compute equity only on trading dates, then forward-fill
        equity_vals = (1 + g.loc[trading_mask, 'strategy_return']).cumprod()
        g['equity_curve'] = np.nan
        g.loc[equity_vals.index, 'equity_curve'] = equity_vals.values
        g['equity_curve'] = g['equity_curve'].ffill().fillna(1.0)
        result_parts.append(g)

    return pd.concat(result_parts, ignore_index=True)

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_ARTIFACTS_PATH = os.path.join(DATA_DIR, "model_artifacts.joblib")
TEST_SIGNALS_PATH = os.path.join(DATA_DIR, "test_signals.csv")
NEWS_PATH = os.path.join(DATA_DIR, "news_headlines.csv")
PRICES_PATH = os.path.join(DATA_DIR, "prices_daily.csv")

# Ticker information for high-beta stocks
TICKER_INFO = {
    "SMCI": {
        "name": "Super Micro Computer",
        "sector": "Semiconductors/AI Infrastructure",
        "description": "AI server & datacenter solutions. High volatility due to rapid growth & tech sector exposure.",
        "characteristics": "Typical 5-day range: 8-15% | High beta: 1.8+"
    },
    "CRSP": {
        "name": "CRISPR Therapeutics",
        "sector": "Biotechnology",
        "description": "Gene editing pioneer. Clinical trial results drive extreme volatility.",
        "characteristics": "Typical 5-day range: 10-20% | News-driven | Beta: 2.0+"
    },
    "PLTR": {
        "name": "Palantir Technologies",
        "sector": "AI/Data Analytics",
        "description": "Government & enterprise AI platform. Meme stock characteristics with institutional backing.",
        "characteristics": "Typical 5-day range: 5-12% | Sentiment-driven | Beta: 1.5+"
    },
    # Legacy support for stable stocks
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "description": "Stable large-cap", "characteristics": "Low volatility"},
    "NVDA": {"name": "NVIDIA", "sector": "Technology", "description": "Stable large-cap", "characteristics": "Moderate volatility"},
    "TSLA": {"name": "Tesla", "sector": "Automotive", "description": "Growth stock", "characteristics": "High volatility"},
    "MSFT": {"name": "Microsoft", "sector": "Technology", "description": "Stable large-cap", "characteristics": "Low volatility"},
    "AMZN": {"name": "Amazon", "sector": "Technology", "description": "Stable large-cap", "characteristics": "Moderate volatility"},
}

# Custom CSS for premium UI
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
    }
    
    .main {
        padding: 1rem 2rem;
    }
    
    .main > div {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    h3 {
        color: rgba(255, 255, 255, 0.8);
        font-weight: 400;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .regime-badge {
        display: inline-block;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 0.5rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-left: 1rem;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        animation: pulse 2s infinite;
    }
    
    .info-banner {
        background: linear-gradient(135deg, rgba(255, 65, 108, 0.15) 0%, rgba(255, 75, 43, 0.15) 100%);
        backdrop-filter: blur(10px);
        border-left: 4px solid #ff416c;
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 65, 108, 0.3);
    }
    
    .info-banner-title {
        color: #ff416c;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .info-banner-text {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    
    .atr-info {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        border-left: 3px solid #667eea;
    }
    
    .atr-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.25rem;
    }
    
    .atr-value {
        color: #00f5a0;
        font-size: 1.1rem;
        font-weight: 700;
    }
    
    .stock-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .stock-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .stock-card:hover::before {
        left: 100%;
    }
    
    .stock-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .ticker-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 16px;
        font-weight: 700;
        font-size: 1.4rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        letter-spacing: 0.05em;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        }
        50% {
            box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
        }
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 100%);
        color: #0a0a0a;
        padding: 1.25rem 2rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.6rem;
        text-align: center;
        box-shadow: 0 12px 40px rgba(0, 245, 160, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        position: relative;
        overflow: hidden;
        animation: glow 2s infinite;
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 12px 40px rgba(0, 245, 160, 0.5);
        }
        50% {
            box-shadow: 0 12px 60px rgba(0, 245, 160, 0.8);
        }
    }
    
    .signal-sell {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white;
        padding: 1.25rem 2rem;
        border-radius: 20px;
        font-weight: 800;
        font-size: 1.6rem;
        text-align: center;
        box-shadow: 0 12px 40px rgba(255, 65, 108, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        position: relative;
        overflow: hidden;
        animation: glowRed 2s infinite;
    }
    
    @keyframes glowRed {
        0%, 100% {
            box-shadow: 0 12px 40px rgba(255, 65, 108, 0.5);
        }
        50% {
            box-shadow: 0 12px 60px rgba(255, 65, 108, 0.8);
        }
    }
    
    .signal-hold {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.05) 100%);
        color: rgba(255, 255, 255, 0.9);
        padding: 1.25rem 2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.6rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .confidence-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 1.5rem;
        border-radius: 20px;
        margin-top: 1.5rem;
        box-shadow: inset 0 2px 10px rgba(102, 126, 234, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.4);
        border-color: rgba(102, 126, 234, 0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.75rem 0;
        text-shadow: 0 0 30px rgba(0, 245, 160, 0.5);
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .news-item {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.03) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        border-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%) 1;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #667eea;
    }
    
    .news-item:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        border-left-color: #00f5a0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.06) 100%);
    }
    
    .news-title {
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 500;
        line-height: 1.6;
    }
    
    .news-date {
        font-size: 0.8rem;
        color: rgba(255, 255, 255, 0.5);
        margin-top: 0.5rem;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.03);
        padding: 1rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 700;
        color: rgba(255, 255, 255, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
        color: white;
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.4);
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.5), transparent);
        margin: 2rem 0;
    }
    
    h4 {
        color: rgba(255, 255, 255, 0.9);
        font-weight: 700;
        font-size: 1.3rem;
        margin: 1.5rem 0 1rem 0;
        letter-spacing: 0.02em;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
"""


@st.cache_data
def load_data():
    """Load all necessary data"""
    signals = pd.DataFrame()
    news = pd.DataFrame()
    prices = pd.DataFrame()
    
    if os.path.exists(TEST_SIGNALS_PATH):
        signals = pd.read_csv(TEST_SIGNALS_PATH, parse_dates=["date"])
        
        # Recalculate equity curve to fix pipeline artifacts (spikes/vertical bars)
        try:
            # Need to drop existing equity_curve if it exists to avoid confusion
            if 'equity_curve' in signals.columns:
                signals = signals.drop(columns=['equity_curve'])
            signals = recalculate_equity_curve(signals, horizon_days=5)
        except Exception as e:
            st.error(f"Error recalculating equity curve: {e}")
    
    if os.path.exists(NEWS_PATH):
        news = pd.read_csv(NEWS_PATH, parse_dates=["date"])
    
    if os.path.exists(PRICES_PATH):
        prices = pd.read_csv(PRICES_PATH, parse_dates=["date"])
    
    return signals, news, prices


def create_price_chart(signals_df, ticker):
    """Create modern price evolution chart"""
    df_ticker = signals_df[signals_df["ticker"] == ticker].copy()
    
    # Get price data from rolling dataset
    if "adj_close_mean" in df_ticker.columns:
        price_col = "adj_close_mean"
    else:
        price_col = "future_return"  # fallback
    
    fig = go.Figure()
    
    # Add price line with gradient
    fig.add_trace(go.Scatter(
        x=df_ticker["date"],
        y=df_ticker[price_col] if price_col in df_ticker.columns else df_ticker.index,
        mode='lines',
        name='Price',
        line=dict(color='#00f5a0', width=3, shape='spline'),
        fill='tozeroy',
        fillcolor='rgba(0, 245, 160, 0.15)'
    ))
    
    # Mark buy signals with star markers
    buy_signals = df_ticker[df_ticker["signal"] == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals["date"],
            y=buy_signals[price_col] if price_col in buy_signals.columns else buy_signals.index,
            mode='markers',
            name='Buy Signal',
            marker=dict(
                color='#00d9f5',
                size=16,
                symbol='star',
                line=dict(color='#00f5a0', width=2)
            )
        ))
    
    # Mark sell signals with inverted triangle markers
    sell_signals = df_ticker[df_ticker["signal"] == -1]
    if not sell_signals.empty:
        fig.add_trace(go.Scatter(
            x=sell_signals["date"],
            y=sell_signals[price_col] if price_col in sell_signals.columns else sell_signals.index,
            mode='markers',
            name='Sell Signal',
            marker=dict(
                color='#ff416c',
                size=16,
                symbol='triangle-down',
                line=dict(color='#ff4b2b', width=2)
            )
        ))
    
    fig.update_layout(
        title=dict(
            text=f"<b>{ticker} Performance Evolution</b>",
            font=dict(size=20, color='rgba(255, 255, 255, 0.9)', family="Poppins")
        ),
        xaxis_title=dict(text="Date", font=dict(color='rgba(255, 255, 255, 0.7)')),
        yaxis_title=dict(text="Value", font=dict(color='rgba(255, 255, 255, 0.7)')),
        hovermode='x unified',
        plot_bgcolor='rgba(15, 12, 41, 0.5)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family="Poppins", size=12, color='rgba(255, 255, 255, 0.8)'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.05)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)', zeroline=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)', zeroline=False)
    
    return fig


def display_stock_card(ticker, latest_data, news_df):
    """Display a premium stock card with signal and news"""
    signal_value = latest_data["signal"]
    if signal_value == 1:
        signal = "BUY"
        signal_class = "signal-buy"
    elif signal_value == -1:
        signal = "SELL"
        signal_class = "signal-sell"
    else:
        # User clarification: Signal 0 means "Cash/Exit".
        # If you hold the stock, this is a SELL signal.
        # If you are in cash, this is a HOLD signal.
        signal = "EXIT / CASH"
        signal_class = "signal-hold"
    
    confidence = latest_data["proba"] * 100
    
    # Get ATR info if available
    atr_info = ""
    if "atr_14_last" in latest_data.index and pd.notna(latest_data["atr_14_last"]):
        atr = latest_data["atr_14_last"]
        atr_pct = (atr / latest_data.get("adj_close_last", 1)) * 100 if "adj_close_last" in latest_data.index else 0
        if atr > 0:
            stop_loss = latest_data.get("adj_close_last", 0) - (2.5 * atr)
            atr_info = f"""
            <div class="atr-info">
                <div class="atr-label">ATR-Based Risk (2.5× multiplier)</div>
                <div class="atr-value">ATR: ${atr:.2f} ({atr_pct:.1f}%) | Stop: ${stop_loss:.2f}</div>
            </div>
            """
    
    # Card HTML with premium styling
    st.markdown(f"""
    <div class="stock-card">
        <div class="ticker-badge">{ticker}</div>
        <div class="{signal_class}">{signal}</div>
        <div class="confidence-box">
            <div style="font-size: 0.95rem; color: rgba(255, 255, 255, 0.7); margin-bottom: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;">Signal Confidence</div>
            <div style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">{confidence:.1f}%</div>
            <div style="height: 8px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; overflow: hidden; margin-top: 1rem;">
                <div style="height: 100%; width: {confidence}%; background: linear-gradient(90deg, #00f5a0 0%, #00d9f5 100%); border-radius: 10px; box-shadow: 0 0 10px rgba(0, 245, 160, 0.5);"></div>
            </div>
        </div>
        {atr_info}
    </div>
    """, unsafe_allow_html=True)
    
    # Display recent news for this ticker
    ticker_news = news_df[news_df["ticker"] == ticker].sort_values("date", ascending=False).head(5)
    
    if not ticker_news.empty:
        st.markdown("#### Recent Market News")
        for _, news_item in ticker_news.iterrows():
            news_date = news_item["date"].strftime("%b %d, %Y")
            st.markdown(f"""
            <div class="news-item">
                <div class="news-title">{news_item["headline"]}</div>
                <div class="news-date">{news_date}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color: rgba(255, 255, 255, 0.5); text-align: center; padding: 2rem; font-style: italic;">
            No recent news available for this ticker
        </div>
        """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="AI Stock Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Load data
    signals, news, prices = load_data()
    
    if signals.empty:
        st.error("⚠️ No data found. Please run `python src/pipeline_train.py` first.")
        return
    
    # Header with premium styling
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <h1>
            AI Stock Intelligence
            <span class="regime-badge">⚡ High-Volatility</span>
        </h1>
        <h3>High-beta growth stocks with ATR-based risk management & sentiment analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Regime info banner
    st.markdown("""
    <div class="info-banner">
        <div class="info-banner-title">🎯 High-Volatility Regime Active</div>
        <div class="info-banner-text">
            Trading high-beta growth stocks (SMCI, CRSP, PLTR) with 5-15% weekly swings.
            Success threshold: <strong>3.5%</strong> (vs 1.5% for stable stocks) | 
            ATR stop losses: <strong>2.5×</strong> multiplier | 
            Sentiment decay: <strong>12-hour</strong> half-life
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Calculate overall metrics
    total_signals = len(signals)
    buy_signals = (signals["signal"] == 1).sum()
    sell_signals = (signals["signal"] == -1).sum()
    hold_signals = (signals["signal"] == 0).sum()
    avg_confidence = signals["proba"].mean() * 100
    hit_rate = ((signals["signal"] == signals["target"]).sum() / len(signals)) * 100 if "target" in signals.columns else 0
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">{total_signals}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Buy Signals</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #00f5a0 0%, #00d9f5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{buy_signals}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sell Signals</div>
            <div class="metric-value" style="background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{sell_signals}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-value">{avg_confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Hit Rate</div>
            <div class="metric-value" style="color: #11998e;">{hit_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stock tabs
    tickers = sorted(signals["ticker"].unique())
    tabs = st.tabs([f"{ticker}" for ticker in tickers])
    
    for idx, ticker in enumerate(tickers):
        with tabs[idx]:
            # Ticker info banner
            if ticker in TICKER_INFO:
                info = TICKER_INFO[ticker]
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                            padding: 1rem 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; 
                            border-left: 4px solid #667eea;">
                    <div style="color: #667eea; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                        {info['name']} <span style="color: rgba(255,255,255,0.5); font-weight: 400; font-size: 0.85rem;">({info['sector']})</span>
                    </div>
                    <div style="color: rgba(255, 255, 255, 0.8); font-size: 0.9rem; margin-bottom: 0.5rem;">
                        {info['description']}
                    </div>
                    <div style="color: rgba(255, 255, 255, 0.6); font-size: 0.8rem; font-style: italic;">
                        {info['characteristics']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Get latest data for this ticker
            ticker_data = signals[signals["ticker"] == ticker].sort_values("date", ascending=False)
            latest = ticker_data.iloc[0]
            
            # Two columns: chart and info
            col_chart, col_info = st.columns([2, 1])
            
            with col_chart:
                # Price evolution chart
                fig = create_price_chart(signals, ticker)
                st.plotly_chart(fig, width='stretch', key=f'price_chart_{ticker}')
                
                # Equity curve - aggregate by date (one value per date)
                equity_by_date = ticker_data.groupby("date")["equity_curve"].first().reset_index()
                
                # NOTE: Equity updates every 5 days (non-overlapping periods), creates step pattern
                # Use 'hv' (horizontal-vertical) line shape to show holding periods clearly
                equity_fig = go.Figure()
                equity_fig.add_trace(go.Scatter(
                    x=equity_by_date["date"],
                    y=equity_by_date["equity_curve"],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#00d9f5', width=3, shape='hv'),  # 'hv' creates step pattern
                    fillcolor='rgba(0, 217, 245, 0.15)',
                    name='Portfolio Value'
                ))
                
                equity_fig.update_layout(
                    title=dict(
                        text="<b>Portfolio Performance (Following Signals)</b>",
                        font=dict(size=20, color='rgba(255, 255, 255, 0.9)', family="Poppins")
                    ),
                    xaxis_title=dict(text="Date", font=dict(color='rgba(255, 255, 255, 0.7)')),
                    yaxis_title=dict(text="Equity Multiplier", font=dict(color='rgba(255, 255, 255, 0.7)')),
                    hovermode='x unified',
                    plot_bgcolor='rgba(15, 12, 41, 0.5)',
                    paper_bgcolor='rgba(0, 0, 0, 0)',
                    font=dict(family="Poppins", size=12, color='rgba(255, 255, 255, 0.8)'),
                    margin=dict(l=20, r=20, t=60, b=20),
                    height=350
                )
                
                equity_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)', zeroline=False)
                equity_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)', zeroline=False)
                
                st.plotly_chart(equity_fig, width='stretch', key=f'equity_chart_{ticker}')
                
                # Performance stats for this ticker
                if not ticker_data.empty:
                    final_equity = ticker_data["equity_curve"].iloc[-1]
                    max_equity = ticker_data["equity_curve"].max()
                    min_equity = ticker_data["equity_curve"].min()
                    
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    with perf_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Final Return</div>
                            <div class="metric-value">{(final_equity - 1) * 100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with perf_col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Max Gain</div>
                            <div class="metric-value">{(max_equity - 1) * 100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with perf_col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Max Drawdown</div>
                            <div class="metric-value" style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{(min_equity - 1) * 100:.1f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            with col_info:
                # Signal card and news
                display_stock_card(ticker, latest, news)


if __name__ == "__main__":
    main()
