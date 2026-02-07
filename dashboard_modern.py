import os
from datetime import datetime, timedelta

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SIGNALS_PATH = os.path.join(DATA_DIR, "test_signals.csv")
NEWS_PATH = os.path.join(DATA_DIR, "news_headlines.csv")
PRICES_PATH = os.path.join(DATA_DIR, "prices_daily.csv")

# Custom CSS for modern UI
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    .stApp {
        background: #f8f9fa;
    }
    
    h1 {
        color: #1a1a1a;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stock-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin-bottom: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stock-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
    }
    
    .ticker-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    .signal-buy {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(56, 239, 125, 0.4);
    }
    
    .signal-hold {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
    }
    
    .confidence-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .news-item {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s;
    }
    
    .news-item:hover {
        border-left-color: #764ba2;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .news-title {
        font-size: 0.95rem;
        color: #2c3e50;
        font-weight: 500;
        line-height: 1.5;
    }
    
    .news-date {
        font-size: 0.8rem;
        color: #95a5a6;
        margin-top: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
    }
    
    .price-positive {
        color: #38ef7d;
        font-weight: 700;
    }
    
    .price-negative {
        color: #f5576c;
        font-weight: 700;
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
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=df_ticker["date"],
        y=df_ticker[price_col] if price_col in df_ticker.columns else df_ticker.index,
        mode='lines',
        name='Price',
        line=dict(color='#667eea', width=3),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    # Mark buy signals
    buy_signals = df_ticker[df_ticker["signal"] == 1]
    if not buy_signals.empty:
        fig.add_trace(go.Scatter(
            x=buy_signals["date"],
            y=buy_signals[price_col] if price_col in buy_signals.columns else buy_signals.index,
            mode='markers',
            name='Buy Signal',
            marker=dict(
                color='#38ef7d',
                size=12,
                symbol='triangle-up',
                line=dict(color='white', width=2)
            )
        ))
    
    fig.update_layout(
        title=f"{ticker} Performance",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", size=12),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=400
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
    
    return fig


def display_stock_card(ticker, latest_data, news_df):
    """Display a modern stock card with signal and news"""
    signal = "BUY 🚀" if latest_data["signal"] == 1 else "HOLD 💼"
    confidence = latest_data["proba"] * 100
    
    # Determine signal class
    signal_class = "signal-buy" if latest_data["signal"] == 1 else "signal-hold"
    
    # Card HTML
    st.markdown(f"""
    <div class="stock-card">
        <div class="ticker-badge">{ticker}</div>
        <div class="{signal_class}">{signal}</div>
        <div class="confidence-box">
            <div style="font-size: 0.9rem; color: #6c757d; margin-bottom: 0.5rem;">Signal Confidence</div>
            <div style="font-size: 2rem; font-weight: 700; color: #667eea;">{confidence:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display recent news for this ticker
    ticker_news = news_df[news_df["ticker"] == ticker].sort_values("date", ascending=False).head(5)
    
    if not ticker_news.empty:
        st.markdown("#### 📰 Recent News")
        for _, news_item in ticker_news.iterrows():
            news_date = news_item["date"].strftime("%b %d, %Y")
            st.markdown(f"""
            <div class="news-item">
                <div class="news-title">{news_item["headline"]}</div>
                <div class="news-date">{news_date}</div>
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
    
    # Header
    st.markdown("# 📈 AI Stock Intelligence Dashboard")
    st.markdown("### Real-time trading signals powered by machine learning")
    st.markdown("---")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate overall metrics
    total_signals = len(signals)
    buy_signals = (signals["signal"] == 1).sum()
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
            <div class="metric-value" style="color: #38ef7d;">{buy_signals}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-value">{avg_confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Hit Rate</div>
            <div class="metric-value" style="color: #11998e;">{hit_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Stock tabs
    tickers = sorted(signals["ticker"].unique())
    tabs = st.tabs([f"📊 {ticker}" for ticker in tickers])
    
    for idx, ticker in enumerate(tickers):
        with tabs[idx]:
            # Get latest data for this ticker
            ticker_data = signals[signals["ticker"] == ticker].sort_values("date", ascending=False)
            latest = ticker_data.iloc[0]
            
            # Two columns: chart and info
            col_chart, col_info = st.columns([2, 1])
            
            with col_chart:
                # Price evolution chart
                fig = create_price_chart(signals, ticker)
                st.plotly_chart(fig, use_container_width=True)
                
                # Equity curve
                equity_fig = go.Figure()
                equity_fig.add_trace(go.Scatter(
                    x=ticker_data["date"],
                    y=ticker_data["equity_curve"],
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#11998e', width=3),
                    fillcolor='rgba(17, 153, 142, 0.1)',
                    name='Portfolio Value'
                ))
                
                equity_fig.update_layout(
                    title="Portfolio Performance (Following Signals)",
                    xaxis_title="Date",
                    yaxis_title="Equity Multiplier",
                    hovermode='x unified',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="Inter", size=12),
                    margin=dict(l=20, r=20, t=60, b=20),
                    height=350
                )
                
                equity_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                equity_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
                
                st.plotly_chart(equity_fig, use_container_width=True)
            
            with col_info:
                # Signal card and news
                display_stock_card(ticker, latest, news)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; padding: 2rem;">
        <p>🤖 Powered by XGBoost + FinBERT Sentiment Analysis</p>
        <p style="font-size: 0.85rem;">Data updated from FRED & Finnhub APIs</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
