import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dicts import sectors

st.set_page_config(page_title="Hanse Alpha", layout="wide")
st.title("HANSE ALPHA")

# ── INDICATOR HELPERS ────────────────────────────────────────────────────────

def calc_sma(s, w):
    return s.rolling(w).mean()

def calc_ema(s, span):
    return s.ewm(span=span, adjust=False).mean()

def calc_bollinger(s, w=20, k=2):
    mid = calc_sma(s, w)
    std = s.rolling(w).std()
    return mid, mid + k * std, mid - k * std

def calc_rsi(s, p=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(com=p - 1, min_periods=p).mean()
    loss = (-d.clip(upper=0)).ewm(com=p - 1, min_periods=p).mean()
    return 100 - 100 / (1 + gain / loss)

def calc_macd(s, fast=12, slow=26, sig=9):
    macd = calc_ema(s, fast) - calc_ema(s, slow)
    signal = calc_ema(macd, sig)
    return macd, signal, macd - signal

# ── LAYOUT ───────────────────────────────────────────────────────────────────

mode = st.sidebar.radio("Mode", ["Stocks", "Sectors"])

# ── STOCKS ───────────────────────────────────────────────────────────────────
if mode == "Stocks":
    ticker = st.sidebar.text_input("Ticker Symbol").strip().upper()

    if ticker:
        dat = yf.Ticker(ticker)
        st.header(ticker)
        tab_chart, tab_macro, tab_cal, tab_targets, tab_income = st.tabs(
            ["Chart", "Macro Overlay", "Calendar", "Price Targets", "Quarterly Income"]
        )

        with tab_chart:
            period = st.selectbox(
                "Timeframe",
                ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
                index=3,
            )
            indicators = st.multiselect(
                "Overlay Indicators",
                ["SMA 20", "SMA 50", "SMA 200", "EMA 20", "EMA 50", "Bollinger Bands"],
                default=["SMA 50"],
            )
            col1, col2 = st.columns(2)
            show_rsi = col1.checkbox("Show RSI (14)")
            show_macd = col2.checkbox("Show MACD (12/26/9)")

            with st.spinner("Loading..."):
                data = dat.history(period=period)

            if data.empty:
                st.warning("No price data returned.")
            else:
                close = data["Close"]

                n_subplots = 1 + int(show_rsi) + int(show_macd)
                row_heights = [0.6] + [0.2] * (n_subplots - 1)
                subplot_titles = [ticker] + (["RSI"] if show_rsi else []) + (["MACD"] if show_macd else [])

                fig = make_subplots(
                    rows=n_subplots, cols=1, shared_xaxes=True,
                    row_heights=row_heights, subplot_titles=subplot_titles,
                    vertical_spacing=0.04,
                )

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=data.index, open=data["Open"], high=data["High"],
                    low=data["Low"], close=close, name="Price",
                    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                ), row=1, col=1)

                # Overlay indicators
                ind_colors = {
                    "SMA 20": "orange", "SMA 50": "royalblue", "SMA 200": "mediumpurple",
                    "EMA 20": "cyan", "EMA 50": "magenta",
                }
                for ind in indicators:
                    if ind.startswith("SMA"):
                        w = int(ind.split()[1])
                        fig.add_trace(go.Scatter(
                            x=data.index, y=calc_sma(close, w),
                            name=ind, line=dict(color=ind_colors[ind], width=1.2),
                        ), row=1, col=1)
                    elif ind.startswith("EMA"):
                        span = int(ind.split()[1])
                        fig.add_trace(go.Scatter(
                            x=data.index, y=calc_ema(close, span),
                            name=ind, line=dict(color=ind_colors[ind], width=1.2),
                        ), row=1, col=1)
                    elif ind == "Bollinger Bands":
                        mid, upper, lower = calc_bollinger(close)
                        fig.add_trace(go.Scatter(
                            x=data.index, y=upper, name="BB Upper",
                            line=dict(color="gray", dash="dash", width=1),
                        ), row=1, col=1)
                        fig.add_trace(go.Scatter(
                            x=data.index, y=lower, name="BB Lower",
                            line=dict(color="gray", dash="dash", width=1),
                            fill="tonexty", fillcolor="rgba(128,128,128,0.08)",
                        ), row=1, col=1)
                        fig.add_trace(go.Scatter(
                            x=data.index, y=mid, name="BB Mid",
                            line=dict(color="gray", width=1),
                        ), row=1, col=1)

                current_row = 2
                if show_rsi:
                    rsi = calc_rsi(close)
                    fig.add_trace(go.Scatter(
                        x=data.index, y=rsi, name="RSI",
                        line=dict(color="orange", width=1.2),
                    ), row=current_row, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red",
                                  annotation_text="70", row=current_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green",
                                  annotation_text="30", row=current_row, col=1)
                    current_row += 1

                if show_macd:
                    macd, signal, hist = calc_macd(close)
                    bar_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hist]
                    fig.add_trace(go.Bar(
                        x=data.index, y=hist, name="Histogram",
                        marker_color=bar_colors, opacity=0.6,
                    ), row=current_row, col=1)
                    fig.add_trace(go.Scatter(
                        x=data.index, y=macd, name="MACD",
                        line=dict(color="royalblue", width=1.2),
                    ), row=current_row, col=1)
                    fig.add_trace(go.Scatter(
                        x=data.index, y=signal, name="Signal",
                        line=dict(color="orange", width=1.2),
                    ), row=current_row, col=1)

                fig.update_layout(
                    height=300 + 180 * n_subplots,
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab_macro:
            period_m = st.selectbox(
                "Timeframe",
                ["1y", "2y", "5y", "10y", "max"],
                index=2,
                key="macro_period",
            )
            with st.spinner("Loading..."):
                price_data = dat.history(period=period_m)
                macro = pd.read_csv(
                    "BOGMBASE.csv",
                    parse_dates=["observation_date"],
                    index_col="observation_date",
                )

            if price_data.empty:
                st.warning("No price data available.")
            else:
                macro = macro[macro.index >= price_data.index.min()]

                fig_m = make_subplots(specs=[[{"secondary_y": True}]])
                fig_m.add_trace(
                    go.Scatter(
                        x=price_data.index, y=price_data["Close"],
                        name=ticker, line=dict(color="steelblue", width=1.5),
                    ),
                    secondary_y=False,
                )
                fig_m.add_trace(
                    go.Scatter(
                        x=macro.index, y=macro["BOGMBASE"],
                        name="Fed Monetary Base",
                        line=dict(color="orange", dash="dot", width=1.5),
                    ),
                    secondary_y=True,
                )
                fig_m.update_layout(
                    height=500,
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", y=1.02),
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_dark",
                )
                fig_m.update_yaxes(title_text=f"{ticker} Price (USD)", secondary_y=False)
                fig_m.update_yaxes(title_text="Monetary Base (Billions USD)", secondary_y=True)
                st.plotly_chart(fig_m, use_container_width=True)
                st.caption("Source: Federal Reserve Bank of St. Louis (FRED) — BOGMBASE")

        with tab_cal:
            try:
                cal = dat.calendar
                if cal is not None and not pd.DataFrame([cal]).empty:
                    st.dataframe(pd.DataFrame([cal]), use_container_width=True)
                else:
                    st.info("No calendar data available.")
            except Exception as e:
                st.error(f"Could not load calendar: {e}")

        with tab_targets:
            try:
                targets = dat.analyst_price_targets
                df_targets = pd.DataFrame(targets) if not isinstance(targets, pd.DataFrame) else targets
                if df_targets is not None and not df_targets.empty:
                    st.dataframe(df_targets, use_container_width=True)
                else:
                    st.info("No analyst price targets available.")
            except Exception as e:
                st.error(f"Could not load price targets: {e}")

        with tab_income:
            with st.spinner("Loading..."):
                try:
                    income = dat.quarterly_income_stmt
                    if income is not None and not income.empty:
                        formatted = income.copy()
                        for col in formatted.columns:
                            formatted[col] = formatted[col].apply(
                                lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else x
                            )
                        st.dataframe(formatted, use_container_width=True)
                    else:
                        st.info("No quarterly income statement available.")
                except Exception as e:
                    st.error(f"Could not load income statement: {e}")

# ── SECTORS ───────────────────────────────────────────────────────────────────
elif mode == "Sectors":
    sector_name = st.sidebar.selectbox("Sector", list(sectors.keys()))
    action = st.sidebar.radio(
        "Action",
        ["key", "name", "symbol", "ticker", "overview", "top_companies", "research_reports"],
    )

    st.header(sector_name.replace("-", " ").title())
    with st.spinner("Loading..."):
        try:
            sec = yf.Sector(sector_name)
            result = getattr(sec, action)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
            elif isinstance(result, dict):
                st.dataframe(pd.DataFrame(result), use_container_width=True)
            else:
                st.write(result)
        except Exception as e:
            st.error(f"Could not load sector data: {e}")