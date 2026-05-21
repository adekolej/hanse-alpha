import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from dicts import sectors

st.set_page_config(page_title="Hanse Alpha", layout="wide")
st.title("HANSE ALPHA")

mode = st.sidebar.radio("Mode", ["Stocks", "Sectors"])

# ── STOCKS ──────────────────────────────────────────────────────────────────
if mode == "Stocks":
    ticker = st.sidebar.text_input("Ticker Symbol").strip().upper()

    if ticker:
        dat = yf.Ticker(ticker)
        st.header(ticker)
        tab_hist, tab_cal, tab_targets, tab_income = st.tabs(
            ["Price History", "Calendar", "Price Targets", "Quarterly Income"]
        )

        with tab_hist:
            period = st.selectbox(
                "Timeframe",
                ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
                index=5,
            )
            with st.spinner("Loading..."):
                data = dat.history(period=period)
            if data.empty:
                st.warning("No price data returned for this ticker/timeframe.")
            else:
                st.dataframe(data, use_container_width=True)
                if st.checkbox("Show Close Price Chart"):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(data.index, data["Close"], color="steelblue")
                    ax.set_ylabel("Close Price")
                    ax.grid(True, linestyle="--", alpha=0.5)
                    st.pyplot(fig)

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

# ── SECTORS ─────────────────────────────────────────────────────────────────
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