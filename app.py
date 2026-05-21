import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dicts import sectors

st.set_page_config(page_title="Hanse Alpha", layout="wide")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ── CACHED DATA FETCHERS ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def get_info(ticker):
    return yf.Ticker(ticker).info

@st.cache_data(ttl=300)
def get_history(ticker, period):
    return yf.Ticker(ticker).history(period=period)

@st.cache_data(ttl=600)
def get_news(ticker):
    return yf.Ticker(ticker).news

@st.cache_data(ttl=3600)
def load_macro():
    return pd.read_csv(
        "BOGMBASE.csv", parse_dates=["observation_date"], index_col="observation_date"
    )

# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────
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

def fmt(val, spec=".2f"):
    try:
        return f"{val:{spec}}" if val is not None else "N/A"
    except (TypeError, ValueError):
        return "N/A"

def pct(val):
    try:
        return f"{val * 100:.1f}%" if val is not None else "N/A"
    except (TypeError, ValueError):
        return "N/A"

def large(val):
    try:
        if val >= 1e12:
            return f"${val / 1e12:.2f}T"
        if val >= 1e9:
            return f"${val / 1e9:.1f}B"
        if val >= 1e6:
            return f"${val / 1e6:.1f}M"
        return f"${val:,.0f}"
    except (TypeError, ValueError):
        return "N/A"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("Hanse Alpha")
mode = st.sidebar.radio(
    "Navigation", ["Stocks", "Watchlist", "Sectors"], label_visibility="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# STOCKS
# ══════════════════════════════════════════════════════════════════════════════
if mode == "Stocks":
    ticker = st.sidebar.text_input(
        "Ticker Symbol", placeholder="e.g. AAPL"
    ).strip().upper()

    if not ticker:
        st.title("Hanse Alpha")
        st.markdown(
            "Enter a ticker symbol in the sidebar to view price charts, "
            "fundamentals, news, and macro data."
        )
    else:
        with st.spinner(f"Loading {ticker}..."):
            info = get_info(ticker)

        name     = info.get("longName") or ticker
        price    = info.get("currentPrice") or info.get("regularMarketPrice")
        prev     = info.get("previousClose")
        change   = (price - prev) / prev * 100 if price and prev else None
        mkt_cap  = info.get("marketCap")
        w52h     = info.get("fiftyTwoWeekHigh")
        w52l     = info.get("fiftyTwoWeekLow")
        pe       = info.get("trailingPE")
        exchange = info.get("exchange", "")
        sector_s = info.get("sector", "")
        industry = info.get("industry", "")

        # ── Company header ────────────────────────────────────────────────────
        st.title(name)
        meta_parts = [p for p in [exchange, sector_s, industry] if p]
        if meta_parts:
            st.caption("  •  ".join(meta_parts))

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric(
            "Price",
            f"${price:,.2f}" if price else "N/A",
            f"{change:+.2f}%" if change is not None else None,
        )
        c2.metric("Market Cap", large(mkt_cap) if mkt_cap else "N/A")
        c3.metric("P/E Ratio", fmt(pe, ".1f") if pe else "N/A")
        c4.metric("52W High", f"${w52h:.2f}" if w52h else "N/A")
        c5.metric("52W Low",  f"${w52l:.2f}" if w52l else "N/A")

        # ── Watchlist button ──────────────────────────────────────────────────
        st.markdown("")
        if ticker in st.session_state.watchlist:
            if st.button(f"Remove {ticker} from Watchlist"):
                st.session_state.watchlist.remove(ticker)
                st.rerun()
        else:
            if st.button(f"+ Add {ticker} to Watchlist"):
                st.session_state.watchlist.append(ticker)
                st.rerun()

        st.divider()

        # ── Tabs ──────────────────────────────────────────────────────────────
        tab_chart, tab_fund, tab_news, tab_macro, tab_cal, tab_targets, tab_income = st.tabs([
            "Chart", "Fundamentals", "News", "Macro Overlay",
            "Calendar", "Price Targets", "Quarterly Income",
        ])

        # ── CHART ─────────────────────────────────────────────────────────────
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
            col_r1, col_r2 = st.columns(2)
            show_rsi  = col_r1.checkbox("Show RSI (14)")
            show_macd = col_r2.checkbox("Show MACD (12/26/9)")

            with st.spinner("Loading chart..."):
                data = get_history(ticker, period)

            if data.empty:
                st.warning("No price data returned.")
            else:
                close   = data["Close"]
                n_sub   = 1 + int(show_rsi) + int(show_macd)
                heights = [0.6] + [0.2] * (n_sub - 1)
                titles  = [name] + (["RSI"] if show_rsi else []) + (["MACD"] if show_macd else [])

                fig = make_subplots(
                    rows=n_sub, cols=1, shared_xaxes=True,
                    row_heights=heights, subplot_titles=titles,
                    vertical_spacing=0.04,
                )

                fig.add_trace(go.Candlestick(
                    x=data.index, open=data["Open"], high=data["High"],
                    low=data["Low"], close=close, name="Price",
                    increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                ), row=1, col=1)

                ind_colors = {
                    "SMA 20": "orange", "SMA 50": "royalblue",
                    "SMA 200": "mediumpurple", "EMA 20": "cyan", "EMA 50": "magenta",
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
                    height=300 + 180 * n_sub,
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01),
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_dark",
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── FUNDAMENTALS ──────────────────────────────────────────────────────
        with tab_fund:
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("**Valuation**")
                st.dataframe(
                    pd.DataFrame.from_dict({
                        "P/E (Trailing)":  fmt(info.get("trailingPE")),
                        "P/E (Forward)":   fmt(info.get("forwardPE")),
                        "P/B Ratio":       fmt(info.get("priceToBook")),
                        "EV/EBITDA":       fmt(info.get("enterpriseToEbitda")),
                        "EV/Revenue":      fmt(info.get("enterpriseToRevenue")),
                        "Price/Sales":     fmt(info.get("priceToSalesTrailing12Months")),
                    }, orient="index", columns=["Value"]),
                    use_container_width=True,
                )
                st.markdown("**Growth & Dividends**")
                st.dataframe(
                    pd.DataFrame.from_dict({
                        "Revenue Growth (YoY)": pct(info.get("revenueGrowth")),
                        "Earnings Growth":      pct(info.get("earningsGrowth")),
                        "Dividend Yield":       pct(info.get("dividendYield")),
                        "Payout Ratio":         pct(info.get("payoutRatio")),
                    }, orient="index", columns=["Value"]),
                    use_container_width=True,
                )

            with col_b:
                st.markdown("**Profitability**")
                st.dataframe(
                    pd.DataFrame.from_dict({
                        "Profit Margin":    pct(info.get("profitMargins")),
                        "Operating Margin": pct(info.get("operatingMargins")),
                        "Gross Margin":     pct(info.get("grossMargins")),
                        "ROE":              pct(info.get("returnOnEquity")),
                        "ROA":              pct(info.get("returnOnAssets")),
                    }, orient="index", columns=["Value"]),
                    use_container_width=True,
                )
                st.markdown("**Financial Health**")
                st.dataframe(
                    pd.DataFrame.from_dict({
                        "Revenue (TTM)": large(info.get("totalRevenue")),
                        "Total Debt":    large(info.get("totalDebt")),
                        "Debt/Equity":   fmt(info.get("debtToEquity")),
                        "Current Ratio": fmt(info.get("currentRatio")),
                        "Quick Ratio":   fmt(info.get("quickRatio")),
                        "EPS (TTM)":     f"${info.get('trailingEps'):.2f}" if info.get("trailingEps") else "N/A",
                    }, orient="index", columns=["Value"]),
                    use_container_width=True,
                )

        # ── NEWS ──────────────────────────────────────────────────────────────
        with tab_news:
            with st.spinner("Loading news..."):
                try:
                    news_items = get_news(ticker)
                    if news_items:
                        for item in news_items[:12]:
                            title     = item.get("title", "No title")
                            link      = item.get("link") or item.get("url", "#")
                            publisher = item.get("publisher", "")
                            ts        = item.get("providerPublishTime")
                            date_str  = datetime.fromtimestamp(ts).strftime("%b %d, %Y") if ts else ""

                            with st.container(border=True):
                                st.markdown(f"**[{title}]({link})**")
                                st.caption(f"{publisher}  •  {date_str}")
                    else:
                        st.info("No news available for this ticker.")
                except Exception as e:
                    st.error(f"Could not load news: {e}")

        # ── MACRO OVERLAY ─────────────────────────────────────────────────────
        with tab_macro:
            period_m = st.selectbox(
                "Timeframe", ["1y", "2y", "5y", "10y", "max"], index=2, key="macro_period"
            )
            with st.spinner("Loading..."):
                price_data = get_history(ticker, period_m)
                macro      = load_macro()

            if price_data.empty:
                st.warning("No price data available.")
            else:
                macro  = macro[macro.index >= price_data.index.min().tz_localize(None)]
                fig_m  = make_subplots(specs=[[{"secondary_y": True}]])
                fig_m.add_trace(
                    go.Scatter(x=price_data.index, y=price_data["Close"],
                               name=ticker, line=dict(color="steelblue", width=1.5)),
                    secondary_y=False,
                )
                fig_m.add_trace(
                    go.Scatter(x=macro.index, y=macro["BOGMBASE"],
                               name="Fed Monetary Base",
                               line=dict(color="orange", dash="dot", width=1.5)),
                    secondary_y=True,
                )
                fig_m.update_layout(
                    height=500, xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", y=1.02),
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_dark",
                )
                fig_m.update_yaxes(title_text=f"{ticker} Price (USD)", secondary_y=False)
                fig_m.update_yaxes(title_text="Monetary Base (Billions USD)", secondary_y=True)
                st.plotly_chart(fig_m, use_container_width=True)
                st.caption("Source: Federal Reserve Bank of St. Louis (FRED) — BOGMBASE")

        # ── CALENDAR ──────────────────────────────────────────────────────────
        with tab_cal:
            try:
                cal = yf.Ticker(ticker).calendar
                if cal is not None and not pd.DataFrame([cal]).empty:
                    st.dataframe(pd.DataFrame([cal]), use_container_width=True)
                else:
                    st.info("No calendar data available.")
            except Exception as e:
                st.error(f"Could not load calendar: {e}")

        # ── PRICE TARGETS ─────────────────────────────────────────────────────
        with tab_targets:
            try:
                targets  = yf.Ticker(ticker).analyst_price_targets
                df_t     = pd.DataFrame(targets) if not isinstance(targets, pd.DataFrame) else targets
                if df_t is not None and not df_t.empty:
                    st.dataframe(df_t, use_container_width=True)
                else:
                    st.info("No analyst price targets available.")
            except Exception as e:
                st.error(f"Could not load price targets: {e}")

        # ── QUARTERLY INCOME ──────────────────────────────────────────────────
        with tab_income:
            with st.spinner("Loading..."):
                try:
                    income = yf.Ticker(ticker).quarterly_income_stmt
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

# ══════════════════════════════════════════════════════════════════════════════
# WATCHLIST
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Watchlist":
    st.title("Watchlist")

    col_in, col_btn = st.columns([4, 1])
    new_t = col_in.text_input(
        "Add ticker", placeholder="e.g. MSFT", label_visibility="collapsed"
    ).strip().upper()
    if col_btn.button("Add", use_container_width=True) and new_t:
        if new_t not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_t)
            st.rerun()
        else:
            st.warning(f"{new_t} is already in your watchlist.")

    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add tickers above or from any stock page.")
    else:
        with st.spinner("Fetching data..."):
            rows = []
            for t in st.session_state.watchlist:
                try:
                    inf   = get_info(t)
                    price = inf.get("currentPrice") or inf.get("regularMarketPrice")
                    prev  = inf.get("previousClose")
                    chg   = (price - prev) / prev * 100 if price and prev else None
                    rows.append({
                        "Ticker":      t,
                        "Company":     inf.get("shortName", t),
                        "Price":       f"${price:,.2f}" if price else "N/A",
                        "Change (1D)": f"{chg:+.2f}%" if chg is not None else "N/A",
                        "Market Cap":  large(inf.get("marketCap")),
                        "P/E":         fmt(inf.get("trailingPE"), ".1f") if inf.get("trailingPE") else "N/A",
                        "52W High":    f"${inf.get('fiftyTwoWeekHigh'):.2f}" if inf.get("fiftyTwoWeekHigh") else "N/A",
                        "52W Low":     f"${inf.get('fiftyTwoWeekLow'):.2f}" if inf.get("fiftyTwoWeekLow") else "N/A",
                    })
                except Exception:
                    rows.append({"Ticker": t, "Company": "Error loading data",
                                 "Price": "N/A", "Change (1D)": "N/A",
                                 "Market Cap": "N/A", "P/E": "N/A",
                                 "52W High": "N/A", "52W Low": "N/A"})

        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

        st.markdown("---")
        to_remove = st.multiselect("Remove from watchlist", st.session_state.watchlist)
        if st.button("Remove Selected") and to_remove:
            for t in to_remove:
                st.session_state.watchlist.remove(t)
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SECTORS
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Sectors":
    sector_name = st.sidebar.selectbox("Sector", list(sectors.keys()))
    action      = st.sidebar.radio(
        "Action",
        ["key", "name", "symbol", "ticker", "overview", "top_companies", "research_reports"],
    )

    st.title(sector_name.replace("-", " ").title())
    with st.spinner("Loading..."):
        try:
            sec    = yf.Sector(sector_name)
            result = getattr(sec, action)
            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
            elif isinstance(result, dict):
                st.dataframe(pd.DataFrame(result), use_container_width=True)
            else:
                st.write(result)
        except Exception as e:
            st.error(f"Could not load sector data: {e}")
