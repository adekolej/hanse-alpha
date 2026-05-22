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
# fast_info uses a lightweight ticker endpoint — far less likely to be
# rate-limited on cloud IPs than the full quote-summary used by .info.
@st.cache_data(ttl=300)
def get_fast_info(ticker):
    fi = yf.Ticker(ticker).fast_info
    return {
        "last_price":          getattr(fi, "last_price", None),
        "previous_close":      getattr(fi, "previous_close", None),
        "market_cap":          getattr(fi, "market_cap", None),
        "fifty_two_week_high": getattr(fi, "fifty_two_week_high", None),
        "fifty_two_week_low":  getattr(fi, "fifty_two_week_low", None),
        "exchange":            getattr(fi, "exchange", None),
        "currency":            getattr(fi, "currency", "USD"),
    }

# Full .info — only used for Fundamentals; may fail on rate-limited IPs.
@st.cache_data(ttl=300)
def get_info(ticker):
    return yf.Ticker(ticker).info

# yf.download() hits Yahoo's chart API, a separate endpoint to quote-summary.
@st.cache_data(ttl=300)
def get_history(ticker, period):
    data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    # yfinance 1.x returns a (Price, Ticker) MultiIndex even for single tickers.
    # Drop the ticker level so data["Close"] is a plain Series, not a DataFrame.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Ensure index is timezone-naive for consistent comparisons
    if hasattr(data.index, "tz") and data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    return data

@st.cache_data(ttl=600)
def get_news(ticker):
    return yf.Ticker(ticker).news

@st.cache_data(ttl=900)
def get_calendar(ticker):
    return yf.Ticker(ticker).calendar

@st.cache_data(ttl=900)
def get_price_targets(ticker):
    return yf.Ticker(ticker).analyst_price_targets

@st.cache_data(ttl=900)
def get_income_stmt(ticker):
    return yf.Ticker(ticker).quarterly_income_stmt

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
        # ── fast_info: lightweight, reliable on cloud IPs ─────────────────────
        with st.spinner(f"Loading {ticker}..."):
            fi = get_fast_info(ticker)

        price    = fi.get("last_price")
        prev     = fi.get("previous_close")
        change   = (price - prev) / prev * 100 if price and prev else None
        mkt_cap  = fi.get("market_cap")
        w52h     = fi.get("fifty_two_week_high")
        w52l     = fi.get("fifty_two_week_low")
        exchange = fi.get("exchange", "")

        # Try full .info for name/sector/P/E — fall back silently if blocked
        try:
            info     = get_info(ticker)
            name     = info.get("longName") or ticker
            sector_s = info.get("sector", "")
            industry = info.get("industry", "")
            pe       = info.get("trailingPE")
        except Exception:
            info     = {}
            name     = ticker
            sector_s = ""
            industry = ""
            pe       = None

        # ── Company header ────────────────────────────────────────────────────
        hcol, rcol = st.columns([5, 1])
        hcol.title(name)
        if rcol.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

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
            if not info:
                st.warning(
                    "Fundamental data is temporarily unavailable — Yahoo Finance "
                    "rate-limited this request. Click **Refresh Data** to try again."
                )
                st.stop()
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
                            # yfinance >= 1.x nests everything under item['content']
                            c         = item.get("content") or item  # fallback for older shape
                            title     = c.get("title", "No title")
                            link      = (
                                (c.get("canonicalUrl") or c.get("clickThroughUrl") or {}).get("url")
                                or c.get("link") or "#"
                            )
                            publisher = (c.get("provider") or {}).get("displayName") or c.get("publisher", "")
                            pub_date  = c.get("pubDate") or ""
                            date_str  = pub_date[:10] if pub_date else ""   # "2026-05-21T14:07:08Z" → "2026-05-21"

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
                macro = macro[macro.index >= price_data.index.min()]
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
                cal = get_calendar(ticker)
                if cal:
                    # Convert to a vertical Series → clean single-column table
                    df_cal = pd.Series({k: str(v) for k, v in cal.items()}).to_frame("Value")
                    st.dataframe(df_cal, use_container_width=True)
                else:
                    st.info("No calendar data available.")
            except Exception as e:
                st.error(f"Could not load calendar: {e}")

        # ── PRICE TARGETS ─────────────────────────────────────────────────────
        with tab_targets:
            try:
                targets = get_price_targets(ticker)
                if targets:
                    if isinstance(targets, pd.DataFrame):
                        df_t = targets
                    elif isinstance(targets, dict):
                        df_t = pd.Series(targets).to_frame("Value")
                    else:
                        df_t = pd.DataFrame(targets)
                    st.dataframe(df_t, use_container_width=True)
                else:
                    st.info("No analyst price targets available.")
            except Exception as e:
                st.error(f"Could not load price targets: {e}")

        # ── QUARTERLY INCOME ──────────────────────────────────────────────────
        with tab_income:
            with st.spinner("Loading..."):
                try:
                    income = get_income_stmt(ticker)
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
                    fi    = get_fast_info(t)
                    price = fi.get("last_price")
                    prev  = fi.get("previous_close")
                    chg   = (price - prev) / prev * 100 if price and prev else None
                    rows.append({
                        "Ticker":      t,
                        "Price":       f"${price:,.2f}" if price else "N/A",
                        "Change (1D)": f"{chg:+.2f}%" if chg is not None else "N/A",
                        "Market Cap":  large(fi.get("market_cap")),
                        "52W High":    f"${fi.get('fifty_two_week_high'):.2f}" if fi.get("fifty_two_week_high") else "N/A",
                        "52W Low":     f"${fi.get('fifty_two_week_low'):.2f}" if fi.get("fifty_two_week_low") else "N/A",
                    })
                except Exception:
                    rows.append({"Ticker": t, "Price": "N/A", "Change (1D)": "N/A",
                                 "Market Cap": "N/A", "52W High": "N/A", "52W Low": "N/A"})

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
                # top_companies → plain table
                st.dataframe(result, use_container_width=True)

            elif isinstance(result, dict):
                # overview → vertical key/value table (scalar values)
                st.dataframe(pd.Series(result).to_frame("Value"), use_container_width=True)

            elif isinstance(result, list):
                # research_reports → list of dicts → table
                if result and isinstance(result[0], dict):
                    cols = ["reportDate", "headHtml", "provider", "investmentRating",
                            "targetPrice", "targetPriceStatus"]
                    df_rr = pd.DataFrame(result)
                    display_cols = [c for c in cols if c in df_rr.columns]
                    st.dataframe(df_rr[display_cols] if display_cols else df_rr,
                                 use_container_width=True)
                else:
                    st.write(result)

            elif hasattr(result, "ticker"):
                # ticker → yfinance.Ticker object; just show the symbol string
                st.write(result.ticker)

            else:
                # key / name / symbol → plain strings
                st.write(result)

        except Exception as e:
            st.error(f"Could not load sector data: {e}")
