import io
import time
import random
import urllib.request
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dicts import sectors, gpw_indices, gpw_stocks, supply_chains
import cache as disk_cache

st.set_page_config(page_title="Hanse Alpha", layout="wide")

# ── RETRY HELPER ──────────────────────────────────────────────────────────────
# Keywords that indicate a transient / rate-limit error worth retrying.
_RETRYABLE = (
    "rate limit", "429", "too many requests",
    "connection", "timeout", "ssl", "remote end closed",
    "read timed out", "failed to establish",
)

def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in _RETRYABLE)

def _yf_call(fn, retries: int = 3, base_delay: float = 2.0):
    """Call fn() with exponential back-off on transient / rate-limit errors.

    Usage:
        result = _yf_call(lambda: yf.Ticker(ticker).fast_info)

    If all retries fail with a rate-limit error the last exception is re-raised
    so st.cache_data does NOT cache the failure, allowing the next user
    interaction to try again.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1 and _is_rate_limited(exc):
                jitter = random.uniform(0.0, 0.5)
                time.sleep(base_delay * (2 ** attempt) + jitter)
            else:
                raise          # non-retryable or last attempt → propagate
    raise last_exc             # should be unreachable, but satisfies type-checkers

# ── CACHING ───────────────────────────────────────────────────────────────────
# Two-layer cache for company data:
#   1. st.cache_data → in-memory, per-process, very fast, lost on restart.
#   2. cache.py      → on-disk, survives restarts / Cloud redeploys.
# cached_data() stacks both: a cold process first looks on disk before hitting
# the API, so previously fetched data is reused instead of re-downloaded.
# The ttl applies to BOTH layers and can be overridden globally via the
# HANSE_CACHE_TTL env var (see cache.py).

def cached_data(ttl):
    def decorator(fn):
        namespace = fn.__name__

        @st.cache_data(ttl=ttl)
        def wrapper(*args, **kwargs):
            key = repr((args, tuple(sorted(kwargs.items()))))
            return disk_cache.cached(namespace, key, ttl, lambda: fn(*args, **kwargs))

        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper

    return decorator

def clear_all_caches():
    """Clear both the in-memory and the on-disk cache (used by Refresh buttons)."""
    st.cache_data.clear()
    try:
        disk_cache.clear()
    except Exception:
        pass

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ── CACHED DATA FETCHERS ──────────────────────────────────────────────────────
# Every fetcher wraps its yfinance call in _yf_call() so transient rate-limit
# or connection errors are retried up to 3 times with exponential back-off
# before surfacing to the UI.  TTLs are set aggressively long for data that
# barely changes intraday (calendar, income, price targets) to minimise the
# total number of outbound requests from Streamlit Cloud.

# fast_info → lightweight endpoint, separate from quote-summary.
@cached_data(ttl=300)
def get_fast_info(ticker):
    def _fetch():
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
    return _yf_call(_fetch)

# Full .info — used for Fundamentals; heavier endpoint, more likely throttled.
@cached_data(ttl=600)
def get_info(ticker):
    return _yf_call(lambda: yf.Ticker(ticker).info)

# yf.download() hits Yahoo's chart API — a separate endpoint to quote-summary.
@cached_data(ttl=300)
def get_history(ticker, period):
    def _fetch():
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        # yfinance 1.x returns a (Price, Ticker) MultiIndex even for single tickers.
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        # Ensure tz-naive index for consistent date comparisons.
        if hasattr(data.index, "tz") and data.index.tz is not None:
            data.index = data.index.tz_localize(None)
        return data
    return _yf_call(_fetch)

@cached_data(ttl=600)
def get_news(ticker):
    return _yf_call(lambda: yf.Ticker(ticker).news)

# Calendar, income, price targets change at most once a quarter — cache 1 hour.
@cached_data(ttl=3600)
def get_calendar(ticker):
    return _yf_call(lambda: yf.Ticker(ticker).calendar)

@cached_data(ttl=3600)
def get_price_targets(ticker):
    return _yf_call(lambda: yf.Ticker(ticker).analyst_price_targets)

@cached_data(ttl=3600)
def get_income_stmt(ticker):
    return _yf_call(lambda: yf.Ticker(ticker).quarterly_income_stmt)

@cached_data(ttl=1800)
def get_sector_data(sector_name, action):
    """Fetch a single sector attribute. Returns a (kind, value) tuple so the
    yfinance.Ticker object case can be flattened to its symbol string before
    Streamlit tries to cache it (raw Ticker objects don't serialise)."""
    def _fetch():
        sec    = yf.Sector(sector_name)
        result = getattr(sec, action, None)
        if result is None:
            return ("none", None)
        if isinstance(result, pd.DataFrame):
            return ("dataframe", result)
        if isinstance(result, dict):
            return ("dict", result)
        if isinstance(result, list):
            return ("list", result)
        if hasattr(result, "ticker"):            # yfinance.Ticker object
            return ("ticker", result.ticker)
        return ("scalar", result)
    return _yf_call(_fetch)

@st.cache_data(ttl=3600)
def load_macro():
    return pd.read_csv(
        "BOGMBASE.csv", parse_dates=["observation_date"], index_col="observation_date"
    )

# ── STOOQ.PL (GPW / Warsaw Stock Exchange) ────────────────────────────────────
# stooq.pl exposes two relevant CSV endpoints:
#   • /q/l/  — live multi-symbol quote snapshot (OHLCV + previous close). FREE,
#              no apikey required. This powers the GPW quotes board.
#   • /q/d/l/ — daily historical series. Now gated behind an apikey obtained via
#              captcha at https://stooq.pl/q/d/?s=<sym>&get_apikey . When the user
#              supplies that key, historical candlestick charts are unlocked.
_STOOQ_UA = "Mozilla/5.0 (compatible; HanseAlpha/1.0)"

# Field order requested from /q/l/ → maps 1:1 to the columns we rename below.
_STOOQ_QUOTE_FIELDS = "snd2t2ohlcvp"
_STOOQ_COL_MAP = {
    "Symbol": "Symbol", "Nazwa": "Name", "Data": "Date", "Czas": "Time",
    "Otwarcie": "Open", "Najwyzszy": "High", "Najnizszy": "Low",
    "Zamkniecie": "Close", "Wolumen": "Volume", "Poprzedni": "PrevClose",
}

def _http_get(url: str, timeout: float = 15.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": _STOOQ_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")

def _stooq_apikey_wall(text: str) -> bool:
    """stooq returns a plain-text 'get your apikey' notice instead of CSV when
    the historical endpoint is rate/quota gated."""
    head = text[:80].lower()
    return "apikey" in head and ("uzyskaj" in head or "get your" in head)

@cached_data(ttl=120)
def get_stooq_quotes(symbols: tuple[str, ...]) -> pd.DataFrame:
    """Live snapshot for a set of GPW symbols. Returns a DataFrame indexed by
    Symbol with numeric OHLCV/PrevClose columns plus a computed Change% (close
    vs. previous close). Rows that stooq couldn't price are dropped silently."""
    if not symbols:
        return pd.DataFrame()
    sym_q = "+".join(s.strip().lower() for s in symbols if s.strip())
    url = f"https://stooq.pl/q/l/?s={sym_q}&f={_STOOQ_QUOTE_FIELDS}&h&e=csv"

    def _fetch():
        text = _http_get(url)
        df = pd.read_csv(io.StringIO(text))
        df = df.rename(columns=_STOOQ_COL_MAP)
        for col in ("Open", "High", "Low", "Close", "Volume", "PrevClose"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # stooq writes "N/D" for instruments it has no data for → become NaN.
        df = df.dropna(subset=["Close"])
        if "PrevClose" in df.columns:
            df["Change%"] = (df["Close"] - df["PrevClose"]) / df["PrevClose"] * 100
        return df.set_index("Symbol")

    return _yf_call(_fetch)

@cached_data(ttl=300)
def get_stooq_history(symbol: str, apikey: str) -> pd.DataFrame:
    """Daily OHLCV history for a single GPW symbol via the apikey-gated endpoint.
    Returns a DataFrame indexed by tz-naive date with Open/High/Low/Close/Volume.
    Raises RuntimeError if stooq responds with its apikey wall (bad/expired key
    or exhausted quota)."""
    sym = symbol.strip().lower()
    url = f"https://stooq.pl/q/d/l/?s={sym}&i=d&apikey={apikey.strip()}"

    def _fetch():
        text = _http_get(url)
        if _stooq_apikey_wall(text):
            raise RuntimeError(
                "stooq rejected the request (invalid/expired apikey or quota "
                "exhausted). Get a fresh key at stooq.pl …&get_apikey."
            )
        df = pd.read_csv(io.StringIO(text))
        # stooq.pl historical headers: Data,Otwarcie,Najwyzszy,Najnizszy,Zamkniecie,Wolumen
        df = df.rename(columns=_STOOQ_COL_MAP)
        if "Date" not in df.columns:
            raise RuntimeError(f"Unexpected stooq response: {text[:120]!r}")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    return _yf_call(_fetch)

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

# Sector palette for the Supply-Chain sub-supplier nodes (Tier-2/3/4).
SECTOR_COLORS = {
    "Semiconductors":       "#5b8ff9",
    "Semi Equipment":       "#5ad8a6",
    "EDA Software":         "#5d7092",
    "Electronic Materials": "#f6bd16",
    "Specialty Chemicals":  "#945fb9",
    "Industrial Gases":     "#6dc8ec",
    "Mining & Refining":    "#ff9845",
    "Metals & Alloys":      "#e8684a",
    "Battery Materials":    "#269a99",
    "Optics & Lasers":      "#ff99c3",
    "Composite Materials":  "#9270ca",
    "Aerospace Components": "#a0d911",
}
_SECTOR_FALLBACK = "#888888"

def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("Hanse Alpha")
mode = st.sidebar.radio(
    "Navigation",
    ["Stocks", "Watchlist", "Sectors", "GPW (Stooq)", "Supply Chain"],
    label_visibility="collapsed",
)

# ── CACHE CONTROLS ────────────────────────────────────────────────────────────
with st.sidebar.expander("Local cache", expanded=False):
    _cs = disk_cache.stats()
    st.caption(
        f"{_cs['files']} entr{'y' if _cs['files'] == 1 else 'ies'} · "
        f"{_cs['bytes'] / 1024:.0f} KB on disk"
    )
    st.caption(
        "Company data is cached locally and reused until it expires, so it isn't "
        "re-downloaded on every load or restart."
    )
    if st.button("Clear cached data", use_container_width=True):
        n = disk_cache.clear()
        st.cache_data.clear()
        st.success(f"Cleared {n} cached entr{'y' if n == 1 else 'ies'}.")
        st.rerun()

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
            clear_all_caches()
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
                    "⚠️ Fundamental data is temporarily unavailable — Yahoo Finance "
                    "rate-limited this request (it has already been retried 3×). "
                    "Wait a minute then click **Refresh Data**."
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
            for i, t in enumerate(st.session_state.watchlist):
                # Small stagger between requests so Yahoo doesn't see a burst
                # from a single IP — 0.3 s per ticker, skipped for the first.
                if i > 0:
                    time.sleep(0.3)
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
                except Exception as exc:
                    label = "rate limited — click Refresh" if _is_rate_limited(exc) else "N/A"
                    rows.append({"Ticker": t, "Price": label, "Change (1D)": "N/A",
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

    # Header + refresh button
    hcol, rcol = st.columns([5, 1])
    hcol.title(sector_name.replace("-", " ").title())
    if rcol.button("Refresh Data", key="sector_refresh"):
        clear_all_caches()
        st.rerun()

    with st.spinner("Loading..."):
        try:
            kind, value = get_sector_data(sector_name, action)
        except Exception as e:
            kind, value = "error", str(e)

    if kind == "none" or value is None:
        st.warning(
            f"⚠️ No data returned for **{action}** — Yahoo Finance may be "
            "rate-limiting this request (it has already been retried 3×). "
            "Wait a minute then click **Refresh Data**."
        )

    elif kind == "error":
        st.error(f"Could not load sector data: {value}")

    elif kind == "dataframe":
        st.dataframe(value, use_container_width=True)

    elif kind == "dict":
        # overview etc. — render as vertical key/value table
        st.dataframe(pd.Series(value).to_frame("Value"), use_container_width=True)

    elif kind == "list":
        # research_reports → list of dicts → table
        if value and isinstance(value[0], dict):
            cols = ["reportDate", "headHtml", "provider", "investmentRating",
                    "targetPrice", "targetPriceStatus"]
            df_rr = pd.DataFrame(value)
            display_cols = [c for c in cols if c in df_rr.columns]
            st.dataframe(df_rr[display_cols] if display_cols else df_rr,
                         use_container_width=True)
        else:
            st.write(value)

    elif kind == "ticker":
        # The yfinance.Sector.ticker attribute → underlying index symbol
        st.subheader("Index Symbol")
        st.code(value)

    else:
        # key / name / symbol → plain strings
        st.subheader(action.replace("_", " ").title())
        st.write(value)

# ══════════════════════════════════════════════════════════════════════════════
# GPW (STOOQ) — Warsaw Stock Exchange data via stooq.pl
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "GPW (Stooq)":
    st.title("GPW — Giełda Papierów Wartościowych")
    st.caption("Source: stooq.pl")

    view = st.sidebar.radio("View", ["Notowania", "Wykres"])

    # ── QUOTES BOARD ──────────────────────────────────────────────────────────
    if view == "Notowania":
        group = st.sidebar.radio("Group", ["Indices", "Stocks", "Custom"])

        if group == "Indices":
            symbols = list(gpw_indices.keys())
            labels  = gpw_indices
        elif group == "Stocks":
            symbols = list(gpw_stocks.keys())
            labels  = gpw_stocks
        else:
            raw = st.sidebar.text_area(
                "Symbols (one per line or space-separated)",
                value="pkn pko kgh cdr wig20",
            )
            symbols = [s for s in raw.replace("\n", " ").split() if s]
            labels  = {}

        hcol, rcol = st.columns([5, 1])
        hcol.subheader(group + " — live quotes")
        if rcol.button("Refresh Data", key="gpw_refresh"):
            clear_all_caches()
            st.rerun()

        if not symbols:
            st.info("Enter at least one stooq symbol (e.g. pkn, wig20).")
        else:
            with st.spinner("Loading GPW quotes..."):
                try:
                    df = get_stooq_quotes(tuple(symbols))
                except Exception as e:
                    df = None
                    err = e

            if df is None:
                st.error(f"Could not load quotes from stooq.pl: {err}")
            elif df.empty:
                st.warning("No data returned for the requested symbols.")
            else:
                disp = pd.DataFrame(index=df.index)
                disp["Name"]      = [labels.get(s.lower(), df.at[s, "Name"]
                                     if "Name" in df.columns else s) for s in df.index]
                disp["Date"]      = df.get("Date")
                disp["Open"]      = df["Open"].map(lambda v: fmt(v))
                disp["High"]      = df["High"].map(lambda v: fmt(v))
                disp["Low"]       = df["Low"].map(lambda v: fmt(v))
                disp["Close"]     = df["Close"].map(lambda v: fmt(v))
                if "Change%" in df.columns:
                    disp["Change %"] = df["Change%"].map(
                        lambda v: f"{v:+.2f}%" if pd.notna(v) else "N/A"
                    )
                disp["Volume"]    = df["Volume"].map(
                    lambda v: f"{v:,.0f}" if pd.notna(v) else "N/A"
                )
                st.dataframe(disp, use_container_width=True)
                st.caption(
                    "Change % is the last price vs. previous close. "
                    "Quotes are delayed per stooq.pl terms."
                )

    # ── HISTORICAL CHART ──────────────────────────────────────────────────────
    else:
        # apikey from st.secrets (preferred) or a sidebar field
        try:
            apikey = st.secrets.get("stooq_apikey", "")
        except Exception:
            apikey = ""   # no secrets.toml configured
        if not apikey:
            apikey = st.sidebar.text_input(
                "stooq apikey", type="password",
                help="Historical data needs a stooq apikey. Get one at "
                     "stooq.pl/q/d/?s=pkn&get_apikey",
            ).strip()

        sym_default = list(gpw_stocks.keys())[0]
        symbol = st.sidebar.text_input(
            "Symbol", value=sym_default, help="stooq symbol, e.g. pkn, kgh, wig20"
        ).strip().lower()

        indicators = st.sidebar.multiselect(
            "Overlay Indicators",
            ["SMA 20", "SMA 50", "SMA 200", "EMA 20", "EMA 50", "Bollinger Bands"],
            default=["SMA 50"],
        )
        show_rsi  = st.sidebar.checkbox("Show RSI (14)")
        show_macd = st.sidebar.checkbox("Show MACD (12/26/9)")

        label = gpw_stocks.get(symbol) or gpw_indices.get(symbol) or symbol.upper()
        st.subheader(label)

        if not apikey:
            st.info(
                "Historical data from stooq.pl requires an **apikey** (the bulk "
                "download endpoint is captcha-gated).\n\n"
                "1. Open https://stooq.pl/q/d/?s=pkn&get_apikey\n"
                "2. Enter the captcha and copy the apikey from the download link.\n"
                "3. Paste it in the sidebar (or add `stooq_apikey` to Streamlit secrets).\n\n"
                "The **Notowania** view works without a key."
            )
        elif not symbol:
            st.info("Enter a stooq symbol in the sidebar.")
        else:
            with st.spinner(f"Loading {symbol.upper()} history..."):
                try:
                    data = get_stooq_history(symbol, apikey)
                except Exception as e:
                    data = None
                    err = e

            if data is None:
                st.error(f"Could not load history: {err}")
            elif data.empty or "Close" not in data.columns:
                st.warning("No historical data returned for this symbol.")
            else:
                close   = data["Close"]
                n_sub   = 1 + int(show_rsi) + int(show_macd)
                heights = [0.6] + [0.2] * (n_sub - 1)
                titles  = [label] + (["RSI"] if show_rsi else []) + (["MACD"] if show_macd else [])

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
                st.caption("Source: stooq.pl — daily OHLCV")

# ══════════════════════════════════════════════════════════════════════════════
# SUPPLY CHAIN — curated supplier map + live market enrichment
# ══════════════════════════════════════════════════════════════════════════════
elif mode == "Supply Chain":
    st.title("Supply Chain Explorer")

    company = st.sidebar.selectbox("Company", list(supply_chains.keys()))
    chain      = supply_chains[company]
    suppliers  = chain["suppliers"]
    crit_label = {1: "Secondary", 2: "Important", 3: "Critical"}
    crit_color = {1: "#7e8aa2", 2: "#f0a020", 3: "#ef5350"}

    hcol, rcol = st.columns([5, 1])
    hcol.subheader(f"{company} — supplier map")
    if rcol.button("Refresh Data", key="sc_refresh"):
        clear_all_caches()
        st.rerun()

    show_live = st.sidebar.checkbox("Load live market data", value=True)
    min_crit  = st.sidebar.select_slider(
        "Min. criticality", options=[1, 2, 3], value=1,
        format_func=lambda c: crit_label[c],
    )
    suppliers = [s for s in suppliers if s["criticality"] >= min_crit]

    # Sub-supplier tiers (2→4). Each entry's "supplies" lists names in the tier
    # directly toward the company. Visibility cascades inward from the
    # (criticality-filtered) Tier-1, so hidden parents drop their children too.
    tier_data = {
        "Tier-2": chain.get("tier2", []),
        "Tier-3": chain.get("tier3", []),
        "Tier-4": chain.get("tier4", []),
    }
    has_subtiers = any(tier_data.values())
    include_sub  = st.sidebar.checkbox(
        "Show sub-supplier tiers (2–4)", value=True, disabled=not has_subtiers
    )

    def _filter_tier(entries, allowed):
        out = []
        for e in entries:
            targets = [n for n in e["supplies"] if n in allowed]
            if targets:
                out.append({**e, "targets": targets})
        return out

    tier2 = tier3 = tier4 = []
    if include_sub and has_subtiers:
        tier2 = _filter_tier(tier_data["Tier-2"], {s["name"] for s in suppliers})
        tier3 = _filter_tier(tier_data["Tier-3"], {e["name"] for e in tier2})
        tier4 = _filter_tier(tier_data["Tier-4"], {e["name"] for e in tier3})
    subtiers = [("Tier-2", tier2), ("Tier-3", tier3), ("Tier-4", tier4)]
    all_sub  = tier2 + tier3 + tier4

    st.caption(
        "Relationships are **curated** from public supplier lists and filings "
        "(illustrative — Tier-3/4 especially). Market data is live via Yahoo Finance."
    )

    if not suppliers:
        st.info("No suppliers match the selected criticality filter.")
        st.stop()

    categories = sorted({s["category"] for s in suppliers})

    # ── NODE INFO LOOKUP (used for hover tooltips and Node Inspector) ─────────
    node_info: dict[str, dict] = {
        company: {
            "tier": "Company", "ticker": chain.get("ticker", "—"),
            "country": "—", "role": "Subject company", "extra": "",
        }
    }
    for s in suppliers:
        node_info[s["name"]] = {
            "tier": f"Tier-1 · {crit_label[s['criticality']]}",
            "ticker": s["ticker"] or "private",
            "country": s["country"],
            "role": s["role"],
            "extra": s["category"],
        }
    for tlabel, grp in subtiers:
        for t in grp:
            node_info[t["name"]] = {
                "tier": tlabel,
                "ticker": t["ticker"] or "private",
                "country": t["country"],
                "role": t["role"],
                "extra": t.get("sector", "—"),
            }

    def _node_cd(name: str) -> list:
        info = node_info.get(name, {})
        return [
            info.get("tier", "—"),
            info.get("ticker", "—"),
            info.get("country", "—"),
            info.get("role", "—"),
            info.get("extra", ""),
        ]

    _HOVER_TMPL = (
        "<b>%{label}</b><br>"
        "<i style='color:#aaa'>%{customdata[0]}</i><br>"
        "Ticker: <b>%{customdata[1]}</b>  ·  %{customdata[2]}<br>"
        "Role: %{customdata[3]}<br>"
        "<span style='color:#aaa'>%{customdata[4]}</span>"
        "<extra></extra>"
    )

    if all_sub:
        # ── SANKEY: Tier-4 → Tier-3 → Tier-2 → Tier-1 → Company ───────────────
        # Sub-supplier nodes coloured by SECTOR; Tier-1 by criticality.
        node_labels, node_colors = [], []
        groups    = [("Tier-4", tier4), ("Tier-3", tier3), ("Tier-2", tier2)]
        group_idx = []                       # name→node-index, one dict per group
        for _, grp in groups:
            idx = {}
            for e in grp:
                idx[e["name"]] = len(node_labels)
                node_labels.append(e["name"])
                node_colors.append(SECTOR_COLORS.get(e["sector"], _SECTOR_FALLBACK))
            group_idx.append(idx)
        t1_idx = {}
        for s in suppliers:
            t1_idx[s["name"]] = len(node_labels)
            node_labels.append(s["name"])
            node_colors.append(crit_color[s["criticality"]])
        company_idx = len(node_labels)
        node_labels.append(company)
        node_colors.append("#26a69a")
        node_customdata = [_node_cd(n) for n in node_labels]

        # each group supplies INTO the next group rightward: T4→T3, T3→T2, T2→T1
        right_maps = [group_idx[1], group_idx[2], t1_idx]
        src, tgt, val, link_color = [], [], [], []
        for gi, (_, grp) in enumerate(groups):
            rmap = right_maps[gi]
            for e in grp:
                lc = _hex_rgba(SECTOR_COLORS.get(e["sector"], _SECTOR_FALLBACK), 0.35)
                for name in e["targets"]:
                    if name in rmap:
                        src.append(group_idx[gi][e["name"]])
                        tgt.append(rmap[name]); val.append(1); link_color.append(lc)
        for s in suppliers:
            src.append(t1_idx[s["name"]]); tgt.append(company_idx)
            val.append(s["criticality"]); link_color.append("rgba(38,166,154,0.25)")

        height = 170 + 22 * max(len(tier4), len(tier3), len(tier2), len(suppliers))
        caption = ("Left→right: **Tier-4 → Tier-3 → Tier-2 → Tier-1 → Company**. "
                   "Sub-supplier nodes are coloured by **sector** (legend above); "
                   "Tier-1 by criticality (🔴 critical · 🟠 important · ⚪ secondary).")

        # Sector legend (HTML swatches) rendered above the chart.
        present = sorted({e["sector"] for e in all_sub})
        legend  = "  ".join(
            f'<span style="display:inline-block;width:11px;height:11px;'
            f'background:{SECTOR_COLORS.get(s, _SECTOR_FALLBACK)};border-radius:2px;'
            f'margin:0 4px -1px 0"></span>{s}'
            for s in present
        )
        st.markdown("**Sector legend** &nbsp; " + legend, unsafe_allow_html=True)
    else:
        # ── SANKEY: Tier-1 → Category → Company (sub-tiers hidden) ─────────────
        node_labels, node_colors = [], []
        for s in suppliers:
            node_labels.append(s["name"])
            node_colors.append(crit_color[s["criticality"]])
        cat_base = len(suppliers)
        for c in categories:
            node_labels.append(c)
            node_colors.append("#4a6fa5")
        company_idx = len(node_labels)
        node_labels.append(company)
        node_colors.append("#26a69a")
        # category aggregation nodes get a placeholder tooltip
        for c in categories:
            node_info[c] = {"tier": "Category", "ticker": "—", "country": "—",
                            "role": c, "extra": ""}
        node_customdata = [_node_cd(n) for n in node_labels]

        cat_idx = {c: cat_base + i for i, c in enumerate(categories)}
        src, tgt, val, link_color = [], [], [], []
        cat_totals = {c: 0 for c in categories}
        for i, s in enumerate(suppliers):
            src.append(i); tgt.append(cat_idx[s["category"]]); val.append(s["criticality"])
            link_color.append("rgba(240,160,32,0.25)")
            cat_totals[s["category"]] += s["criticality"]
        for c in categories:
            src.append(cat_idx[c]); tgt.append(company_idx); val.append(cat_totals[c])
            link_color.append("rgba(38,166,154,0.25)")

        height = 130 + 26 * len(suppliers)
        caption = ("Flow width ∝ supplier criticality. "
                   "Node colour: 🔴 critical · 🟠 important · ⚪ secondary.")

    sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=node_labels, color=node_colors,
            customdata=node_customdata,
            hovertemplate=_HOVER_TMPL,
            pad=12, thickness=15,
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        link=dict(source=src, target=tgt, value=val, color=link_color),
    ))
    sankey.update_layout(
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
        template="plotly_dark",
        font=dict(size=11),
    )
    st.plotly_chart(sankey, use_container_width=True)
    st.caption(caption)

    # Live-quote helper (cached + rate-limit aware); sleep is the caller's job.
    def live_cols(ticker):
        try:
            fi    = get_fast_info(ticker)
            price = fi.get("last_price")
            prev  = fi.get("previous_close")
            chg   = (price - prev) / prev * 100 if price and prev else None
            ccy   = fi.get("currency", "")
            return {
                "Price":       f"{price:,.2f} {ccy}".strip() if price else "N/A",
                "Change (1D)": f"{chg:+.2f}%" if chg is not None else "N/A",
                "Market Cap":  large(fi.get("market_cap")) if fi.get("market_cap") else "N/A",
            }
        except Exception as exc:
            label = "rate limited" if _is_rate_limited(exc) else "N/A"
            return {"Price": label, "Change (1D)": "N/A", "Market Cap": "N/A"}

    # ── NODE INSPECTOR ────────────────────────────────────────────────────────
    # Hover over any node for a quick tooltip; use the inspector for full details.
    inspectable = sorted(
        n for n in node_labels
        if n != company and n not in categories
    )
    with st.expander("Node Inspector — click a name to see company details", expanded=False):
        sel_node = st.selectbox(
            "Select node", ["—"] + inspectable, key="sc_node_inspector",
            label_visibility="collapsed",
        )
        if sel_node != "—":
            info = node_info[sel_node]
            st.markdown(f"### {sel_node}")
            d1, d2, d3 = st.columns(3)
            d1.metric("Tier", info["tier"])
            d2.metric("Ticker", info["ticker"])
            d3.metric("Country", info["country"])
            st.markdown(f"**Role:** {info['role']}")
            if info["extra"]:
                label = "Sector" if info["tier"].startswith("Tier-2") or info["tier"].startswith("Tier-3") or info["tier"].startswith("Tier-4") else "Category"
                st.markdown(f"**{label}:** {info['extra']}")
            ticker = info["ticker"]
            if show_live and ticker not in ("—", "private"):
                with st.spinner(f"Fetching live data for {ticker}…"):
                    lq = live_cols(ticker)
                p1, p2, p3 = st.columns(3)
                p1.metric("Price", lq["Price"])
                p2.metric("Change (1D)", lq["Change (1D)"])
                p3.metric("Market Cap", lq["Market Cap"])

    # ── TIER-1 SUPPLIER TABLE (+ optional live quotes) ────────────────────────
    st.markdown("### Tier-1 suppliers")
    rows, _n = [], 0
    for s in suppliers:
        row = {
            "Supplier":    s["name"],
            "Ticker":      s["ticker"] or "—",
            "Category":    s["category"],
            "Country":     s["country"],
            "Role":        s["role"],
            "Criticality": crit_label[s["criticality"]],
        }
        if show_live and s["ticker"]:
            if _n > 0:
                time.sleep(0.2)   # stagger requests (same pattern as Watchlist)
            _n += 1
            row.update(live_cols(s["ticker"]))
        elif show_live:
            row["Price"], row["Change (1D)"], row["Market Cap"] = "private", "—", "—"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).set_index("Supplier"), use_container_width=True)

    # ── SUB-SUPPLIER TABLE (Tiers 2–4, with sector) ───────────────────────────
    if all_sub:
        st.markdown("### Sub-suppliers (Tiers 2–4)")
        sub_rows = []
        for tlabel, grp in subtiers:
            for t in grp:
                row = {
                    "Sub-supplier": t["name"],
                    "Tier":         tlabel,
                    "Sector":       t["sector"],
                    "Ticker":       t["ticker"] or "—",
                    "Country":      t["country"],
                    "Role":         t["role"],
                    "Supplies":     ", ".join(t["targets"]),
                    "# served":     len(t["targets"]),
                }
                if show_live and t["ticker"]:
                    if _n > 0:
                        time.sleep(0.15)
                    _n += 1
                    row.update(live_cols(t["ticker"]))
                elif show_live:
                    row["Price"], row["Change (1D)"], row["Market Cap"] = "private", "—", "—"
                sub_rows.append(row)
        st.dataframe(pd.DataFrame(sub_rows).set_index("Sub-supplier"),
                     use_container_width=True)
        st.caption("**Sector** drives the node colour in the diagram. **# served** ≥ 2 "
                   "marks a convergence point — a sub-supplier several firms in the "
                   "tier above depend on.")

    # ── SUMMARY METRICS ───────────────────────────────────────────────────────
    st.markdown("### Concentration")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Tier-1 suppliers", len(suppliers))
    m2.metric("Sub-suppliers (T2–4)", len(all_sub))
    n_conv = sum(1 for t in all_sub if len(t["targets"]) >= 2)
    m3.metric("Convergence points", n_conv)
    n_sectors = len({t["sector"] for t in all_sub})
    m4.metric("Sectors", n_sectors)
    n_countries = len({s["country"] for s in suppliers} | {t["country"] for t in all_sub})
    m5.metric("Countries", n_countries)

    # Geographic concentration table (all tiers)
    geo = (
        pd.Series([s["country"] for s in suppliers] + [t["country"] for t in all_sub])
        .value_counts()
        .rename_axis("Country")
        .to_frame("Suppliers")
    )
    st.markdown("**Geographic exposure** (all tiers)")
    st.dataframe(geo, use_container_width=True)

    # Sector concentration table (sub-suppliers)
    if all_sub:
        sec = (
            pd.Series([t["sector"] for t in all_sub])
            .value_counts()
            .rename_axis("Sector")
            .to_frame("Sub-suppliers")
        )
        st.markdown("**Sector exposure** (Tiers 2–4)")
        st.dataframe(sec, use_container_width=True)
