# tech_fetcher.py
import os, time, math, json, itertools
from typing import List, Dict
import pandas as pd
import numpy as np
import yfinance as yf

# ---------- Config ----------
DATA_DIR = "data/prices"
os.makedirs(DATA_DIR, exist_ok=True)

LOOKBACK_DAYS = 200   # fetch extra for warm-up
USE_DAYS = 120        # keep last 120 trading days after computing indicators
BATCH_SIZE = 30       # yfinance batch size (tune 20-50)
RETRY = 2             # retries per batch

# ---------- Utilities ----------

def _start_date(days: int) -> str:
    return (pd.Timestamp.utcnow().normalize()- pd.Timedelta(days=days*1.4)).strftime("%Y-%m-%d")
    # 1.4x cushion so we truly get ~days trading sessions

def _normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    # unify column names
    cols = {c: c.lower() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    if "date" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
        df.rename(columns={"index": "date"}, inplace=True)
    # enforce tz-naive
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date").drop_duplicates(subset=["date"])
    # ensure numeric
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _compute_indicators(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    close = df["close"]

    # RSI(14)
    delta = close.diff()
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gains, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    roll_down = pd.Series(losses, index=df.index).ewm(alpha=1/14, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df["rsi14"] = (100 - 100/(1+rs)).fillna(method="bfill").clip(0,100)

    # MACD (12,26,9)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal

    # SMAs
    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    return df

def _final_slice(df: pd.DataFrame, use_days=USE_DAYS) -> pd.DataFrame:
    if df.empty:
        return df
    return df.tail(use_days)

def _load_cache(ticker: str) -> pd.DataFrame:
    safe = ticker.replace("/", "_")
    path = os.path.join(DATA_DIR, f"{safe}.parquet")
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()
    
def _save_cache(ticker: str, df: pd.DataFrame):
    safe = ticker.replace("/", "_")
    path = os.path.join(DATA_DIR, f"{safe}.parquet")
    try:
        df.to_parquet(path, index=False)
    except Exception:
        # fallback to CSV on environments without parquet engine
        df.to_csv(path.replace(".parquet", ".csv"), index=False)

# ---------- Core ----------
def _download_batch(tickers: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Multi-ticker download with yfinance. Returns dict[ticker]->raw df
    """
    out = {t: pd.DataFrame() for t in tickers}
    if not tickers:
        return out

    start = _start_date(LOOKBACK_DAYS)
    # history() with multiple tickers sometimes more reliable than download()
    try:
        df = yf.download(
            tickers=tickers,
            start=start,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
    except Exception:
        df = pd.DataFrame()

    # yfinance returns wide panel when multiple tickers:
    if isinstance(df.columns, pd.MultiIndex):
        # split per ticker
        for t in tickers:
            if t in df.columns.levels[0]:
                sub = df[t].reset_index()
                sub = sub.rename(columns=str.title)  # Date, Open...
                out[t] = _normalize_frame(sub)
    else:
        # single-frame case or error; if single ticker it will be a flat df
        if len(tickers) == 1:
            out[tickers[0]] = _normalize_frame(df.reset_index())
        # else leave empties; we will retry

    return out

def fetch_prices_with_indicators(
    tickers: List[str],
    force_refresh: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict[ticker] -> DataFrame[date, open, high, low, close, volume, rsi14, macd, macd_signal, macd_hist, sma20, sma50]
    Caches per ticker to data/prices/{ticker}.parquet
    """
    tickers = [t for t in tickers if t and t != "Not Listed"]
    tickers = sorted(set(tickers))

    prices: Dict[str, pd.DataFrame] = {}
    need_download: List[str] = []

    # Try cache first
    for t in tickers:
        df = _load_cache(t)
        if not force_refresh and not df.empty and (df["date"].max() >= _now_utc_date() - pd.Timedelta(days=5)):
            prices[t] = df  # fresh enough (<= 5 days old)
        else:
            need_download.append(t)

    # Batch download the rest with retries
    for i in range(0, len(need_download), BATCH_SIZE):
        batch = need_download[i:i+BATCH_SIZE]
        remains = list(batch)
        for attempt in range(RETRY + 1):
            if not remains:
                break
            fetched = _download_batch(remains)
            new_remains = []
            for t in remains:
                raw = fetched.get(t, pd.DataFrame())
                if raw is None or raw.empty:
                    new_remains.append(t)
                    continue
                # indicators
                with_ind = _compute_indicators(raw)
                final = _final_slice(with_ind)
                if final.empty:
                    new_remains.append(t)
                else:
                    prices[t] = final
                    _save_cache(t, final)
            remains = new_remains
            if remains:
                time.sleep(1.2)  # gentle backoff
        # mark unretrieved as empty
        for t in remains:
            prices[t] = pd.DataFrame()

    return prices

# ---------- Convenience: from your Ettimes JSON ----------
def tickers_from_recos_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tickers = [d.get("ticker") for d in data if d.get("ticker") and d.get("ticker") != "Not Listed"]
    return sorted(set(tickers))

if __name__ == "__main__":
    # Example wiring
    recos_path = "ds_outputs/stock_list.json"  # your file
    tickers = tickers_from_recos_json(recos_path)
    prices = fetch_prices_with_indicators(tickers)

    # Save a tidy combined CSV for quick sanity checks
    frames = []
    for t, df in prices.items():
        if df is None or df.empty: 
            continue
        temp = df.copy()
        temp.insert(1, "ticker", t)
        frames.append(temp)
    if frames:
        tidy = pd.concat(frames, ignore_index=True)
        tidy.to_csv("data/technical_data_tidy.csv", index=False)
        print(f"Saved data/technical_data_tidy.csv with {len(tidy)} rows")
    else:
        print("No data fetched.")
