"""
supabase_storage.py
Paper trading storage via Supabase.
Trade lifecycle: pending -> open -> closed

Signal data  = what the system detected (signal candle close)
Actual data  = what really happened (next day open for entry, real exit price)
"""

import requests
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta

SUPABASE_URL = "https://rmzbadakigsdlwxrjqtg.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InJtemJhZGFraWdzZGx3eHJqcXRnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODE2MTgxMTYsImV4cCI6MjA5NzE5NDExNn0.kGafDc9_5i7VP6jqZUCrYVmafjlCtI9tkrqoZhoh_6s"
TABLE = "trades"

HEADERS = {
    "apikey":        SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type":  "application/json",
    "Prefer":        "return=representation",
}

STOP_ATR  = 1.5
T1_ATR    = 2.0
T2_ATR    = 4.0


# ── Internal helpers ──────────────────────────────────────────

def _get(params: dict) -> list:
    r = requests.get(
        f"{SUPABASE_URL}/rest/v1/{TABLE}",
        headers=HEADERS, params=params, timeout=10,
    )
    return r.json() if r.ok else []


def _post(data: dict) -> dict:
    r = requests.post(
        f"{SUPABASE_URL}/rest/v1/{TABLE}",
        headers=HEADERS, json=data, timeout=10,
    )
    return r.json()[0] if r.ok and r.json() else {}


def _patch(row_id: int, data: dict) -> dict:
    r = requests.patch(
        f"{SUPABASE_URL}/rest/v1/{TABLE}",
        headers=HEADERS,
        params={"id": f"eq.{row_id}"},
        json=data, timeout=10,
    )
    return r.json()[0] if r.ok and r.json() else {}


def _fetch_ohlcv(ticker: str, days: int = 60) -> pd.DataFrame:
    """Fetch daily OHLCV from yfinance."""
    try:
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        raw   = yf.download(ticker, start=start, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns={"Open": "open", "High": "high",
                                   "Low": "low",  "Close": "close"})
        raw.index.name = "date"
        return raw
    except Exception:
        return pd.DataFrame()


# ── Public read API ───────────────────────────────────────────

def get_pending_trades() -> pd.DataFrame:
    rows = _get({"status": "eq.pending", "order": "id.asc"})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_open_trades() -> pd.DataFrame:
    rows = _get({"status": "eq.open", "order": "id.asc"})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_closed_trades() -> pd.DataFrame:
    rows = _get({"status": "eq.closed", "order": "id.asc"})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_all_trades() -> pd.DataFrame:
    rows = _get({"order": "id.asc"})
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_summary() -> dict:
    rows = _get({"status": "eq.closed"})
    if not rows:
        return {"trades": 0, "win_rate": 0, "avg_r": 0, "total_r": 0}
    df   = pd.DataFrame(rows)
    wins = df[df["r_multiple"] > 0]
    return {
        "trades":   len(df),
        "win_rate": round(len(wins) / len(df) * 100, 1),
        "avg_r":    round(df["r_multiple"].mean(), 3),
        "total_r":  round(df["r_multiple"].sum(), 2),
    }


# ── Trade lifecycle ───────────────────────────────────────────

def open_trade(ticker: str, direction: str,
               entry_price: float, stop: float,
               target1: float, target2: float,
               rsi: float = None, atr: float = None,
               app: str = "weekly") -> dict:
    """
    Register a new PENDING trade.
    Signal prices are stored. Actual prices filled on next update.
    """
    row = {
        "app":           app,
        "ticker":        ticker,
        "direction":     direction,
        "status":        "pending",
        "entry_date":    str(date.today()),

        # Signal data
        "signal_date":   str(date.today()),
        "signal_entry":  entry_price,
        "signal_stop":   stop,
        "signal_target1": target1,
        "signal_target2": target2,
        "signal_rsi":    rsi,
        "signal_atr":    atr,

        # Actual data — filled later
        "entry_price":   None,
        "stop":          None,
        "target1":       None,
        "target2":       None,
        "actual_entry":  None,
        "actual_stop":   None,
        "actual_target1": None,
        "actual_target2": None,
        "actual_exit":   None,
        "actual_exit_date": None,
        "slippage":      None,
        "hold_days":     None,
        "outcome":       None,
        "r_multiple":    None,
    }
    return _post(row)


def _activate_pending(trade: dict, df: pd.DataFrame) -> bool:
    """
    Try to activate a pending trade using the next day's open after signal_date.
    Returns True if activated.
    """
    signal_date = pd.Timestamp(trade["signal_date"])
    future = df[df.index > signal_date]
    if future.empty:
        return False

    next_day    = future.iloc[0]
    actual_entry = float(next_day["open"])
    atr          = float(trade["signal_atr"]) if trade["signal_atr"] else None

    if not atr:
        return False

    direction = trade["direction"]
    if direction == "long":
        actual_stop    = round(actual_entry - STOP_ATR * atr, 4)
        actual_target1 = round(actual_entry + T1_ATR  * atr, 4)
        actual_target2 = round(actual_entry + T2_ATR  * atr, 4)
    else:
        actual_stop    = round(actual_entry + STOP_ATR * atr, 4)
        actual_target1 = round(actual_entry - T1_ATR  * atr, 4)
        actual_target2 = round(actual_entry - T2_ATR  * atr, 4)

    slippage = round(actual_entry - float(trade["signal_entry"]), 4)

    _patch(trade["id"], {
        "status":         "open",
        "actual_entry":   actual_entry,
        "actual_stop":    actual_stop,
        "actual_target1": actual_target1,
        "actual_target2": actual_target2,
        "entry_price":    actual_entry,
        "stop":           actual_stop,
        "target1":        actual_target1,
        "target2":        actual_target2,
        "slippage":       slippage,
        "entry_date":     str(next_day.name.date()),
    })
    return True


def _check_exit(trade: dict, df: pd.DataFrame) -> bool:
    """
    Check if an open trade has hit stop or target.
    Returns True if closed.
    """
    entry_date = pd.Timestamp(trade["entry_date"])
    future     = df[df.index > entry_date]
    if future.empty:
        return False

    entry     = float(trade["actual_entry"] or trade["entry_price"])
    stop      = float(trade["actual_stop"]  or trade["stop"])
    target1   = float(trade["actual_target1"] or trade["target1"])
    target2   = float(trade["actual_target2"] or trade["target2"])
    direction = trade["direction"]
    open_date = pd.Timestamp(trade["entry_date"])

    # Check candle by candle
    for dt, row in future.iterrows():
        hi = float(row["high"])
        lo = float(row["low"])

        if direction == "long":
            hit_stop = lo <= stop
            hit_t1   = hi >= target1
            hit_t2   = hi >= target2
        else:
            hit_stop = hi >= stop
            hit_t1   = lo <= target1
            hit_t2   = lo <= target2

        # Conservative: if both stop and target hit same candle, stop wins
        if hit_stop and (hit_t1 or hit_t2):
            outcome, exit_price = "stop", stop
        elif hit_t2:
            outcome, exit_price = "target2", target2
        elif hit_t1:
            outcome, exit_price = "target1", target1
        elif hit_stop:
            outcome, exit_price = "stop", stop
        else:
            continue

        risk       = abs(entry - stop)
        pnl        = (exit_price - entry) if direction == "long" else (entry - exit_price)
        r_multiple = round(pnl / risk, 3) if risk > 0 else 0
        hold_days  = (dt.date() - open_date.date()).days

        _patch(trade["id"], {
            "status":           "closed",
            "outcome":          outcome,
            "actual_exit":      exit_price,
            "actual_exit_date": str(dt.date()),
            "exit_price":       exit_price,
            "r_multiple":       r_multiple,
            "hold_days":        hold_days,
        })
        return True

    return False


def update_all_positions() -> dict:
    """
    Called on app load. Activates pending trades and checks open trades for exits.
    Returns summary of what happened.
    """
    summary = {"activated": 0, "closed": 0, "errors": 0}

    # 1. Activate pending trades
    pending = get_pending_trades()
    if not pending.empty:
        tickers = pending["ticker"].unique().tolist()
        for ticker in tickers:
            df = _fetch_ohlcv(ticker, days=10)
            if df.empty:
                summary["errors"] += 1
                continue
            for _, trade in pending[pending["ticker"] == ticker].iterrows():
                if _activate_pending(trade.to_dict(), df):
                    summary["activated"] += 1

    # 2. Check open trades for exits
    open_trades = get_open_trades()
    if not open_trades.empty:
        tickers = open_trades["ticker"].unique().tolist()
        for ticker in tickers:
            df = _fetch_ohlcv(ticker, days=60)
            if df.empty:
                summary["errors"] += 1
                continue
            for _, trade in open_trades[open_trades["ticker"] == ticker].iterrows():
                if _check_exit(trade.to_dict(), df):
                    summary["closed"] += 1

    return summary


def close_trade(trade_id: int, exit_price: float, outcome: str) -> dict:
    """Manual close."""
    open_df = get_open_trades()
    if open_df.empty:
        return {}
    row = open_df[open_df["id"] == trade_id]
    if row.empty:
        return {}

    entry     = float(row.iloc[0]["actual_entry"] or row.iloc[0]["entry_price"])
    stop      = float(row.iloc[0]["actual_stop"]  or row.iloc[0]["stop"])
    direction = row.iloc[0]["direction"]
    open_date = pd.Timestamp(row.iloc[0]["entry_date"])

    risk       = abs(entry - stop)
    pnl        = (exit_price - entry) if direction == "long" else (entry - exit_price)
    r_multiple = round(pnl / risk, 3) if risk > 0 else 0
    hold_days  = (date.today() - open_date.date()).days

    return _patch(trade_id, {
        "status":           "closed",
        "outcome":          outcome,
        "actual_exit":      exit_price,
        "actual_exit_date": str(date.today()),
        "exit_price":       exit_price,
        "r_multiple":       r_multiple,
        "hold_days":        hold_days,
    })
