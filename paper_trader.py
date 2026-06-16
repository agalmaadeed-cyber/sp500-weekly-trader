"""
paper_trader.py
تتبع الصفقات الافتراضية — فتح وإغلاق وسجل
"""

import pandas as pd
from pathlib import Path
from datetime import date

TRADES_FILE = Path(__file__).parent / "data" / "trades.csv"

COLUMNS = [
    "id", "ticker", "direction", "entry_date", "entry_price",
    "stop", "target1", "target2", "rsi", "atr",
    "exit_date", "exit_price", "outcome", "r_multiple", "status"
]


def _load() -> pd.DataFrame:
    TRADES_FILE.parent.mkdir(exist_ok=True)
    if TRADES_FILE.exists():
        return pd.read_csv(TRADES_FILE, parse_dates=["entry_date", "exit_date"])
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame):
    TRADES_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(TRADES_FILE, index=False)


def get_trades() -> pd.DataFrame:
    return _load()


def get_open_trades() -> pd.DataFrame:
    df = _load()
    return df[df["status"] == "open"] if not df.empty else df


def get_closed_trades() -> pd.DataFrame:
    df = _load()
    return df[df["status"] == "closed"] if not df.empty else df


def open_trade(ticker: str, direction: str, entry_price: float,
               stop: float, target1: float, target2: float,
               rsi: float, atr: float) -> dict:
    df = _load()
    new_id = int(df["id"].max() + 1) if not df.empty and "id" in df.columns else 1

    row = {
        "id":           new_id,
        "ticker":       ticker,
        "direction":    direction,
        "entry_date":   str(date.today()),
        "entry_price":  entry_price,
        "stop":         stop,
        "target1":      target1,
        "target2":      target2,
        "rsi":          rsi,
        "atr":          atr,
        "exit_date":    None,
        "exit_price":   None,
        "outcome":      None,
        "r_multiple":   None,
        "status":       "open",
    }

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    _save(df)
    return row


def close_trade(trade_id: int, exit_price: float, outcome: str):
    df = _load()
    idx = df[df["id"] == trade_id].index
    if idx.empty:
        return

    trade = df.loc[idx[0]]
    entry = trade["entry_price"]
    stop  = trade["stop"]
    risk  = abs(entry - stop)

    if trade["direction"] == "long":
        pnl = exit_price - entry
    else:
        pnl = entry - exit_price

    r_multiple = round(pnl / risk, 3) if risk > 0 else 0

    df.loc[idx[0], "exit_date"]  = str(date.today())
    df.loc[idx[0], "exit_price"] = exit_price
    df.loc[idx[0], "outcome"]    = outcome
    df.loc[idx[0], "r_multiple"] = r_multiple
    df.loc[idx[0], "status"]     = "closed"

    _save(df)


def get_summary() -> dict:
    closed = get_closed_trades()
    if closed.empty:
        return {"trades": 0, "win_rate": 0, "avg_r": 0, "total_r": 0}

    wins     = closed[closed["r_multiple"] > 0]
    win_rate = len(wins) / len(closed) * 100
    avg_r    = closed["r_multiple"].mean()
    total_r  = closed["r_multiple"].sum()

    return {
        "trades":   len(closed),
        "win_rate": round(win_rate, 1),
        "avg_r":    round(avg_r, 3),
        "total_r":  round(total_r, 2),
    }
