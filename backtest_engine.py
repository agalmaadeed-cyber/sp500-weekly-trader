"""
backtest_engine.py
منطق الـ backtest — مستقل عن الواجهة
"""

import pandas as pd
import numpy as np
from data_loader import download_batch, get_sp500_tickers
from rsi_divergence import detect_signals

IS_END      = pd.Timestamp("2021-12-31")
MAX_CANDLES = 48


def simulate_trades(df: pd.DataFrame, signals: pd.DataFrame) -> list:
    if signals.empty:
        return []

    trades   = []
    idx_list = df.index.tolist()

    for sig_date, sig in signals.iterrows():
        entry     = sig["entry"]
        stop      = sig["stop"]
        target1   = sig["target1"]
        target2   = sig["target2"]
        direction = sig["direction"]

        try:
            pos = idx_list.index(sig_date) + 1
        except ValueError:
            continue

        future = df.iloc[pos: pos + MAX_CANDLES]
        if future.empty:
            continue

        outcome    = "timeout"
        exit_price = future.iloc[-1]["close"]
        exit_date  = future.index[-1]

        for dt, row in future.iterrows():
            hi, lo = row["high"], row["low"]
            if direction == "long":
                hit_stop = lo <= stop
                hit_t1   = hi >= target1
                hit_t2   = hi >= target2
            else:
                hit_stop = hi >= stop
                hit_t1   = lo <= target1
                hit_t2   = lo <= target2

            if hit_stop and (hit_t1 or hit_t2):
                outcome, exit_price, exit_date = "stop", stop, dt
                break
            elif hit_t2:
                outcome, exit_price, exit_date = "target2", target2, dt
                break
            elif hit_t1:
                outcome, exit_price, exit_date = "target1", target1, dt
                break
            elif hit_stop:
                outcome, exit_price, exit_date = "stop", stop, dt
                break

        risk = abs(entry - stop)
        pnl  = (exit_price - entry) if direction == "long" else (entry - exit_price)
        r_multiple = round(pnl / risk, 3) if risk > 0 else 0

        trades.append({
            "entry_date": sig_date,
            "exit_date":  exit_date,
            "direction":  direction,
            "entry":      entry,
            "exit":       exit_price,
            "outcome":    outcome,
            "r_multiple": r_multiple,
            "win":        outcome in ("target1", "target2"),
            "period":     "IS" if sig_date <= IS_END else "OOS",
        })

    return trades


def run_backtest(tickers: list[str] | None = None,
                 progress_cb=None) -> pd.DataFrame:
    """
    تشغيل الـ backtest الكامل
    progress_cb: دالة تُستدعى بـ (i, total, ticker) لتحديث شريط التقدم
    """
    if tickers is None:
        tickers = get_sp500_tickers()

    all_data   = download_batch(tickers)
    all_trades = []

    for i, (ticker, df) in enumerate(all_data.items()):
        if progress_cb:
            progress_cb(i, len(all_data), ticker)
        signals = detect_signals(df)
        trades  = simulate_trades(df, signals)
        for t in trades:
            t["ticker"] = ticker
        all_trades.extend(trades)

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def compute_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}

    def period_stats(sub):
        if sub.empty:
            return {"trades": 0, "win_rate": 0, "avg_r": 0, "total_r": 0}
        wins = sub["win"].sum()
        return {
            "trades":   len(sub),
            "win_rate": round(wins / len(sub) * 100, 1),
            "avg_r":    round(sub["r_multiple"].mean(), 3),
            "total_r":  round(sub["r_multiple"].sum(), 1),
        }

    by_year = (df.groupby([df["entry_date"].dt.year, "direction"])
                 .agg(trades=("win","count"),
                      win_rate=("win","mean"),
                      avg_r=("r_multiple","mean"))
                 .reset_index())
    by_year["win_rate"] = (by_year["win_rate"] * 100).round(1)
    by_year["avg_r"]    = by_year["avg_r"].round(3)

    return {
        "is":      period_stats(df[df["period"] == "IS"]),
        "oos":     period_stats(df[df["period"] == "OOS"]),
        "all":     period_stats(df),
        "by_year": by_year,
    }
