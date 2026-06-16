"""
app.py — SP500-RSI Trader
Trade lifecycle: pending (signal night) -> open (next day open) -> closed (auto on stop/target)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date

from data_loader import get_sp500_tickers, download_batch
from rsi_divergence import detect_signals
from supabase_storage import (get_pending_trades, get_open_trades, get_closed_trades,
                               open_trade, close_trade, get_summary, update_all_positions)
from backtest_engine import run_backtest, compute_stats

INITIAL_CAPITAL = 10_000
RISK_PER_TRADE  = 0.01

st.set_page_config(page_title="SP500-RSI Weekly Trader", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
:root {
    --bg:#0a0b0f; --surface:#111318; --border:#1e2230;
    --accent:#00f5a0; --accent2:#00c6ff; --red:#ff4d6d;
    --yellow:#ffd166; --text:#e8eaf0; --muted:#6b7280;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;}
[data-testid="stHeader"]{background:transparent!important;}
[data-testid="stTabs"] button{font-size:0.75rem!important;letter-spacing:0.1em;color:var(--muted)!important;border-bottom:2px solid transparent!important;text-transform:uppercase;}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important;border-bottom:2px solid var(--accent)!important;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1.2rem 1.4rem;margin-bottom:0.5rem;}
.metric-label{font-size:0.65rem;letter-spacing:0.12em;color:var(--muted);text-transform:uppercase;margin-bottom:0.4rem;}
.metric-value{font-size:1.6rem;font-weight:700;color:var(--accent);line-height:1;}
.metric-value.red{color:var(--red);} .metric-value.yellow{color:var(--yellow);} .metric-value.blue{color:var(--accent2);}
.signal-row{background:var(--surface);border:1px solid var(--border);border-left:3px solid var(--accent);border-radius:6px;padding:0.9rem 1.1rem;margin-bottom:0.5rem;}
.signal-row.bear{border-left-color:var(--red);}
.signal-row.pending-row{border-left-color:var(--yellow);}
.signal-symbol{font-weight:700;font-size:1rem;color:var(--text);}
.signal-detail{font-size:0.75rem;color:var(--muted);margin-top:0.5rem;line-height:1.8;}
.tag{font-size:0.6rem;letter-spacing:0.1em;padding:0.2rem 0.6rem;border-radius:3px;text-transform:uppercase;}
.tag-green{background:rgba(0,245,160,0.12);color:var(--accent);border:1px solid rgba(0,245,160,0.25);}
.tag-red{background:rgba(255,77,109,0.12);color:var(--red);border:1px solid rgba(255,77,109,0.25);}
.tag-blue{background:rgba(0,198,255,0.12);color:var(--accent2);border:1px solid rgba(0,198,255,0.25);}
.tag-yellow{background:rgba(255,209,102,0.1);color:var(--yellow);border:1px solid rgba(255,209,102,0.2);}
.trade-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:1rem 1.2rem;margin-bottom:0.6rem;}
.price-grid{display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:0.6rem;}
.price-item label{font-size:0.6rem;letter-spacing:0.1em;color:var(--muted);text-transform:uppercase;display:block;}
.price-item span{font-size:0.85rem;color:var(--text);}
.compare-grid{display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.6rem;}
.compare-col label{font-size:0.6rem;letter-spacing:0.1em;color:var(--muted);text-transform:uppercase;display:block;margin-bottom:0.3rem;}
.compare-item{font-size:0.78rem;color:var(--text);padding:0.15rem 0;}
.compare-item.diff-pos{color:var(--accent);}
.compare-item.diff-neg{color:var(--red);}
.section-title{font-size:0.7rem;letter-spacing:0.14em;color:var(--muted);text-transform:uppercase;margin:1.2rem 0 0.8rem;padding-bottom:0.4rem;border-bottom:1px solid var(--border);}
.empty-state{text-align:center;padding:3rem 1rem;color:var(--muted);font-size:0.75rem;letter-spacing:0.08em;}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px;}
.dot-green{background:var(--accent);box-shadow:0 0 6px var(--accent);}
.dot-yellow{background:var(--yellow);}
hr{border-color:var(--border)!important;}
.stButton>button{background:var(--surface)!important;border:1px solid var(--border)!important;color:var(--text)!important;font-size:0.7rem!important;letter-spacing:0.08em!important;border-radius:6px!important;}
.stButton>button:hover{border-color:var(--accent)!important;color:var(--accent)!important;}
[data-testid="stSelectbox"]>div>div{background:var(--surface)!important;}
[data-testid="stNumberInput"] input{background:var(--surface)!important;color:var(--text)!important;}
</style>
""", unsafe_allow_html=True)


def position_size(entry, stop, capital=INITIAL_CAPITAL):
    risk_per_unit = abs(entry - stop)
    if risk_per_unit == 0:
        return 0
    return round((capital * RISK_PER_TRADE) / risk_per_unit, 4)


# ── Auto-update on every app load ────────────────────────────
if "positions_updated" not in st.session_state:
    with st.spinner("Updating positions..."):
        update_summary = update_all_positions()
    st.session_state["positions_updated"]  = True
    st.session_state["update_summary"]     = update_summary


st.markdown(
    '<div style="padding:1.5rem 0 0.5rem;border-bottom:1px solid #1e2230;margin-bottom:1.5rem;">'
    '<div style="font-size:1.1rem;font-weight:700;letter-spacing:0.08em;color:#e8eaf0;">SP500-RSI WEEKLY TRADER</div>'
    '<div style="font-size:0.8rem;color:#6b7280;margin-top:0.15rem;">Momentum Divergence · Weekly · RSI Divergence on S&P 500</div>'
    '</div>',
    unsafe_allow_html=True
)

# Show update summary if something happened
us = st.session_state.get("update_summary", {})
if us.get("activated", 0) > 0 or us.get("closed", 0) > 0:
    parts = []
    if us["activated"] > 0:
        parts.append(f"{us['activated']} trade(s) activated")
    if us["closed"] > 0:
        parts.append(f"{us['closed']} trade(s) closed automatically")
    st.success(" · ".join(parts))

tab_scanner, tab_paper, tab_backtest, tab_logic = st.tabs([
    "SCANNER", "PAPER TRADER", "BACKTEST", "LOGIC"
])


# ════════════════════════════════════════════════════════════
# TAB 1 — SCANNER
# ════════════════════════════════════════════════════════════
with tab_scanner:
    st.markdown('<div class="section-title">Scanner Settings</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        limit = st.selectbox("Number of stocks", [50, 100, 200, 500], index=1)
    with col2:
        lookback_days = st.number_input("Signal lookback (days)", min_value=7, max_value=60, value=30)

    scan_btn = st.button("SCAN MARKET", use_container_width=True, key="scan_btn")

    if scan_btn:
        tickers       = get_sp500_tickers()[:limit]
        progress      = st.progress(0, text="Loading data...")
        signals_found = []

        with st.spinner("Scanning..."):
            data = download_batch(tickers, use_cache=False)
            for i, (ticker, df) in enumerate(data.items()):
                progress.progress((i + 1) / len(data), text=f"Scanning {ticker}...")
                sigs = detect_signals(df)
                if sigs.empty:
                    continue
                latest_date = df.index[-1].normalize()
                cutoff      = latest_date - pd.Timedelta(days=lookback_days)
                recent      = sigs[sigs.index.normalize() >= cutoff]
                for sig_date, sig in recent.iterrows():
                    size     = position_size(sig["entry"], sig["stop"])
                    risk_usd = round(abs(sig["entry"] - sig["stop"]) * size, 2)
                    signals_found.append({
                        "ticker":    ticker,
                        "date":      str(sig_date.date()),
                        "direction": sig["direction"],
                        "entry":     sig["entry"],
                        "stop":      sig["stop"],
                        "target1":   sig["target1"],
                        "target2":   sig["target2"],
                        "rsi":       sig["rsi"],
                        "atr":       sig["atr"],
                        "size":      size,
                        "risk_usd":  risk_usd,
                    })

        progress.empty()
        st.session_state["scan_results"] = signals_found

    if "scan_results" in st.session_state:
        results = st.session_state["scan_results"]
        st.markdown('<div class="section-title">Signals</div>', unsafe_allow_html=True)

        if not results:
            st.markdown('<div class="empty-state">NO SIGNALS FOUND IN LOOKBACK WINDOW</div>', unsafe_allow_html=True)
        else:
            longs  = [s for s in results if s["direction"] == "long"]
            shorts = [s for s in results if s["direction"] == "short"]
            st.markdown(
                f'<div style="font-size:0.7rem;color:var(--muted);margin-bottom:0.8rem;">'
                f'<span class="dot dot-green"></span>'
                f'{len(results)} SIGNAL{"S" if len(results)>1 else ""} FOUND'
                f' &nbsp;·&nbsp; {len(longs)} LONG &nbsp;·&nbsp; {len(shorts)} SHORT</div>',
                unsafe_allow_html=True
            )

            pending_df = get_pending_trades()
            open_df    = get_open_trades()
            taken_syms = set()
            if not pending_df.empty:
                taken_syms.update(pending_df["ticker"].tolist())
            if not open_df.empty:
                taken_syms.update(open_df["ticker"].tolist())

            for i, sig in enumerate(results):
                is_long   = sig["direction"] == "long"
                cls       = "" if is_long else "bear"
                tag_cls   = "tag-green" if is_long else "tag-red"
                dir_label = "LONG" if is_long else "SHORT"
                taken     = sig["ticker"] in taken_syms
                risk      = abs(sig["entry"] - sig["stop"])
                rr1       = round(abs(sig["target1"] - sig["entry"]) / risk, 1) if risk else 0
                taken_tag = "<span class='tag tag-yellow' style='margin-left:0.4rem;'>IN TRADE</span>" if taken else ""

                with st.container():
                    st.markdown(
                        f'<div class="signal-row {cls}">'
                        f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                        f'<span class="signal-symbol">{sig["ticker"]}</span>'
                        f'<span class="tag {tag_cls}">{dir_label}</span>'
                        f'{taken_tag}'
                        f'</div>'
                        f'<div class="signal-detail">'
                        f'Date: <b>{sig["date"]}</b> &nbsp;|&nbsp;'
                        f'Signal Entry: <b>{sig["entry"]}</b> &nbsp;|&nbsp;'
                        f'Stop: <b>{sig["stop"]}</b> &nbsp;|&nbsp;'
                        f'T1: <b>{sig["target1"]}</b> &nbsp;|&nbsp;'
                        f'T2: <b>{sig["target2"]}</b> &nbsp;|&nbsp;'
                        f'RSI: {sig["rsi"]} &nbsp;|&nbsp; RR: 1:{rr1} &nbsp;|&nbsp;'
                        f'Size: {sig["size"]} &nbsp;|&nbsp; Risk: ${sig["risk_usd"]}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if not taken:
                        if st.button(f"ENTER TRADE -- {sig['ticker']}", key=f"enter_{i}_{sig['ticker']}"):
                            result = open_trade(
                                ticker=sig["ticker"],
                                direction=sig["direction"],
                                entry_price=sig["entry"],
                                stop=sig["stop"],
                                target1=sig["target1"],
                                target2=sig["target2"],
                                rsi=sig["rsi"],
                                atr=sig["atr"],
                            )
                            if result:
                                st.session_state["trade_opened"] = sig["ticker"]
                                del st.session_state["scan_results"]
                                st.rerun()
                            else:
                                st.error(f"Failed to save trade for {sig['ticker']}.")

    if "trade_opened" in st.session_state:
        ticker_opened = st.session_state.pop("trade_opened")
        st.success(
            f"Trade registered as PENDING: {ticker_opened} -- "
            f"Will activate at tomorrow's open price. Check Paper Trader tab."
        )


# ════════════════════════════════════════════════════════════
# TAB 2 — PAPER TRADER
# ════════════════════════════════════════════════════════════
with tab_paper:
    col_refresh, _ = st.columns([1, 3])
    if col_refresh.button("Refresh Positions", key="refresh_btn"):
        st.session_state.pop("positions_updated", None)
        st.rerun()

    st.markdown('<div class="section-title">Portfolio Overview</div>', unsafe_allow_html=True)

    closed_df  = get_closed_trades()
    open_df    = get_open_trades()
    pending_df = get_pending_trades()

    if not closed_df.empty:
        wins     = closed_df[closed_df["r_multiple"] > 0]
        total_r  = closed_df["r_multiple"].sum()
        win_rate = round(len(wins) / len(closed_df) * 100, 1)
        avg_r    = round(closed_df["r_multiple"].mean(), 2)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        for col, label, value, cls in [
            (c1, "Total R",       f"{total_r:+.2f}R", "red" if total_r < 0 else ""),
            (c2, "Win Rate",      f"{win_rate}%",      "yellow" if win_rate < 50 else ""),
            (c3, "Avg R",         f"{avg_r}",          "red" if avg_r < 0 else ""),
            (c4, "Closed Trades", len(closed_df),      ""),
            (c5, "Open",          len(open_df),        "blue"),
            (c6, "Pending",       len(pending_df),     "yellow"),
        ]:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-value {cls}">{value}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.markdown('<div class="empty-state" style="padding:1rem;">NO CLOSED TRADES YET</div>', unsafe_allow_html=True)

    # ── Pending trades ────────────────────────────────────────
    if not pending_df.empty:
        st.markdown('<div class="section-title">Pending Trades — Waiting for Next Open</div>', unsafe_allow_html=True)
        for _, t in pending_df.iterrows():
            is_long   = t["direction"] == "long"
            dir_label = "LONG" if is_long else "SHORT"
            tag_cls   = "tag-green" if is_long else "tag-red"
            st.markdown(
                f'<div class="signal-row pending-row">'
                f'<div style="display:flex;align-items:center;gap:0.5rem;">'
                f'<span class="signal-symbol">{t["ticker"]}</span>'
                f'<span class="tag {tag_cls}">{dir_label}</span>'
                f'<span class="tag tag-yellow" style="margin-left:0.3rem;">PENDING</span>'
                f'</div>'
                f'<div class="signal-detail">'
                f'Signal date: <b>{t["signal_date"]}</b> &nbsp;|&nbsp;'
                f'Signal entry: <b>{t["signal_entry"]}</b> &nbsp;|&nbsp;'
                f'Signal stop: <b>{t["signal_stop"]}</b> &nbsp;|&nbsp;'
                f'Signal T1: <b>{t["signal_target1"]}</b> &nbsp;|&nbsp;'
                f'RSI: {t["signal_rsi"]}'
                f'<br><span style="color:var(--yellow);">Will activate at tomorrow\'s open. Actual prices will be set then.</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # ── Open positions ────────────────────────────────────────
    st.markdown('<div class="section-title">Open Positions</div>', unsafe_allow_html=True)

    if open_df.empty:
        st.markdown(
            '<div class="empty-state" style="padding:1.5rem;">NO OPEN POSITIONS'
            '<br><span style="font-size:0.65rem;opacity:0.5;">Use the Scanner to find signals</span></div>',
            unsafe_allow_html=True
        )
    else:
        for _, t in open_df.iterrows():
            is_long   = t["direction"] == "long"
            dir_tag   = "tag-green" if is_long else "tag-red"
            dir_label = "LONG" if is_long else "SHORT"

            # Slippage display
            slippage     = t.get("slippage") or 0
            slip_cls     = "diff-pos" if (is_long and slippage < 0) or (not is_long and slippage > 0) else "diff-neg"
            slip_display = f"{slippage:+.4f}" if slippage else "N/A"

            with st.container():
                st.markdown(
                    f'<div class="trade-card">'
                    f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.6rem;">'
                    f'<div>'
                    f'<span class="signal-symbol">{t["ticker"]}</span>'
                    f'<span class="tag {dir_tag}" style="margin-left:0.5rem;">{dir_label}</span>'
                    f'</div>'
                    f'<div style="font-size:0.7rem;color:var(--muted);">ID: {t["id"]} &nbsp;·&nbsp; Entered: {t["entry_date"]}</div>'
                    f'</div>'
                    f'<div class="compare-grid">'
                    f'<div><label>Signal Prices</label>'
                    f'<div class="compare-item">Entry: {t["signal_entry"]}</div>'
                    f'<div class="compare-item">Stop: {t["signal_stop"]}</div>'
                    f'<div class="compare-item">T1: {t["signal_target1"]}</div>'
                    f'<div class="compare-item">T2: {t["signal_target2"]}</div>'
                    f'</div>'
                    f'<div><label>Actual Prices</label>'
                    f'<div class="compare-item">Entry: <b>{t["actual_entry"]}</b></div>'
                    f'<div class="compare-item">Stop: <b>{t["actual_stop"]}</b></div>'
                    f'<div class="compare-item">T1: <b>{t["actual_target1"]}</b></div>'
                    f'<div class="compare-item">T2: <b>{t["actual_target2"]}</b></div>'
                    f'</div>'
                    f'</div>'
                    f'<div style="margin-top:0.5rem;font-size:0.7rem;color:var(--muted);">'
                    f'Slippage: <span class="{slip_cls}">{slip_display}</span> &nbsp;|&nbsp;'
                    f'RSI at signal: {t["signal_rsi"]}'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                col1, col2, col3 = st.columns([2, 2, 1])
                exit_price = col1.number_input("Exit Price", key=f"ep_{t['id']}", min_value=0.0, step=0.01, format="%.4f")
                outcome    = col2.selectbox("Outcome", ["target1", "target2", "stop", "timeout", "manual"], key=f"out_{t['id']}")
                if col3.button("CLOSE", key=f"close_{t['id']}"):
                    close_trade(int(t["id"]), exit_price, outcome)
                    st.session_state.pop("positions_updated", None)
                    st.rerun()

    # ── Trade history with signal vs actual comparison ────────
    if not closed_df.empty:
        st.markdown('<div class="section-title">Trade History — Signal vs Actual</div>', unsafe_allow_html=True)

        display_cols = [
            "ticker", "direction",
            "signal_date", "signal_entry", "signal_stop", "signal_target1",
            "entry_date",  "actual_entry", "actual_stop", "actual_target1",
            "slippage", "actual_exit", "actual_exit_date",
            "outcome", "r_multiple", "hold_days",
        ]
        available = [c for c in display_cols if c in closed_df.columns]
        st.dataframe(
            closed_df[available].sort_values("actual_exit_date", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "r_multiple": st.column_config.NumberColumn("R", format="%.3f"),
                "slippage":   st.column_config.NumberColumn("Slippage", format="%.4f"),
            }
        )

        # Cumulative R curve
        closed_sorted = closed_df.sort_values("actual_exit_date").copy()
        closed_sorted["cumulative_r"] = closed_sorted["r_multiple"].cumsum()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=closed_sorted["actual_exit_date"], y=closed_sorted["cumulative_r"],
            mode="lines+markers", line=dict(color="#00f5a0", width=2), marker=dict(size=5),
        ))
        fig.update_layout(
            title="Cumulative Performance (R)",
            paper_bgcolor="#0a0b0f", plot_bgcolor="#111318",
            font=dict(color="#e8eaf0"),
            xaxis=dict(gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230"),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Slippage analysis
        if "slippage" in closed_df.columns and closed_df["slippage"].notna().any():
            st.markdown('<div class="section-title">Slippage Analysis</div>', unsafe_allow_html=True)
            avg_slip = closed_df["slippage"].mean()
            c1, c2 = st.columns(2)
            c1.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Avg Slippage</div>'
                f'<div class="metric-value" style="font-size:1.2rem;color:{"var(--red)" if avg_slip > 0 else "var(--accent)"};">'
                f'{avg_slip:+.4f}</div></div>',
                unsafe_allow_html=True
            )
            c2.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Signal Entry Accuracy</div>'
                f'<div class="metric-value" style="font-size:1.2rem;">comparing signal vs actual</div></div>',
                unsafe_allow_html=True
            )


# ════════════════════════════════════════════════════════════
# TAB 3 — BACKTEST
# ════════════════════════════════════════════════════════════
with tab_backtest:
    st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        run_bt = st.button("RUN BACKTEST", use_container_width=True)
    with col2:
        bt_limit = st.selectbox("Stocks", [20, 50, 100, 500], index=1)

    if run_bt:
        tickers     = get_sp500_tickers()[:bt_limit]
        bt_progress = st.progress(0, text="Loading...")
        def update_progress(i, total, ticker):
            bt_progress.progress((i + 1) / total, text=f"Scanning {ticker}...")
        with st.spinner("Running backtest..."):
            results = run_backtest(tickers, progress_cb=update_progress)
        bt_progress.empty()
        if results.empty:
            st.warning("No trades found.")
        else:
            stats = compute_stats(results)
            st.session_state["bt_results"] = results
            st.session_state["bt_stats"]   = stats

    if "bt_stats" in st.session_state:
        stats   = st.session_state["bt_stats"]
        results = st.session_state["bt_results"]
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
        col_is, col_oos, col_all = st.columns(3)
        for col, key, label in [
            (col_is,  "is",  "IN-SAMPLE (2015-2021)"),
            (col_oos, "oos", "OUT-OF-SAMPLE (2022-2024)"),
            (col_all, "all", "TOTAL"),
        ]:
            s = stats.get(key, {})
            with col:
                st.markdown(f"**{label}**")
                for metric, val in [
                    ("Trades",   s.get("trades", 0)),
                    ("Win Rate", f"{s.get('win_rate', 0)}%"),
                    ("Avg R",    s.get("avg_r", 0)),
                    ("Total R",  s.get("total_r", 0)),
                ]:
                    st.markdown(
                        f'<div class="metric-card" style="margin-bottom:0.3rem;">'
                        f'<div class="metric-label">{metric}</div>'
                        f'<div class="metric-value" style="font-size:1.2rem;">{val}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        by_year = stats.get("by_year", pd.DataFrame())
        if not by_year.empty:
            st.markdown('<div class="section-title">Win Rate by Year</div>', unsafe_allow_html=True)
            fig = px.bar(by_year, x="entry_date", y="win_rate", color="direction",
                         barmode="group",
                         color_discrete_map={"long": "#00f5a0", "short": "#ff4d6d"},
                         labels={"entry_date": "Year", "win_rate": "Win Rate %"})
            fig.update_layout(paper_bgcolor="#0a0b0f", plot_bgcolor="#111318",
                              font=dict(color="#e8eaf0"),
                              xaxis=dict(gridcolor="#1e2230"), yaxis=dict(gridcolor="#1e2230"),
                              height=320)
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("DOWNLOAD RESULTS CSV", results.to_csv(index=False),
                           "backtest_results.csv", "text/csv")


# ════════════════════════════════════════════════════════════
# TAB 4 — LOGIC
# ════════════════════════════════════════════════════════════
with tab_logic:
    st.markdown('<div class="section-title">RSI Divergence -- SP500 Daily</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.85rem;color:#e8eaf0;line-height:1.7;">'
        '<b style="color:#00f5a0;">The Idea</b><br>'
        'When price and momentum disagree -- momentum wins.<br><br>'
        '<b style="color:#00f5a0;">Trade Lifecycle</b><br>'
        '1. Signal detected on close of day N.<br>'
        '2. You review and press Enter Trade that evening (PENDING).<br>'
        '3. Next morning, trade activates at the actual open price of day N+1.<br>'
        '4. Stop and targets are recalculated from the actual entry price.<br>'
        '5. System checks daily -- closes automatically when stop or target is hit.<br><br>'
        '<b style="color:#00f5a0;">Bullish Divergence (Long)</b><br>'
        'Price makes a lower low, RSI makes a higher low. RSI must be below 40.<br><br>'
        '<b style="color:#00f5a0;">Bearish Divergence (Short)</b><br>'
        'Price makes a higher high, RSI makes a lower high. RSI must be above 60. '
        'Short only allowed when price is below MA200.'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">Parameters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-card" style="font-size:0.78rem;line-height:1.9;color:#e8eaf0;">'
        '<span style="color:#6b7280;">RSI Period</span> &nbsp;&nbsp; 14<br>'
        '<span style="color:#6b7280;">Swing Window</span> &nbsp;&nbsp; 5 candles each side<br>'
        '<span style="color:#6b7280;">Bull RSI max</span> &nbsp;&nbsp; below 40<br>'
        '<span style="color:#6b7280;">Bear RSI min</span> &nbsp;&nbsp; above 60<br>'
        '<span style="color:#6b7280;">Stop</span> &nbsp;&nbsp; actual_entry minus 1.5 x ATR(14)<br>'
        '<span style="color:#6b7280;">Target 1</span> &nbsp;&nbsp; actual_entry plus 2.0 x ATR(14)<br>'
        '<span style="color:#6b7280;">Target 2</span> &nbsp;&nbsp; actual_entry plus 4.0 x ATR(14)<br>'
        '<span style="color:#6b7280;">Timeout</span> &nbsp;&nbsp; 48 trading days<br>'
        '<span style="color:#6b7280;">Short Filter</span> &nbsp;&nbsp; price must be below MA200<br>'
        '<span style="color:#6b7280;">Timeframe</span> &nbsp;&nbsp; Daily'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">Backtest Results (2015-2024)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-card" style="font-size:0.78rem;line-height:1.9;color:#e8eaf0;">'
        '<span style="color:#6b7280;">IS (2015-2021)</span> &nbsp;&nbsp; 4,592 trades | Win Rate 86.9% | Avg R 1.063<br>'
        '<span style="color:#6b7280;">OOS (2022-2024)</span> &nbsp;&nbsp; 2,429 trades | Win Rate 83.7% | Avg R 1.020<br>'
        '<span style="color:#6b7280;">Degradation</span> &nbsp;&nbsp; 3.2% win rate | 0.043R -- within normal range'
        '</div>',
        unsafe_allow_html=True
    )
    st.markdown('<div class="section-title">Position Sizing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="metric-card" style="font-size:0.78rem;line-height:1.9;color:#e8eaf0;">'
        '<span style="color:#6b7280;">Risk per trade</span> &nbsp;&nbsp; 1% of $10,000 capital<br>'
        '<span style="color:#6b7280;">Position size</span> &nbsp;&nbsp; (capital x 1%) / (actual_entry - actual_stop)<br>'
        '<span style="color:#6b7280;">Max exposure</span> &nbsp;&nbsp; 10% portfolio (~10 trades max)'
        '</div>',
        unsafe_allow_html=True
    )
