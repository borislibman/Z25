"""
Z25 Strategy Validator
Streamlit app for visual validation of the ZZ Swing (Z25) strategy.

Usage:
    pip install streamlit plotly pandas pyarrow numpy
    streamlit run z25_validator.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import time, date as _date
from decimal import Decimal, ROUND_HALF_UP
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Z25 Strategy Validator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stSelectbox label, .stRadio label, .stSlider label { color: #c9d1d9; }
    .metric-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 4px 0;
    }
    .metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 16px; font-weight: 600; color: #e6edf3; }
    .win  { color: #3fb950; }
    .loss { color: #f85149; }
    .be   { color: #d29922; }
    .cc   { color: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
MIN_SWING   = 10.25
TICK        = 0.25
MULTIPLIER  = 50
ENTRY_PCT   = 0.75
STOP_PCT    = 0.25
TRADE_START = time(8, 31)
TRADE_END   = time(15, 59)
EARLY_END   = time(12, 59)

CASH_CLOSED_DATES = {
    _date(2024,1,15), _date(2024,2,19), _date(2024,3,29), _date(2024,5,27),
    _date(2024,6,19), _date(2024,9,2),  _date(2024,10,14),_date(2024,11,11),
    _date(2025,1,20), _date(2025,2,17), _date(2025,4,18), _date(2025,5,26),
    _date(2025,6,19), _date(2025,9,1),  _date(2025,10,13),_date(2025,11,11),
    _date(2026,1,19), _date(2026,2,16), _date(2026,4,3),  _date(2026,5,25),
    _date(2026,6,19), _date(2026,9,7),  _date(2026,10,12),_date(2026,11,11),
}
EARLY_CLOSE_DATES = {
    _date(2024,7,3),  _date(2024,11,29),_date(2024,12,24),
    _date(2025,7,3),  _date(2025,11,28),_date(2025,12,24),
    _date(2026,7,2),  _date(2026,11,27),_date(2026,12,24),
}


def round_to_tick(price, tick=0.25):
    d = Decimal(str(price))
    t = Decimal(str(tick))
    return float((d / t).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * t)


def finalize_trade(trade, exit_time, exit_price, mult, note):
    pts = (exit_price - trade['entry_price']) if trade['side'] == 'LONG' \
          else (trade['entry_price'] - exit_price)
    mfe = (trade['max_high'] - trade['entry_price']) if trade['side'] == 'LONG' \
          else (trade['entry_price'] - trade['min_low'])
    mae = (trade['entry_price'] - trade['min_low']) if trade['side'] == 'LONG' \
          else (trade['max_high'] - trade['entry_price'])
    trade.update({
        'exit_time':  exit_time,
        'exit_price': exit_price,
        'pnl':        round(pts * mult, 0),
        'notes':      note,
        'mae':        round(mae, 2),
        'mfe':        round(mfe, 2),
        'efficiency': round(pts / mfe if mfe > 0 else 0, 4),
    })
    return trade


# ---------------------------------------------------------------------------
# WARMUP
# ---------------------------------------------------------------------------
def warmup_zigzag(df_full, start_date):
    zz_mode      = 'LOOKING_FOR_LOW'
    zz_extreme   = None
    zz_ext_time  = None
    last_zz_high = None
    last_zz_low  = None

    warmup = df_full[df_full['datetime'] < start_date]
    for i in range(len(warmup)):
        row  = warmup.iloc[i]
        h, l = row['High'], row['Low']
        if zz_extreme is None:
            zz_extreme, zz_ext_time = l, row['datetime']
        if zz_mode == 'LOOKING_FOR_HIGH':
            if h > zz_extreme:
                zz_extreme, zz_ext_time = h, row['datetime']
            elif zz_extreme - l >= MIN_SWING:
                last_zz_high = zz_extreme
                zz_mode      = 'LOOKING_FOR_LOW'
                zz_extreme   = l
                zz_ext_time  = row['datetime']
        else:
            if l < zz_extreme:
                zz_extreme, zz_ext_time = l, row['datetime']
            elif h - zz_extreme >= MIN_SWING:
                last_zz_low = zz_extreme
                zz_mode     = 'LOOKING_FOR_HIGH'
                zz_extreme  = h
                zz_ext_time = row['datetime']

    return (zz_mode, zz_extreme, zz_ext_time, last_zz_high, last_zz_low)


# ---------------------------------------------------------------------------
# INSTRUMENTED ENGINE
# ---------------------------------------------------------------------------
def run_z25(df_full, zz_state, target_ext, start_date, end_date):
    zz_mode, zz_extreme, zz_ext_time, last_zz_high, last_zz_low = zz_state

    df = df_full[
        (df_full['datetime'] >= start_date) &
        (df_full['datetime'] <= end_date)
    ].reset_index(drop=True)

    if df.empty:
        return [], [], []

    trades, bar_states, zz_points = [], [], []

    in_trade      = False
    current_trade = {}

    setup_side   = None
    pivot_a      = None
    pa_time      = None
    pivot_b      = None
    pb_time      = None
    entry_price  = None
    stop_price   = None
    target_price = None
    swing_traded = False

    last_processed_date = None
    daily_trade_num     = 0

    for i in range(1, len(df)):
        row       = df.iloc[i]
        curr_time = row['datetime'].time()
        curr_date = row['datetime'].date()
        h, l      = row['High'], row['Low']

        if curr_date in CASH_CLOSED_DATES:
            continue

        session_end = EARLY_END if curr_date in EARLY_CLOSE_DATES else TRADE_END

        # Day reset
        if last_processed_date is not None and curr_date != last_processed_date:
            daily_trade_num = 0
            if not in_trade:
                pivot_a = pivot_b = entry_price = stop_price = target_price = None
                pa_time = pb_time = None
                setup_side = None
        last_processed_date = curr_date

        # Zigzag engine
        if zz_extreme is None:
            zz_extreme, zz_ext_time = l, row['datetime']

        zz_confirmed = None

        if zz_mode == 'LOOKING_FOR_HIGH':
            if h > zz_extreme:
                zz_extreme, zz_ext_time = h, row['datetime']
            elif zz_extreme - l >= MIN_SWING:
                last_zz_high = zz_extreme
                zz_points.append((zz_ext_time, zz_extreme, 'high'))
                zz_confirmed = ('high', zz_extreme)
                zz_mode      = 'LOOKING_FOR_LOW'
                zz_extreme   = l
                zz_ext_time  = row['datetime']
        else:
            if l < zz_extreme:
                zz_extreme, zz_ext_time = l, row['datetime']
            elif h - zz_extreme >= MIN_SWING:
                last_zz_low = zz_extreme
                zz_points.append((zz_ext_time, zz_extreme, 'low'))
                zz_confirmed = ('low', zz_extreme)
                zz_mode      = 'LOOKING_FOR_HIGH'
                zz_extreme   = h
                zz_ext_time  = row['datetime']

        # New confirmed extreme → arm fresh setup
        if zz_confirmed is not None:
            swing_traded = False
            if zz_confirmed[0] == 'LOW':
                setup_side   = 'LONG'
                pivot_a      = last_zz_low
                pa_time      = row['datetime']
                pivot_b      = None; pb_time = None
                entry_price  = None; stop_price = None; target_price = None
            else:
                setup_side   = 'SHORT'
                pivot_a      = last_zz_high
                pa_time      = row['datetime']
                pivot_b      = None; pb_time = None
                entry_price  = None; stop_price = None; target_price = None

        # Trade management
        if in_trade:
            current_trade['max_high'] = max(current_trade['max_high'], h)
            current_trade['min_low']  = min(current_trade['min_low'],  l)

            if curr_time >= session_end:
                trades.append(finalize_trade(
                    current_trade, row['datetime'],
                    round_to_tick(row['Close']), MULTIPLIER, 'cash close'))
                in_trade = False
                bar_states.append(_snap(row, setup_side, pivot_a, pa_time,
                                        pivot_b, pb_time, entry_price,
                                        stop_price, target_price, False, None, zz_confirmed))
                continue

            side = current_trade['side']
            ep   = current_trade['entry_price']

            if side == 'LONG':
                if h >= current_trade['pb_at_entry']:
                    current_trade['active_stop'] = ep
                if h >= current_trade['target']:
                    trades.append(finalize_trade(current_trade, row['datetime'],
                                                 current_trade['target'], MULTIPLIER, 'target'))
                    in_trade = False
                elif l <= current_trade['active_stop']:
                    note = 'BE' if current_trade['active_stop'] == ep else 'stop hit'
                    trades.append(finalize_trade(current_trade, row['datetime'],
                                                 current_trade['active_stop'], MULTIPLIER, note))
                    in_trade = False
            else:
                if l <= current_trade['pb_at_entry']:
                    current_trade['active_stop'] = ep
                if l <= current_trade['target']:
                    trades.append(finalize_trade(current_trade, row['datetime'],
                                                 current_trade['target'], MULTIPLIER, 'target'))
                    in_trade = False
                elif h >= current_trade['active_stop']:
                    note = 'BE' if current_trade['active_stop'] == ep else 'stop hit'
                    trades.append(finalize_trade(current_trade, row['datetime'],
                                                 current_trade['active_stop'], MULTIPLIER, note))
                    in_trade = False

            bar_states.append(_snap(row, setup_side, pivot_a, pa_time,
                                    pivot_b, pb_time, entry_price,
                                    stop_price, target_price, in_trade,
                                    current_trade if in_trade else None, zz_confirmed))
            continue

        # Setup engine
        if setup_side is None or swing_traded or pivot_a is None:
            bar_states.append(_snap(row, setup_side, pivot_a, pa_time,
                                    pivot_b, pb_time, entry_price,
                                    stop_price, target_price, False, None, zz_confirmed))
            continue

        # PB sliding expansion
        if row['datetime'] > pa_time:
            if setup_side == 'LONG':
                impulse = h - pivot_a
                if impulse >= MIN_SWING:
                    if pivot_b is None or h > pivot_b:
                        pivot_b      = h
                        pb_time      = row['datetime']
                        swing        = pivot_b - pivot_a
                        entry_price  = round_to_tick(pivot_a + ENTRY_PCT * swing)
                        stop_price   = round_to_tick(pivot_a + STOP_PCT  * swing)
                        target_price = round_to_tick(pivot_a + target_ext * swing)
            else:
                impulse = pivot_a - l
                if impulse >= MIN_SWING:
                    if pivot_b is None or l < pivot_b:
                        pivot_b      = l
                        pb_time      = row['datetime']
                        swing        = pivot_a - pivot_b
                        entry_price  = round_to_tick(pivot_a - ENTRY_PCT * swing)
                        stop_price   = round_to_tick(pivot_a - STOP_PCT  * swing)
                        target_price = round_to_tick(pivot_a - target_ext * swing)

        # Fill check
        if entry_price is None or pb_time is None or row['datetime'] <= pb_time:
            bar_states.append(_snap(row, setup_side, pivot_a, pa_time,
                                    pivot_b, pb_time, entry_price,
                                    stop_price, target_price, False, None, zz_confirmed))
            continue

        if not (TRADE_START <= curr_time < session_end):
            bar_states.append(_snap(row, setup_side, pivot_a, pa_time,
                                    pivot_b, pb_time, entry_price,
                                    stop_price, target_price, False, None, zz_confirmed))
            continue

        filled = (setup_side == 'LONG' and l <= entry_price) or \
                 (setup_side == 'SHORT' and h >= entry_price)

        if filled:
            in_trade        = True
            swing_traded    = True
            daily_trade_num += 1
            current_trade = {
                'trade_date':      curr_date,
                'day':             row['datetime'].strftime('%A'),
                'side':            setup_side,
                'daily_trade_num': daily_trade_num,
                'pa_time':         pa_time,
                'pa':              pivot_a,
                'pb_time':         pb_time,
                'pb':              pivot_b,
                'pb_at_entry':     pivot_b,
                'swing':           abs(pivot_b - pivot_a),
                'entry_time':      row['datetime'],
                'entry_price':     entry_price,
                'stop':            stop_price,
                'active_stop':     stop_price,
                'target':          target_price,
                'max_high':        h,
                'min_low':         l,
            }

        bar_states.append(_snap(row, setup_side, pivot_a, pa_time,
                                pivot_b, pb_time, entry_price,
                                stop_price, target_price, in_trade,
                                current_trade if in_trade else None, zz_confirmed))

    return trades, bar_states, zz_points


def _snap(row, setup_side, pa, pa_time, pb, pb_time,
          entry, stop, target, in_trade, trade, zz_confirmed):
    return {
        'datetime':     row['datetime'],
        'setup_side':   setup_side,
        'pivot_a':      pa,
        'pa_time':      pa_time,
        'pivot_b':      pb,
        'pb_time':      pb_time,
        'entry_price':  entry,
        'stop_price':   stop,
        'target_price': target,
        'in_trade':     in_trade,
        'trade':        trade,
        'zz_confirmed': zz_confirmed,
    }


# ---------------------------------------------------------------------------
# CHART BUILDER
# ---------------------------------------------------------------------------
def build_chart(trade, df_full, zz_points, bar_states_day, x_bars_view, y_pad_pts):
    trade_date  = trade['trade_date']
    entry_time  = trade['entry_time']
    exit_time   = trade['exit_time']
    side        = trade['side']
    entry_price = trade['entry_price']
    exit_price  = trade['exit_price']
    stop        = trade['stop']
    target      = trade['target']
    pa          = trade['pa']
    pb          = trade['pb']
    pa_time     = trade['pa_time']
    pb_time     = trade['pb_time']
    pb_at_entry = trade['pb_at_entry']

    entry_idx = df_full[df_full['datetime'] <= entry_time].index[-1]
    exit_idx  = df_full[df_full['datetime'] >= exit_time].index[0]
    trade_mid = (entry_idx + exit_idx) // 2
    ctx_half  = 150
    start_idx = max(0, trade_mid - ctx_half)
    end_idx   = min(len(df_full) - 1, trade_mid + ctx_half)
    plot_df   = df_full.iloc[start_idx:end_idx + 1].copy()

    if plot_df.empty:
        return None

    half_view = x_bars_view // 2
    x_center  = (entry_idx + exit_idx) // 2 - start_idx
    x_lo_idx  = max(0, x_center - half_view)
    x_hi_idx  = min(len(plot_df) - 1, x_center + half_view)
    x_range   = [plot_df['datetime'].iloc[x_lo_idx], plot_df['datetime'].iloc[x_hi_idx]]

    all_prices = [p for p in [target, stop, entry_price, exit_price, pa, pb] if p is not None]
    y_range = [min(all_prices) - y_pad_pts, max(all_prices) + y_pad_pts]

    trend_color = '#3fb950' if side == 'LONG' else '#f85149'

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Candlestick(
        x=plot_df['datetime'],
        open=plot_df['Open'],
        high=plot_df['High'],
        low=plot_df['Low'],
        close=plot_df['Close'],
        increasing_line_color='#3fb950',
        decreasing_line_color='#f85149',
        increasing_fillcolor='#1a3a1a',
        decreasing_fillcolor='#3a1a1a',
        line_width=1,
        name='Price',
        showlegend=False,
    ))

    x0 = plot_df['datetime'].iloc[0]
    x1 = plot_df['datetime'].iloc[-1]

    # Zigzag lines
    nearby_zz = [(dt, p, t) for dt, p, t in zz_points
                 if abs((dt.date() - trade_date).days) <= 2]
    if len(nearby_zz) >= 2:
        for k in range(len(nearby_zz) - 1):
            dt0, p0, _ = nearby_zz[k]
            dt1, p1, _ = nearby_zz[k + 1]
            fig.add_shape(type='line',
                x0=dt0, y0=p0, x1=dt1, y1=p1,
                line=dict(color='#58a6ff', width=1, dash='dot'),
                xref='x', yref='y')

    # Pivot A
    if pa and pa_time:
        fig.add_trace(go.Scatter(
            x=[pa_time], y=[pa],
            mode='markers+text',
            marker=dict(symbol='diamond', color='#a371f7', size=10),
            text=['A'],
            textposition='bottom center' if side == 'LONG' else 'top center',
            textfont=dict(color='#a371f7', size=14),
            name='Pivot A', showlegend=True,
        ))

    # Pivot B
    if pb and pb_time:
        fig.add_trace(go.Scatter(
            x=[pb_time], y=[pb],
            mode='markers+text',
            marker=dict(symbol='diamond', color='#58a6ff', size=10),
            text=['B'],
            textposition='top center' if side == 'LONG' else 'bottom center',
            textfont=dict(color='#58a6ff', size=14),
            name='Pivot B', showlegend=True,
        ))

    # PB / BE trigger line
    if pb_at_entry is not None:
        fig.add_shape(type='line',
            x0=pb_time or x0, y0=pb_at_entry, x1=x1, y1=pb_at_entry,
            line=dict(color='#d29922', width=1, dash='dot'),
            xref='x', yref='y')
        fig.add_annotation(x=x1, y=pb_at_entry,
            text=f'BE trigger (PB) {pb_at_entry:.2f}',
            showarrow=False, xanchor='right',
            font=dict(color='#d29922', size=11))

    # Entry line
    fig.add_shape(type='line',
        x0=pb_time or x0, y0=entry_price, x1=x1, y1=entry_price,
        line=dict(color='#e6edf3', width=1, dash='dash'),
        xref='x', yref='y')
    fig.add_annotation(x=x1, y=entry_price, text=f'Entry {entry_price:.2f}',
        showarrow=False, xanchor='right', font=dict(color='#e6edf3', size=12))

    # Target line
    fig.add_shape(type='line',
        x0=pb_time or x0, y0=target, x1=x1, y1=target,
        line=dict(color='#3fb950', width=1, dash='dash'),
        xref='x', yref='y')
    fig.add_annotation(x=x1, y=target, text=f'Target {target:.2f}',
        showarrow=False, xanchor='right', font=dict(color='#3fb950', size=12))

    # Stop line
    fig.add_shape(type='line',
        x0=pb_time or x0, y0=stop, x1=x1, y1=stop,
        line=dict(color='#f85149', width=1, dash='dash'),
        xref='x', yref='y')
    fig.add_annotation(x=x1, y=stop, text=f'Stop {stop:.2f}',
        showarrow=False, xanchor='right', font=dict(color='#f85149', size=12))

    # Entry marker
    entry_symbol = 'triangle-up' if side == 'LONG' else 'triangle-down'
    fig.add_trace(go.Scatter(
        x=[entry_time], y=[entry_price],
        mode='markers',
        marker=dict(symbol=entry_symbol, color=trend_color, size=14,
                    line=dict(color='white', width=1)),
        name='Entry', showlegend=True,
    ))

    # Exit marker
    note = trade.get('notes', '')
    exit_color = '#3fb950' if note == 'target' else \
                 '#f85149' if note == 'stop hit' else \
                 '#d29922' if note == 'BE' else '#58a6ff'
    exit_symbol = 'triangle-up' if side == 'SHORT' else 'triangle-down'
    fig.add_trace(go.Scatter(
        x=[exit_time], y=[exit_price],
        mode='markers',
        marker=dict(symbol=exit_symbol, color=exit_color, size=14,
                    line=dict(color='white', width=1)),
        name=f'Exit ({note})', showlegend=True,
    ))

    # In-trade shading
    fig.add_vrect(x0=entry_time, x1=exit_time,
        fillcolor=trend_color, opacity=0.05, line_width=0)

    pnl = trade.get('pnl', 0)
    pnl_str = f"+${int(pnl):,}" if pnl >= 0 else f"-${abs(int(pnl)):,}"

    fig.update_layout(
        title=dict(
            text=f"{trade_date}  |  {side}  |  {note.upper()}  |  {pnl_str}",
            font=dict(color='#e6edf3', size=14),
            x=0.01,
        ),
        plot_bgcolor='#0d1117',
        paper_bgcolor='#0d1117',
        font=dict(color='#8b949e'),
        xaxis=dict(
            rangeslider_visible=False,
            showgrid=True, gridcolor='#21262d', gridwidth=1,
            tickformat='%H:%M',
            range=x_range,
        ),
        yaxis=dict(showgrid=True, gridcolor='#21262d', gridwidth=1, range=y_range),
        legend=dict(bgcolor='#161b22', bordercolor='#30363d', borderwidth=1,
                    font=dict(size=10)),
        margin=dict(l=10, r=80, t=40, b=10),
        height=520,
    )

    return fig


# ---------------------------------------------------------------------------
# STREAMLIT APP
# ---------------------------------------------------------------------------
def main():
    st.title("📈 Z25 Strategy Validator")
    st.caption("ZZ Swing — entry 75%, stop 25%, BE at PB, targets 127.2% / 138.2%")

    # Sidebar
    with st.sidebar:
        st.header("Configuration")

        uploaded_file = st.file_uploader(
            "Upload parquet data file",
            type=["parquet"],
            help="Upload ES.750t.ATR10.Cleaned.parquet from your OneDrive"
        )

        st.divider()

        target_opt = st.radio(
            "Target extension",
            ["127.2%", "138.2%"],
            help="Extension from Pivot A: 127.2% = 1.272 × swing, 138.2% = 1.382 × swing"
        )
        target_ext = 1.272 if target_opt == "127.2%" else 1.382

        start_date = st.text_input("Start date", value="2026-01-01")
        end_date   = st.text_input("End date",   value="2026-05-31")

        st.subheader("Chart Zoom")
        x_bars_view = st.slider("X zoom (bars in view)", 20, 300, 80, 10)
        y_pad_pts   = st.slider("Y zoom (price padding, pts)", 2, 60, 15, 1)

        run_btn = st.button("▶  Run Engine", use_container_width=True, type="primary")

    # Session state init
    if 'trades' not in st.session_state:
        st.session_state.trades     = []
        st.session_state.bar_states = []
        st.session_state.zz_points  = []
        st.session_state.df_full    = None
        st.session_state.trade_idx  = 0

    if run_btn:
        if uploaded_file is None:
            st.error("Please upload a parquet file first.")
            return

        with st.spinner("Loading data and running engine…"):
            try:
                df_full = pd.read_parquet(uploaded_file)
                df_full['datetime'] = pd.to_datetime(df_full['datetime'])
                df_full = df_full.sort_values('datetime').reset_index(drop=True)

                zz_state = warmup_zigzag(df_full, start_date)

                trades, bar_states, zz_points = run_z25(
                    df_full, zz_state, target_ext, start_date, end_date)

                st.session_state.trades     = trades
                st.session_state.bar_states = bar_states
                st.session_state.zz_points  = zz_points
                st.session_state.df_full    = df_full
                st.session_state.trade_idx  = 0
                st.success(f"✓ {len(trades)} trades loaded")
            except Exception as e:
                st.error(f"Error: {e}")
                return

    trades = st.session_state.trades
    if not trades:
        st.info("Upload your parquet file, configure settings, and click **Run Engine**.")
        return

    df_full    = st.session_state.df_full
    zz_points  = st.session_state.zz_points
    bar_states = st.session_state.bar_states

    bs_by_date = {}
    for bs in bar_states:
        d = bs['datetime'].date()
        bs_by_date.setdefault(d, []).append(bs)

    # Summary stats
    df_t       = pd.DataFrame(trades)
    total_pnl  = df_t['pnl'].sum()
    n_target   = len(df_t[df_t['notes'] == 'target'])
    n_stop     = len(df_t[df_t['notes'] == 'stop hit'])
    n_be       = len(df_t[df_t['notes'] == 'BE'])
    n_cc       = len(df_t[df_t['notes'] == 'cash close'])
    win_rate   = n_target / (n_target + n_stop) if (n_target + n_stop) > 0 else 0
    gross_win  = df_t[df_t['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df_t[df_t['pnl'] < 0]['pnl'].sum())
    pf         = gross_win / gross_loss if gross_loss > 0 else float('inf')

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Trades",           len(trades))
    c2.metric("Net PnL",          f"${int(total_pnl):,}")
    c3.metric("Win Rate",         f"{win_rate:.1%}")
    c4.metric("Profit Factor",    f"{pf:.2f}")
    c5.metric("Targets / Stops",  f"{n_target} / {n_stop}")
    c6.metric("BE / CC",          f"{n_be} / {n_cc}")

    st.divider()

    # Trade navigator — filters + chained date/time selectors
    f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
    with f3:
        filter_side = st.selectbox("Filter side", ["All", "LONG", "SHORT"])
    with f4:
        filter_exit = st.selectbox("Filter exit", ["All", "target", "stop hit", "BE", "cash close"])

    filtered = [(i, t) for i, t in enumerate(trades)
                if (filter_side == "All" or t['side'] == filter_side)
                and (filter_exit == "All" or t.get('notes') == filter_exit)]

    if not filtered:
        st.warning("No trades match the current filter.")
        return

    # Date selector
    dates = sorted(set(str(t['trade_date']) for _, t in filtered))
    # Clamp stored index to valid range
    stored_idx = min(st.session_state.trade_idx, len(filtered) - 1)
    stored_date = str(filtered[stored_idx][1]['trade_date'])
    date_idx = dates.index(stored_date) if stored_date in dates else 0

    with f1:
        sel_date = st.selectbox("Date", dates, index=date_idx)

    # Time selector — only trades on selected date
    day_filtered = [(i, t) for i, t in filtered if str(t['trade_date']) == sel_date]

    def trade_time_label(t):
        ent = str(t.get('entry_time', ''))[-8:-3]
        note = t.get('notes', '').upper()
        icon = '🟢' if t.get('pnl', 0) > 0 else '🔴' if t.get('pnl', 0) < 0 else '🟡'
        return f"{ent}  {t['side']}  {note}  {icon}  ${int(t.get('pnl', 0)):+,}"

    time_labels = [trade_time_label(t) for _, t in day_filtered]

    # If the stored trade is on this date, pre-select it; otherwise default to 0
    stored_orig_idx = filtered[stored_idx][0]
    day_sel_default = next(
        (k for k, (i, _) in enumerate(day_filtered) if i == stored_orig_idx), 0)

    with f2:
        day_sel = st.selectbox("Time / trade", range(len(day_filtered)),
                               index=day_sel_default,
                               format_func=lambda x: time_labels[x])

    orig_idx, trade = day_filtered[day_sel]
    # Sync session state to absolute position in filtered list
    st.session_state.trade_idx = next(
        k for k, (i, _) in enumerate(filtered) if i == orig_idx)

    # Prev / Next step through full filtered list
    abs_sel = st.session_state.trade_idx
    b1, b2, b3 = st.columns([1, 6, 1])
    with b1:
        if st.button("◀ Prev") and abs_sel > 0:
            st.session_state.trade_idx = abs_sel - 1
            st.rerun()
    with b3:
        if st.button("Next ▶") and abs_sel < len(filtered) - 1:
            st.session_state.trade_idx = abs_sel + 1
            st.rerun()

    # Chart
    trade_date_key = trade['trade_date']
    day_bar_states = bs_by_date.get(trade_date_key, [])

    fig = build_chart(trade, df_full, zz_points, day_bar_states, x_bars_view, y_pad_pts)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not render chart for this trade.")

    # Trade detail panel
    st.divider()
    d1, d2, d3, d4 = st.columns(4)

    note       = trade.get('notes', '')
    note_color = 'win'  if note == 'target'   else \
                 'loss' if note == 'stop hit'  else \
                 'be'   if note == 'BE'        else 'cc'

    with d1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Date / Day</div>
            <div class="metric-value">{trade['trade_date']} {trade.get('day','')}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Side</div>
            <div class="metric-value">{trade['side']}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Exit Reason</div>
            <div class="metric-value {note_color}">{note.upper()}</div>
        </div>
        """, unsafe_allow_html=True)

    with d2:
        pa_t  = str(trade.get('pa_time',''))[-8:-3] if trade.get('pa_time') else 'N/A'
        pb_t  = str(trade.get('pb_time',''))[-8:-3] if trade.get('pb_time') else 'N/A'
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Pivot A</div>
            <div class="metric-value">{trade.get('pa', 0):.2f} @ {pa_t}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Pivot B (BE trigger)</div>
            <div class="metric-value">{trade.get('pb', 0):.2f} @ {pb_t}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Swing Size</div>
            <div class="metric-value">{trade.get('swing', 0):.2f} pts</div>
        </div>
        """, unsafe_allow_html=True)

    with d3:
        ent_t = str(trade.get('entry_time',''))[-8:-3] if trade.get('entry_time') else 'N/A'
        ext_t = str(trade.get('exit_time', ''))[-8:-3] if trade.get('exit_time')  else 'N/A'
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Entry</div>
            <div class="metric-value">{trade.get('entry_price', 0):.2f} @ {ent_t}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Target / Stop</div>
            <div class="metric-value">{trade.get('target', 0):.2f} / {trade.get('stop', 0):.2f}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Exit</div>
            <div class="metric-value">{trade.get('exit_price', 0):.2f} @ {ext_t}</div>
        </div>
        """, unsafe_allow_html=True)

    with d4:
        pnl       = trade.get('pnl', 0)
        pnl_class = 'win' if pnl > 0 else 'loss' if pnl < 0 else 'be'
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">PnL</div>
            <div class="metric-value {pnl_class}">${int(pnl):+,}</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">MAE / MFE</div>
            <div class="metric-value">{trade.get('mae', 0):.2f} / {trade.get('mfe', 0):.2f} pts</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Daily Trade #</div>
            <div class="metric-value">{trade.get('daily_trade_num', '—')}</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
