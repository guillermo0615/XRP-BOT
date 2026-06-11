#!/usr/bin/env python3
"""
XRP Bot Backtester
==================
Fetches real historical data from Kraken (15min + 1h) and Alternative.me (Fear & Greed)
then replays all 4 strategies with exact bot.py logic bar by bar.

Run:
    pip install pandas numpy requests
    python3 backtest.py

Output:
    - Printed report in terminal
    - backtest_trades.json  — every trade as structured JSON
    - backtest_report.txt   — full report saved to file
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time
import json
import sys
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — exact copy from bot.py
# ═══════════════════════════════════════════════════════════════════════════════
ATR_PERIOD    = 14
VOLUME_PERIOD = 20
ADX_PERIOD    = 14
ADX_TREND     = 25

BB_PERIOD        = 20
BB_STD           = 2.0
RSI_PERIOD       = 14
RSI_ENTRY_MR     = 35
RSI_EXIT_MR      = 65
RSI_1H_LONG_MIN  = 40
RSI_1H_SHORT_MAX = 60
ATR_STOP_MR      = 1.5
ATR_RATIO_MAX    = 1.5
TIME_EXIT_MR     = 8
COOLDOWN_MR      = 4
FNG_MR_SHORT_MIN = 20   # short-only; longs have no FNG floor
FNG_MR_MAX       = 75

EMA_FAST       = 9
EMA_SLOW       = 21
ATR_STOP_TF    = 2.0
ATR_STOP_TF_SH = 1.5
ATR_TARGET_TF  = 3.5
ATR_TRAIL_MULT = 1.5
ATR_TRAIL_LOCK = 0.5
RSI_TF_MIN     = 35
RSI_TF_MAX     = 65
COOLDOWN_TF    = 4
FNG_TF_MIN     = 40
FNG_TF_MAX_SH  = 60

FNG_SS_BUY    = 15
FNG_SS_SHORT  = 85
SS_TARGET_PCT = 0.03
SS_STOP_PCT   = 0.015
TIME_EXIT_SS  = 48

BM_ADX_MAX  = 20
BM_ADX_BARS = 8
BM_ATR_RATIO = 0.8
BM_VOL_MULT  = 2.0
BM_ATR_STOP  = 3.0
BM_ATR_TRAIL = 2.0
FNG_BM_MIN   = 10

TRADE_AMOUNT_MR = 8.0
TRADE_AMOUNT_TF = 7.0
TRADE_AMOUNT_SS = 5.0
TRADE_AMOUNT_BM = 6.0

KRAKEN_TAKER_FEE = 0.0026   # 0.26% per side; round-trip = 0.52%

# Using Bitstamp public API — US-regulated, no geo-restrictions, real XRP/USD,
# historical data going back to 2017. Free, no API key needed.
BITSTAMP_BASE = "https://www.bitstamp.net"

# ═══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_bitstamp_page(step_sec, start_ts, limit=1000):
    """Single Bitstamp OHLC page (up to 1000 candles)."""
    params = {"step": step_sec, "limit": limit, "start": int(start_ts)}
    r = requests.get(
        f"{BITSTAMP_BASE}/api/v2/ohlc/xrpusd/",
        params=params, timeout=15
    )
    r.raise_for_status()
    return r.json()["data"]["ohlc"]   # [{timestamp, open, high, low, close, volume}]


def fetch_all_candles(interval_min, years_back=5):
    """
    Paginate Bitstamp OHLC to collect up to years_back years.
    1000 candles per page. 15-min: ~10 days/page, ~183 pages for 5 years.
    Bitstamp is US-accessible, has real XRP/USD, data back to 2017.
    """
    step_map  = {15: 900, 60: 3600}
    step_sec  = step_map.get(interval_min, interval_min * 60)
    label     = f"{interval_min}min"

    now_ts   = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=365 * years_back)).timestamp())
    current  = start_ts
    all_rows = []
    seen_ts  = set()

    print(f"\nFetching {label} candles (up to {years_back}y) from Bitstamp...")

    while current < now_ts:
        try:
            ohlc = fetch_bitstamp_page(step_sec, current)
        except Exception as e:
            print(f"\n  Fetch error ({label}): {e}")
            break

        if not ohlc:
            break

        new_count = 0
        for k in ohlc:
            ts = int(k["timestamp"])
            if ts not in seen_ts:
                seen_ts.add(ts)
                all_rows.append({
                    "time":   ts,
                    "open":   float(k["open"]),
                    "high":   float(k["high"]),
                    "low":    float(k["low"]),
                    "close":  float(k["close"]),
                    "volume": float(k["volume"]),
                })
                new_count += 1

        if new_count == 0:
            break

        latest_ts = int(ohlc[-1]["timestamp"])
        latest_dt = datetime.fromtimestamp(latest_ts).strftime("%Y-%m-%d")
        print(f"  +{new_count:4d} candles | total {len(all_rows):6d} | up to {latest_dt}", end="\r")

        current = latest_ts + step_sec
        time.sleep(0.2)

    print()
    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)

    start = datetime.fromtimestamp(df.iloc[0]["time"]).strftime("%Y-%m-%d")
    end   = datetime.fromtimestamp(df.iloc[-1]["time"]).strftime("%Y-%m-%d")
    print(f"  -> {len(df):,} candles  |  {start} -> {end}")
    return df


def fetch_fng_history():
    """
    All historical Fear & Greed from Alternative.me (free, no key needed).
    Returns Series indexed by date with fng values.
    """
    print("\nFetching Fear & Greed history (Alternative.me)...")
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=0&format=json",
            timeout=15
        )
        r.raise_for_status()
        raw  = r.json()["data"]
        raw  = list(reversed(raw))
        rows = [{"date": datetime.fromtimestamp(int(d["timestamp"]), tz=timezone.utc).date(),
                 "fng":  int(d["value"])} for d in raw]
        fng = pd.DataFrame(rows).set_index("date")["fng"]
        start = fng.index[0]; end = fng.index[-1]
        print(f"  -> {len(fng)} days  |  {start} -> {end}")
        return fng
    except Exception as e:
        print(f"  FNG fetch failed: {e}  -- using neutral FNG=50 everywhere")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS — exact copy from bot.py
# ═══════════════════════════════════════════════════════════════════════════════

def calc_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calc_rsi(s, n=14):
    d  = s.diff()
    g  = d.clip(lower=0).rolling(n).mean()
    l  = (-d.clip(upper=0)).rolling(n).mean()
    rs = g / (l + 1e-9)
    return 100 - (100 / (1 + rs))

def calc_macd(s, fast=12, slow=26, sig=9):
    line   = calc_ema(s, fast) - calc_ema(s, slow)
    signal = calc_ema(line, sig)
    return line, signal

def calc_atr(df, n=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def calc_adx(df, n=14):
    high, low  = df["high"], df["low"]
    plus_dm    = high.diff().clip(lower=0)
    minus_dm   = (-low.diff()).clip(lower=0)
    overlap    = (high.diff() < -low.diff()) | (high.diff() < 0)
    plus_dm[overlap] = 0
    minus_dm[~overlap & (high.diff() > -low.diff())] = 0
    atr_s  = calc_atr(df, n)
    alpha  = 1.0 / n    # Wilder smoothing — matches TradingView/TC2000 (Bug 2 fix)
    pdi    = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s
    mdi    = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_s
    dx     = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    return dx.ewm(alpha=alpha, adjust=False).mean()

def compute(df):
    df = df.copy()
    c  = df["close"]
    df["rsi"]      = calc_rsi(c, RSI_PERIOD)
    df["atr"]      = calc_atr(df, ATR_PERIOD)
    df["atr_avg"]  = df["atr"].rolling(BB_PERIOD).mean()
    df["adx"]      = calc_adx(df, ADX_PERIOD)
    df["vol_avg"]  = df["volume"].rolling(VOLUME_PERIOD).mean()
    df["bb_mid"]   = c.rolling(BB_PERIOD).mean()
    df["bb_std"]   = c.rolling(BB_PERIOD).std()
    df["bb_lower"] = df["bb_mid"] - BB_STD * df["bb_std"]
    df["bb_upper"] = df["bb_mid"] + BB_STD * df["bb_std"]
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_w_avg"] = df["bb_width"].rolling(BB_PERIOD).mean()
    df["ema9"]     = calc_ema(c, EMA_FAST)
    df["ema21"]    = calc_ema(c, EMA_SLOW)
    df["macd"], df["macd_sig"] = calc_macd(c)
    df["high20"]   = df["high"].rolling(20).max()
    df["adx_low"]  = (df["adx"] < BM_ADX_MAX).rolling(BM_ADX_BARS).sum()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def empty_pos():
    return {
        "active":            False,
        "direction":         "",
        "entry":             0.0,
        "qty":               0.0,
        "stop":              0.0,
        "target":            0.0,
        "bars_held":         0,
        "entry_bar":         0,
        "trail_active":      False,
        "trail_stop":        0.0,
        "atr_at_entry":      0.0,
        "fng_at_entry":      50,
        "adx_at_entry":      0.0,
        "atr_ratio_at_entry": 0.0,
    }


def run_backtest(df15, df1h, fng_series):
    """
    Bar-by-bar simulation.
    Signal detected on bar close → entry executed at NEXT bar open (realistic).
    Management (stops/targets) checked at bar close each period.
    """
    trades   = []
    pos_mr   = empty_pos()
    pos_tf   = empty_pos()
    pos_ss   = empty_pos()
    pos_bm   = empty_pos()
    cooldown_mr_until = 0   # bar index
    cooldown_tf_until = 0

    # Build 1h lookup: for each 15-min bar timestamp, find the most recent 1h bar
    df1h_ts  = df1h["time"].values.astype(int)
    df1h_idx = {ts: i for i, ts in enumerate(df1h_ts)}

    def get_1h_row(bar_ts):
        hour_ts = (bar_ts // 3600) * 3600
        # Find nearest 1h bar at or before this timestamp
        candidates = df1h_ts[df1h_ts <= bar_ts]
        if len(candidates) == 0:
            return None
        return df1h.iloc[df1h_idx.get(candidates[-1], 0)]

    # FNG lookup by date
    def get_fng(bar_ts):
        if fng_series is None:
            return 50, 50
        dt      = datetime.fromtimestamp(bar_ts, tz=timezone.utc).date()
        prev_dt = (datetime.fromtimestamp(bar_ts, tz=timezone.utc) - timedelta(days=1)).date()
        today   = int(fng_series.get(dt, 50))
        yest    = int(fng_series.get(prev_dt, 50))
        return today, yest

    def record_trade(strategy, pos, exit_price, exit_reason, bar_idx, fng_today, adx, atr_ratio):
        entry     = pos["entry"]
        qty       = pos["qty"]
        direction = pos["direction"]
        bars      = bar_idx - pos["entry_bar"]

        raw_pct = ((exit_price - entry) / entry * 100) if direction == "long" \
                  else ((entry - exit_price) / entry * 100)

        # Round-trip Kraken taker fee on both legs
        fee_pct = KRAKEN_TAKER_FEE * 2 * 100
        net_pct = raw_pct - fee_pct

        t = {
            "strategy":           strategy,
            "direction":          direction,
            "entry_price":        round(entry, 6),
            "exit_price":         round(exit_price, 6),
            "pct_gain_gross":     round(raw_pct, 4),
            "pct_gain_net":       round(net_pct, 4),
            "exit_reason":        exit_reason,
            "bars_held":          bars,
            "hours_held":         round(bars * 15 / 60, 2),
            "fng_at_entry":       pos["fng_at_entry"],
            "adx_at_entry":       round(pos["adx_at_entry"], 2),
            "atr_ratio_at_entry": round(pos["atr_ratio_at_entry"], 3),
            "win":                net_pct > 0,
            "entry_bar":          pos["entry_bar"],
            "exit_bar":           bar_idx,
        }
        trades.append(t)
        return t

    N      = len(df15)
    WARMUP = 60   # bars for indicators to stabilize

    d15_times  = df15["time"].values.astype(int)
    d15_open   = df15["open"].values.astype(float)
    d15_close  = df15["close"].values.astype(float)
    d15_high   = df15["high"].values.astype(float)
    d15_low    = df15["low"].values.astype(float)
    d15_vol    = df15["volume"].values.astype(float)

    print(f"\nRunning backtest — {N:,} bars  "
          f"({datetime.fromtimestamp(d15_times[0]).strftime('%Y-%m-%d')} → "
          f"{datetime.fromtimestamp(d15_times[-1]).strftime('%Y-%m-%d')})")

    for i in range(WARMUP, N - 1):
        bar      = df15.iloc[i]
        prev_bar = df15.iloc[i - 1]
        bar_ts   = int(bar["time"])

        # Execution price = next bar open (realistic)
        exec_price = d15_open[i + 1]

        # Fear & Greed
        fng_today, fng_yesterday = get_fng(bar_ts)

        # 1h context
        row_1h = get_1h_row(bar_ts)
        if row_1h is None:
            continue

        # Helper: safe float
        def sf(v, default=0.0):
            return float(v) if not pd.isna(v) else default

        price     = sf(bar["close"])
        adx       = sf(bar["adx"])
        atr_v     = sf(bar["atr"])
        atr_avg   = sf(bar["atr_avg"], 1.0)
        atr_ratio = atr_v / atr_avg if atr_avg > 0 else 1.0
        vol       = sf(bar["volume"])
        vol_avg   = sf(bar["vol_avg"], 1.0)
        vol_ok    = vol > vol_avg
        vol_spike = vol > vol_avg * BM_VOL_MULT

        rsi_15  = sf(bar["rsi"],     50.0)
        rsi_1h  = sf(row_1h["rsi"], 50.0)
        ema9_1h = sf(row_1h["ema9"], price)

        bb_lower = sf(bar["bb_lower"], price)
        bb_upper = sf(bar["bb_upper"], price)
        bb_mid   = sf(bar["bb_mid"],   price)
        bb_width = sf(bar["bb_width"], 0.0)
        bb_w_avg = sf(bar["bb_w_avg"], 1.0)
        squeeze  = bb_width < bb_w_avg
        atr_ok   = atr_ratio < ATR_RATIO_MAX

        ema9        = sf(bar["ema9"],     price)
        ema21       = sf(bar["ema21"],    price)
        macd_v      = sf(bar["macd"],     0.0)
        macd_sig_v  = sf(bar["macd_sig"], 0.0)
        prev_ema9   = sf(prev_bar["ema9"],     ema9)
        prev_ema21  = sf(prev_bar["ema21"],    ema21)
        prev_macd   = sf(prev_bar["macd"],     macd_v)
        prev_msig   = sf(prev_bar["macd_sig"], macd_sig_v)

        ema_bull = ema9 > ema21 and prev_ema9 <= prev_ema21
        ema_bear = ema9 < ema21 and prev_ema9 >= prev_ema21
        macd_bull = macd_v > macd_sig_v and prev_macd <= prev_msig
        macd_bear = macd_v < macd_sig_v and prev_macd >= prev_msig
        rsi_ok_tf = RSI_TF_MIN <= rsi_15 <= RSI_TF_MAX

        regime       = "RANGING" if adx < ADX_TREND else "TRENDING"
        mr_oversold  = price <= bb_lower and rsi_15 < RSI_ENTRY_MR
        mr_overbought= price >= bb_upper and rsi_15 > RSI_EXIT_MR

        # SS cross detection uses yesterday vs today (Bug 3 fix — exact bot logic)
        fng_cross_fear  = fng_yesterday >= FNG_SS_BUY   and fng_today < FNG_SS_BUY
        fng_cross_greed = fng_yesterday <= FNG_SS_SHORT and fng_today > FNG_SS_SHORT

        high20         = sf(bar["high20"], price)
        consolidating  = sf(bar["adx_low"]) >= BM_ADX_BARS
        atr_compressed = atr_ratio < BM_ATR_RATIO
        bm_breakout    = price > high20 and vol_spike and consolidating and atr_compressed

        # ── STRATEGY 1: Mean Reversion ────────────────────────────────────────
        if pos_mr["active"]:
            d         = pos_mr["direction"]
            bars_held = i - pos_mr["entry_bar"]
            mgmt_p    = price

            stop_hit   = (d == "long"  and mgmt_p <= pos_mr["stop"])  or \
                         (d == "short" and mgmt_p >= pos_mr["stop"])
            target_hit = (d == "long"  and mgmt_p >= pos_mr["target"]) or \
                         (d == "short" and mgmt_p <= pos_mr["target"])
            time_exit  = bars_held >= TIME_EXIT_MR

            if stop_hit:
                record_trade("MR", pos_mr, mgmt_p, "stop_loss", i, fng_today, adx, atr_ratio)
                pos_mr = empty_pos()
                cooldown_mr_until = i + COOLDOWN_MR
            elif target_hit:
                record_trade("MR", pos_mr, mgmt_p, "take_profit", i, fng_today, adx, atr_ratio)
                pos_mr = empty_pos()
            elif time_exit:
                record_trade("MR", pos_mr, mgmt_p, "time_exit", i, fng_today, adx, atr_ratio)
                pos_mr = empty_pos()
                cooldown_mr_until = i + COOLDOWN_MR   # Bug 7 fix

        elif i >= cooldown_mr_until and regime == "RANGING":
            if atr_ok and not squeeze and vol_ok:
                if mr_oversold and rsi_1h > RSI_1H_LONG_MIN:
                    pos_mr.update({
                        "active": True, "direction": "long",
                        "entry": exec_price, "qty": TRADE_AMOUNT_MR / exec_price,
                        "stop":   exec_price - ATR_STOP_MR * atr_v,
                        "target": bb_mid,   # fixed at entry bb_mid
                        "bars_held": 0, "entry_bar": i + 1,
                        "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_v,
                        "fng_at_entry": fng_today, "adx_at_entry": adx,
                        "atr_ratio_at_entry": atr_ratio,
                    })
                elif mr_overbought and FNG_MR_SHORT_MIN <= fng_today <= FNG_MR_MAX \
                        and rsi_1h < RSI_1H_SHORT_MAX:
                    pos_mr.update({
                        "active": True, "direction": "short",
                        "entry": exec_price, "qty": TRADE_AMOUNT_MR / exec_price,
                        "stop":   exec_price + ATR_STOP_MR * atr_v,
                        "target": bb_mid,
                        "bars_held": 0, "entry_bar": i + 1,
                        "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_v,
                        "fng_at_entry": fng_today, "adx_at_entry": adx,
                        "atr_ratio_at_entry": atr_ratio,
                    })

        # ── STRATEGY 2: Trend Following ───────────────────────────────────────
        if pos_tf["active"]:
            d         = pos_tf["direction"]
            bars_held = i - pos_tf["entry_bar"]
            mgmt_p    = price
            cur_atr   = atr_v   # live ATR for ratchet (Bug 4 fix)

            gain = (mgmt_p - pos_tf["entry"]) if d == "long" \
                   else (pos_tf["entry"] - mgmt_p)

            # Activate trail (threshold uses entry ATR)
            if not pos_tf["trail_active"] \
                    and gain >= ATR_TRAIL_MULT * pos_tf["atr_at_entry"]:
                pos_tf["trail_active"] = True
                lock = ATR_TRAIL_LOCK * pos_tf["atr_at_entry"]
                pos_tf["trail_stop"] = (pos_tf["entry"] + lock) if d == "long" \
                                       else (pos_tf["entry"] - lock)

            # Ratchet uses current ATR each bar (Bug 4 fix)
            if pos_tf["trail_active"]:
                if d == "long":
                    new_trail = mgmt_p - ATR_TRAIL_MULT * cur_atr
                    if new_trail > pos_tf["trail_stop"]:
                        pos_tf["trail_stop"] = new_trail
                else:
                    new_trail = mgmt_p + ATR_TRAIL_MULT * cur_atr
                    if new_trail < pos_tf["trail_stop"]:
                        pos_tf["trail_stop"] = new_trail

            trail_hit = pos_tf["trail_active"] and (
                (d == "long"  and mgmt_p <= pos_tf["trail_stop"]) or
                (d == "short" and mgmt_p >= pos_tf["trail_stop"])
            )
            stop_hit   = (d == "long"  and mgmt_p <= pos_tf["stop"])   or \
                         (d == "short" and mgmt_p >= pos_tf["stop"])
            target_hit = (d == "long"  and mgmt_p >= pos_tf["target"]) or \
                         (d == "short" and mgmt_p <= pos_tf["target"])
            ema_bearish = ema9 < ema21 and prev_ema9 >= prev_ema21 and rsi_15 > 55
            ema_bullish = ema9 > ema21 and prev_ema9 <= prev_ema21 and rsi_15 < 45

            if trail_hit:
                record_trade("TF", pos_tf, mgmt_p, "trailing_stop", i, fng_today, adx, atr_ratio)
                pos_tf = empty_pos()
                cooldown_tf_until = i + COOLDOWN_TF   # Bug 5 fix
            elif stop_hit:
                record_trade("TF", pos_tf, mgmt_p, "stop_loss", i, fng_today, adx, atr_ratio)
                pos_tf = empty_pos()
                cooldown_tf_until = i + COOLDOWN_TF
            elif target_hit:
                record_trade("TF", pos_tf, mgmt_p, "take_profit", i, fng_today, adx, atr_ratio)
                pos_tf = empty_pos()
            elif d == "long" and ema_bearish:
                record_trade("TF", pos_tf, mgmt_p, "ema_cross_bearish", i, fng_today, adx, atr_ratio)
                pos_tf = empty_pos()
            elif d == "short" and ema_bullish:
                record_trade("TF", pos_tf, mgmt_p, "ema_cross_bullish", i, fng_today, adx, atr_ratio)
                pos_tf = empty_pos()

        elif i >= cooldown_tf_until and vol_ok:
            if ema_bull and rsi_ok_tf and macd_bull \
                    and fng_today >= FNG_TF_MIN and regime == "TRENDING":
                pos_tf.update({
                    "active": True, "direction": "long",
                    "entry": exec_price, "qty": TRADE_AMOUNT_TF / exec_price,
                    "stop":   exec_price - ATR_STOP_TF * atr_v,
                    "target": exec_price + ATR_TARGET_TF * atr_v,
                    "bars_held": 0, "entry_bar": i + 1,
                    "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_v,
                    "fng_at_entry": fng_today, "adx_at_entry": adx,
                    "atr_ratio_at_entry": atr_ratio,
                })
            elif ema_bear and rsi_ok_tf and macd_bear and fng_today <= FNG_TF_MAX_SH:
                pos_tf.update({
                    "active": True, "direction": "short",
                    "entry": exec_price, "qty": TRADE_AMOUNT_TF / exec_price,
                    "stop":   exec_price + ATR_STOP_TF_SH * atr_v,
                    "target": exec_price - ATR_TARGET_TF * atr_v,
                    "bars_held": 0, "entry_bar": i + 1,
                    "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_v,
                    "fng_at_entry": fng_today, "adx_at_entry": adx,
                    "atr_ratio_at_entry": atr_ratio,
                })

        # ── STRATEGY 3: Sentiment ─────────────────────────────────────────────
        if pos_ss["active"]:
            d         = pos_ss["direction"]
            bars_held = i - pos_ss["entry_bar"]
            mgmt_p    = price

            pct = ((mgmt_p - pos_ss["entry"]) / pos_ss["entry"]) * 100
            if d == "short":
                pct = -pct

            stop_hit   = (d == "long"  and mgmt_p <= pos_ss["stop"])   or \
                         (d == "short" and mgmt_p >= pos_ss["stop"])
            target_hit = (d == "long"  and mgmt_p >= pos_ss["target"]) or \
                         (d == "short" and mgmt_p <= pos_ss["target"])
            time_exit  = bars_held >= TIME_EXIT_SS

            if stop_hit:
                record_trade("SS", pos_ss, mgmt_p, "stop_loss", i, fng_today, adx, atr_ratio)
                pos_ss = empty_pos()
            elif target_hit:
                record_trade("SS", pos_ss, mgmt_p, "take_profit", i, fng_today, adx, atr_ratio)
                pos_ss = empty_pos()
            elif time_exit:
                record_trade("SS", pos_ss, mgmt_p, "time_exit", i, fng_today, adx, atr_ratio)
                pos_ss = empty_pos()

        else:
            if fng_cross_fear and vol_ok and price > ema9_1h:
                pos_ss.update({
                    "active": True, "direction": "long",
                    "entry": exec_price, "qty": TRADE_AMOUNT_SS / exec_price,
                    "stop":   exec_price * (1 - SS_STOP_PCT),
                    "target": exec_price * (1 + SS_TARGET_PCT),
                    "bars_held": 0, "entry_bar": i + 1,
                    "trail_active": False, "trail_stop": 0.0, "atr_at_entry": 0.0,
                    "fng_at_entry": fng_today, "adx_at_entry": adx,
                    "atr_ratio_at_entry": atr_ratio,
                })
            elif fng_cross_greed and vol_ok and price < ema9_1h:
                pos_ss.update({
                    "active": True, "direction": "short",
                    "entry": exec_price, "qty": TRADE_AMOUNT_SS / exec_price,
                    "stop":   exec_price * (1 + SS_STOP_PCT),
                    "target": exec_price * (1 - SS_TARGET_PCT),
                    "bars_held": 0, "entry_bar": i + 1,
                    "trail_active": False, "trail_stop": 0.0, "atr_at_entry": 0.0,
                    "fng_at_entry": fng_today, "adx_at_entry": adx,
                    "atr_ratio_at_entry": atr_ratio,
                })

        # ── STRATEGY 4: Breakout Momentum ─────────────────────────────────────
        if pos_bm["active"]:
            bars_held = i - pos_bm["entry_bar"]
            mgmt_p    = price

            # Ratchet only after 3-bar noise window (Bug 6 fix)
            if bars_held > 3:
                new_trail = mgmt_p - BM_ATR_TRAIL * pos_bm["atr_at_entry"]
                if new_trail > pos_bm["trail_stop"]:
                    pos_bm["trail_stop"] = new_trail

            stop_hit  = mgmt_p <= pos_bm["stop"]
            trail_hit = mgmt_p <= pos_bm["trail_stop"] and bars_held > 3

            if stop_hit:
                record_trade("BM", pos_bm, mgmt_p, "hard_stop_loss", i, fng_today, adx, atr_ratio)
                pos_bm = empty_pos()
            elif trail_hit:
                record_trade("BM", pos_bm, mgmt_p, "trailing_stop", i, fng_today, adx, atr_ratio)
                pos_bm = empty_pos()

        elif fng_today >= FNG_BM_MIN and bm_breakout:
            pos_bm.update({
                "active": True, "direction": "long",
                "entry": exec_price, "qty": TRADE_AMOUNT_BM / exec_price,
                "stop":       exec_price - BM_ATR_STOP  * atr_v,
                "trail_stop": exec_price - BM_ATR_TRAIL * atr_v,
                "target": 0.0,
                "bars_held": 0, "entry_bar": i + 1,
                "trail_active": True, "atr_at_entry": atr_v,
                "fng_at_entry": fng_today, "adx_at_entry": adx,
                "atr_ratio_at_entry": atr_ratio,
            })

    # Close any still-open positions at end of data
    final_p   = float(df15.iloc[-1]["close"])
    final_bar = N - 1
    final_fng, _ = get_fng(int(df15.iloc[-1]["time"]))
    for pos, name in [(pos_mr,"MR"), (pos_tf,"TF"), (pos_ss,"SS"), (pos_bm,"BM")]:
        if pos["active"]:
            record_trade(name, pos, final_p, "end_of_data", final_bar,
                         final_fng, 0.0, 0.0)

    print(f"Done — {len(trades)} trades executed.")
    return trades


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def analyze(trades, df15):
    lines = []
    def p(s=""):
        lines.append(s)
        print(s)

    if not trades:
        p("No trades generated — check FNG filter or data range.")
        return

    df      = pd.DataFrame(trades)
    start   = datetime.fromtimestamp(int(df15.iloc[60]["time"])).strftime("%Y-%m-%d")
    end     = datetime.fromtimestamp(int(df15.iloc[-1]["time"])).strftime("%Y-%m-%d")
    wins    = df[df["win"]]
    losses  = df[~df["win"]]
    wr      = len(wins) / len(df) * 100
    avg_win = wins["pct_gain_net"].mean() if len(wins) else 0
    avg_los = losses["pct_gain_net"].mean() if len(losses) else 0
    pf      = abs(wins["pct_gain_net"].sum() / losses["pct_gain_net"].sum()) \
              if len(losses) and losses["pct_gain_net"].sum() != 0 else float("inf")

    p("═"*64)
    p("BACKTEST RESULTS  —  XRP 4-STRATEGY BOT")
    p("═"*64)
    p(f"Period      : {start} → {end}")
    p(f"Total trades: {len(df)}")
    p(f"Fee model   : {KRAKEN_TAKER_FEE*100:.2f}% per side (round-trip {KRAKEN_TAKER_FEE*200:.2f}%)")

    p()
    p("── OVERALL ─────────────────────────────────────────────────")
    p(f"Win rate        {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    p(f"Avg win         {avg_win:+.2f}%")
    p(f"Avg loss        {avg_los:+.2f}%")
    p(f"Profit factor   {pf:.2f}")
    p(f"Sum net %       {df['pct_gain_net'].sum():+.2f}%  (not compounded)")
    p(f"Avg hold        {df['bars_held'].mean():.1f} bars  ({df['bars_held'].mean()*15/60:.1f}h)")
    est_trades_per_month = len(df) / ((int(df15.iloc[-1]["time"]) - int(df15.iloc[60]["time"])) / 86400 / 30)
    p(f"Trades/month    {est_trades_per_month:.1f} (estimated)")

    p()
    p("── BY STRATEGY ──────────────────────────────────────────────")
    for strat, amount in [("MR", TRADE_AMOUNT_MR), ("TF", TRADE_AMOUNT_TF),
                          ("SS", TRADE_AMOUNT_SS), ("BM", TRADE_AMOUNT_BM)]:
        s  = df[df["strategy"] == strat]
        if len(s) == 0:
            p(f"\n{strat} (${amount}/trade) — no trades")
            continue
        sw = s[s["win"]]; sl = s[~s["win"]]
        sr = len(sw) / len(s) * 100
        p()
        p(f"{strat}  (${amount}/trade | {len(s)} trades | {sr:.0f}% WR)")
        p(f"  Net sum    {s['pct_gain_net'].sum():+.2f}%  |  avg {s['pct_gain_net'].mean():+.2f}%")
        if len(sw) and len(sl):
            p(f"  Avg W/L   {sw['pct_gain_net'].mean():+.2f}% / {sl['pct_gain_net'].mean():+.2f}%")
        p(f"  Avg hold  {s['bars_held'].mean():.1f} bars ({s['bars_held'].mean()*15/60:.1f}h)")

        # Direction split (MR, TF, SS have both)
        if strat != "BM" and "long" in s["direction"].values and "short" in s["direction"].values:
            for dirn in ["long", "short"]:
                sub = s[s["direction"] == dirn]
                if len(sub):
                    dwr = len(sub[sub["win"]]) / len(sub) * 100
                    p(f"  {dirn.capitalize():5s}  {len(sub):3d} trades | {dwr:.0f}% WR | avg {sub['pct_gain_net'].mean():+.2f}%")

        # Exit reason breakdown
        er = s["exit_reason"].value_counts()
        parts = [f"{r}={c}" for r, c in er.items()]
        p(f"  Exits     {' | '.join(parts)}")

    # ── Open considerations answered by data ──────────────────────────────────
    p()
    p("── OPEN STRATEGY CONSIDERATIONS — DATA ANSWERS ─────────────")

    # 1. MR long win rate at different FNG levels
    mr_L = df[(df["strategy"] == "MR") & (df["direction"] == "long")]
    if len(mr_L) >= 3:
        lo = mr_L[mr_L["fng_at_entry"] < 20]
        hi = mr_L[mr_L["fng_at_entry"] >= 20]
        p()
        p("1. MR Long — win rate by FNG at entry:")
        if len(lo):
            p(f"   FNG < 20 (Extreme Fear) : {len(lo[lo['win']])}/{len(lo)} = {len(lo[lo['win']])/len(lo)*100:.0f}% WR  avg {lo['pct_gain_net'].mean():+.2f}%")
        if len(hi):
            p(f"   FNG ≥ 20               : {len(hi[hi['win']])}/{len(hi)} = {len(hi[hi['win']])/len(hi)*100:.0f}% WR  avg {hi['pct_gain_net'].mean():+.2f}%")
        if len(lo) and len(hi):
            diff = len(lo[lo['win']])/len(lo)*100 - len(hi[hi['win']])/len(hi)*100
            verdict = "Add FNG floor to MR longs" if diff < -10 else \
                      "No FNG floor needed for MR longs"
            p(f"   → Verdict: {verdict}")

    # 2. TF RSI edge entries
    tf_all = df[df["strategy"] == "TF"]
    if len(tf_all) >= 3:
        rsi_edge = tf_all[(tf_all["adx_at_entry"] > 0)]  # just check exists
        p()
        p("2. TF — overall entry quality:")
        for rng, label in [((35,40), "RSI 35-40"), ((40,60), "RSI 40-60"), ((60,65), "RSI 60-65")]:
            sub = tf_all[(tf_all["adx_at_entry"] >= 0)]  # placeholder; RSI at entry not logged
        p("   Note: RSI at entry not stored in trade log — add 'rsi_at_entry' field to bot.py")

    # 3. TF shorts — does ADX at entry matter?
    tf_sh = df[(df["strategy"] == "TF") & (df["direction"] == "short")]
    if len(tf_sh) >= 3:
        lo = tf_sh[tf_sh["adx_at_entry"] < 25]
        hi = tf_sh[tf_sh["adx_at_entry"] >= 25]
        p()
        p("3. TF Shorts — win rate by ADX at entry:")
        if len(lo):
            p(f"   ADX < 25 (no trend) : {len(lo[lo['win']])}/{len(lo)} = {len(lo[lo['win']])/len(lo)*100:.0f}% WR  avg {lo['pct_gain_net'].mean():+.2f}%")
        if len(hi):
            p(f"   ADX ≥ 25 (trending) : {len(hi[hi['win']])}/{len(hi)} = {len(hi[hi['win']])/len(hi)*100:.0f}% WR  avg {hi['pct_gain_net'].mean():+.2f}%")
        if len(lo) and len(hi):
            diff = len(lo[lo['win']])/len(lo)*100 - len(hi[hi['win']])/len(hi)*100
            verdict = "Add ADX requirement to TF shorts" if diff < -10 else \
                      "ADX requirement not needed for TF shorts"
            p(f"   → Verdict: {verdict}")

    # 4. BM fake breakout rate
    bm_all = df[df["strategy"] == "BM"]
    if len(bm_all) >= 3:
        hs = bm_all[bm_all["exit_reason"] == "hard_stop_loss"]
        p()
        p(f"4. BM Fake breakout rate: {len(hs)}/{len(bm_all)} = {len(hs)/len(bm_all)*100:.0f}% hit hard stop")
        if len(hs)/len(bm_all) > 0.5:
            p("   → High fake rate — consider raising BM_ADX_BARS to 16 or tightening volume filter")
        else:
            p("   → Acceptable fake rate — BM_ADX_BARS=8 seems OK")

    # 6. TF exit reason breakdown
    if len(tf_all) >= 3:
        p()
        p("6. TF exit reason breakdown:")
        er = tf_all["exit_reason"].value_counts()
        for reason, count in er.items():
            sub = tf_all[tf_all["exit_reason"] == reason]
            dwr = len(sub[sub["win"]]) / len(sub) * 100
            p(f"   {reason:30s}: {count:3d} trades | {dwr:.0f}% WR | avg {sub['pct_gain_net'].mean():+.2f}%")
        ema_pct = (er.get("ema_cross_bearish", 0) + er.get("ema_cross_bullish", 0)) / len(tf_all) * 100
        if ema_pct > 50:
            p("   → EMA cross exits > 50% of TF closes — may be premature; consider loosening")

    p()
    p("═"*64)

    # Save report
    report_path = "backtest_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved → {report_path}")

    # Save trades JSON (convert numpy types for JSON compatibility)
    trades_path = "backtest_trades.json"
    def json_safe(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f"Not serializable: {type(obj)}")
    with open(trades_path, "w") as f:
        json.dump(trades, f, indent=2, default=json_safe)
    print(f"Trade log  → {trades_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("XRP BOT BACKTESTER")
    print("="*64)
    print(f"Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    YEARS = 5   # how far back to pull data; reduce to 1 if Kraken limits you

    # 1. Fetch data
    df15_raw = fetch_all_candles(15, years_back=YEARS)
    if df15_raw.empty:
        print("ERROR: Could not fetch 15-min data. Check internet connection.")
        sys.exit(1)

    df1h_raw = fetch_all_candles(60, years_back=YEARS)
    if df1h_raw.empty:
        print("ERROR: Could not fetch 1h data.")
        sys.exit(1)

    fng = fetch_fng_history()

    # 2. Compute indicators
    print("\nComputing indicators...")
    df15 = compute(df15_raw)
    df1h = compute(df1h_raw)
    print(f"  15min: {len(df15):,} bars | 1h: {len(df1h):,} bars")

    # 3. Run backtest
    trades = run_backtest(df15, df1h, fng)

    # 4. Report
    analyze(trades, df15)


if __name__ == "__main__":
    main()
