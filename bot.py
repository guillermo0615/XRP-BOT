import os
import time
import json
import hmac
import base64
import hashlib
import urllib.parse
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ── Credentials & config ────────────────────────────────────────────────────
KRAKEN_API_KEY     = os.environ["KRAKEN_API_KEY"]
KRAKEN_PRIVATE_KEY = os.environ["KRAKEN_PRIVATE_KEY"]
PAIR               = "XXRPZUSD"
INTERVAL_SEC       = int(os.environ.get("INTERVAL_SEC", "900"))

TRADE_AMOUNT_MR = float(os.environ.get("TRADE_AMOUNT_MR", "8"))
TRADE_AMOUNT_TF = float(os.environ.get("TRADE_AMOUNT_TF", "7"))
TRADE_AMOUNT_SS = float(os.environ.get("TRADE_AMOUNT_SS", "5"))
TRADE_AMOUNT_BM = float(os.environ.get("TRADE_AMOUNT_BM", "6"))

# ── Shared indicator settings ───────────────────────────────────────────────
ATR_PERIOD    = 14
VOLUME_PERIOD = 20
ADX_PERIOD    = 14
ADX_TREND     = 25

# ── Mean Reversion (MR) ─────────────────────────────────────────────────────
BB_PERIOD      = 20
BB_STD         = 2.0
RSI_PERIOD     = 14
RSI_ENTRY_MR   = 35
RSI_EXIT_MR    = 65
RSI_1H_LONG_MIN  = 40
RSI_1H_SHORT_MAX = 60
ATR_STOP_MR    = 1.5
ATR_RATIO_MAX  = 1.5
TIME_EXIT_MR   = 8
COOLDOWN_MR    = 4
FNG_MR_MIN     = 20
FNG_MR_MAX     = 75

# ── Trend Following (TF) ────────────────────────────────────────────────────
EMA_FAST        = 9
EMA_SLOW        = 21
ATR_STOP_TF     = 2.0
ATR_STOP_TF_SH  = 1.5
ATR_TARGET_TF   = 3.5
ATR_TRAIL_MULT  = 1.5
ATR_TRAIL_LOCK  = 0.5
RSI_TF_MIN      = 35
RSI_TF_MAX      = 65
COOLDOWN_TF     = 4
FNG_TF_MIN      = 40
FNG_TF_MAX_SH   = 60

# ── Sentiment Strategy (SS) ─────────────────────────────────────────────────
FNG_SS_BUY      = 15
FNG_SS_SHORT    = 85
SS_TARGET_PCT   = 0.03
SS_STOP_PCT     = 0.015
TIME_EXIT_SS    = 48

# ── Breakout Momentum (BM) ──────────────────────────────────────────────────
BM_ADX_MAX      = 20
BM_ADX_BARS     = 8
BM_ATR_RATIO    = 0.8
BM_VOL_MULT     = 2.0
BM_ATR_STOP     = 3.0
BM_ATR_TRAIL    = 2.0
FNG_BM_MIN      = 10

KRAKEN_BASE = "https://api.kraken.com"

# ── Kraken HMAC-SHA512 auth ─────────────────────────────────────────────────
def kraken_sign(urlpath, data):
    post_data = urllib.parse.urlencode(data)
    encoded   = (str(data["nonce"]) + post_data).encode()
    message   = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac       = hmac.new(base64.b64decode(KRAKEN_PRIVATE_KEY), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def kraken_post(urlpath, data):
    data["nonce"] = str(int(time.time() * 1000))
    headers = {
        "API-Key":  KRAKEN_API_KEY,
        "API-Sign": kraken_sign(urlpath, data),
    }
    r = requests.post(KRAKEN_BASE + urlpath, headers=headers, data=data, timeout=10)
    r.raise_for_status()
    result = r.json()
    if result.get("error"):
        raise ValueError(f"Kraken API error: {result['error']}")
    return result["result"]

def kraken_get(urlpath, params=None):
    r = requests.get(KRAKEN_BASE + urlpath, params=params, timeout=10)
    r.raise_for_status()
    result = r.json()
    if result.get("error"):
        raise ValueError(f"Kraken API error: {result['error']}")
    return result["result"]

# ── Market data ─────────────────────────────────────────────────────────────
def fetch_candles(interval=15):
    data = kraken_get("/0/public/OHLC", params={"pair": "XRPUSD", "interval": interval})
    pts  = data.get("XXRPZUSD") or data.get("XRPUSD")
    if pts is None:
        raise ValueError(f"Unexpected Kraken keys: {list(data.keys())}")
    df = pd.DataFrame(pts, columns=["time","open","high","low","close","vwap","volume","count"])
    for col in ["close","high","low","volume"]:
        df[col] = df[col].astype(float)
    return df.iloc[:-1].reset_index(drop=True)

def get_price():
    data   = kraken_get("/0/public/Ticker", params={"pair": "XRPUSD"})
    ticker = data.get("XXRPZUSD") or data.get("XRPUSD")
    bid    = float(ticker["b"][0])
    ask    = float(ticker["a"][0])
    return (bid + ask) / 2

def get_balances():
    result = kraken_post("/0/private/Balance", {})
    usd    = float(result.get("ZUSD", 0))
    xrp    = float(result.get("XXRP", 0))
    return usd, xrp

def get_fear_greed():
    try:
        r     = requests.get("https://api.alternative.me/fng/?limit=2", timeout=10)
        r.raise_for_status()
        data      = r.json()
        today     = int(data["data"][0]["value"])
        yesterday = int(data["data"][1]["value"])
        label     = data["data"][0]["value_classification"]
        return today, yesterday, label
    except Exception as e:
        log(f"Fear/Greed API failed: {e} -- defaulting to neutral")
        return 50, 50, "Neutral"

# ── Order execution ─────────────────────────────────────────────────────────
def place_order(side, volume, leverage=None):
    data = {
        "pair":      "XRPUSD",
        "type":      side,
        "ordertype": "market",
        "volume":    str(round(volume, 6)),
    }
    if leverage:
        data["leverage"] = str(leverage)
    result = kraken_post("/0/private/AddOrder", data)
    log(f"ORDER {side.upper()} {volume:.6f} XRP leverage={leverage} txid={result.get('txid')}")
    return result

def buy_spot(usd_amount, tag=""):
    price  = get_price()
    volume = usd_amount / price
    log(f"BUY SPOT ${usd_amount} -> {volume:.6f} XRP {tag}")
    return place_order("buy", volume)

def sell_spot(volume, tag=""):
    log(f"SELL SPOT {volume:.6f} XRP {tag}")
    return place_order("sell", volume)

def sell_short(usd_amount, tag=""):
    price  = get_price()
    volume = usd_amount / price
    log(f"SELL SHORT ${usd_amount} -> {volume:.6f} XRP {tag}")
    return place_order("sell", volume, leverage=2)

def buy_cover(volume, tag=""):
    log(f"BUY COVER {volume:.6f} XRP {tag}")
    return place_order("buy", volume, leverage=2)

# ── Indicator math ──────────────────────────────────────────────────────────
def calc_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + g / l))

def calc_macd(s, fast=12, slow=26, sig=9):
    line   = calc_ema(s, fast) - calc_ema(s, slow)
    signal = calc_ema(line, sig)
    return line, signal

def calc_atr(df, n=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def calc_adx(df, n=14):
    high, low = df["high"], df["low"]
    plus_dm   = high.diff().clip(lower=0)
    minus_dm  = (-low.diff()).clip(lower=0)
    overlap   = (high.diff() < -low.diff()) | (high.diff() < 0)
    plus_dm[overlap] = 0
    minus_dm[~overlap & (high.diff() > -low.diff())] = 0
    atr_s     = calc_atr(df, n)
    plus_di   = 100 * plus_dm.ewm(span=n, adjust=False).mean() / atr_s
    minus_di  = 100 * minus_dm.ewm(span=n, adjust=False).mean() / atr_s
    dx        = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=n, adjust=False).mean()

def compute(df):
    close          = df["close"]
    df["rsi"]      = calc_rsi(close, RSI_PERIOD)
    df["atr"]      = calc_atr(df, ATR_PERIOD)
    df["atr_avg"]  = df["atr"].rolling(BB_PERIOD).mean()
    df["adx"]      = calc_adx(df, ADX_PERIOD)
    df["vol_avg"]  = df["volume"].rolling(VOLUME_PERIOD).mean()
    df["bb_mid"]   = close.rolling(BB_PERIOD).mean()
    df["bb_std"]   = close.rolling(BB_PERIOD).std()
    df["bb_lower"] = df["bb_mid"] - BB_STD * df["bb_std"]
    df["bb_upper"] = df["bb_mid"] + BB_STD * df["bb_std"]
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_w_avg"] = df["bb_width"].rolling(BB_PERIOD).mean()
    df["ema9"]     = calc_ema(close, EMA_FAST)
    df["ema21"]    = calc_ema(close, EMA_SLOW)
    df["macd"], df["macd_sig"] = calc_macd(close)
    df["high20"]   = df["high"].rolling(20).max()
    df["adx_low"]  = (df["adx"] < BM_ADX_MAX).rolling(BM_ADX_BARS).sum()
    return df

# ── Position state ──────────────────────────────────────────────────────────
def empty_pos():
    return {
        "active":       False,
        "direction":    "",
        "entry":        0.0,
        "qty":          0.0,
        "stop":         0.0,
        "target":       0.0,
        "bars_held":    0,
        "trail_active": False,
        "trail_stop":   0.0,
        "atr_at_entry": 0.0,
    }

pos_mr = empty_pos()
pos_tf = empty_pos()
pos_ss = empty_pos()
pos_bm = empty_pos()

cooldown_mr = 0
cooldown_tf = 0

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ── MR management ───────────────────────────────────────────────────────────
def open_mr_long(price, atr_val, bb_mid):
    qty = TRADE_AMOUNT_MR / price
    pos_mr.update({
        "active": True, "direction": "long",
        "entry": price, "qty": qty,
        "stop":   price - (ATR_STOP_MR * atr_val),
        "target": bb_mid,
        "bars_held": 0, "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_val,
    })
    log(f"[MR] LONG OPEN | Entry={price:.4f} | SL={pos_mr['stop']:.4f} | TP={pos_mr['target']:.4f} | Size=${TRADE_AMOUNT_MR}")

def open_mr_short(price, atr_val, bb_mid):
    qty = TRADE_AMOUNT_MR / price
    pos_mr.update({
        "active": True, "direction": "short",
        "entry": price, "qty": qty,
        "stop":   price + (ATR_STOP_MR * atr_val),
        "target": bb_mid,
        "bars_held": 0, "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_val,
    })
    log(f"[MR] SHORT OPEN | Entry={price:.4f} | SL={pos_mr['stop']:.4f} | TP={pos_mr['target']:.4f} | Size=${TRADE_AMOUNT_MR}")

def manage_mr(price):
    global cooldown_mr
    pos_mr["bars_held"] += 1
    d   = pos_mr["direction"]
    pct = ((price - pos_mr["entry"]) / pos_mr["entry"]) * 100
    if d == "short":
        pct = -pct
    hit_stop   = (d == "long"  and price <= pos_mr["stop"]) or (d == "short" and price >= pos_mr["stop"])
    hit_target = (d == "long"  and price >= pos_mr["target"]) or (d == "short" and price <= pos_mr["target"])
    if hit_stop:
        sell_spot(pos_mr["qty"], "[MR-L]") if d == "long" else buy_cover(pos_mr["qty"], "[MR-S]")
        log(f"[MR] CLOSED {d.upper()} - Stop loss | PnL={pct:+.2f}%")
        pos_mr["active"] = False
        cooldown_mr = COOLDOWN_MR
    elif hit_target:
        sell_spot(pos_mr["qty"], "[MR-L]") if d == "long" else buy_cover(pos_mr["qty"], "[MR-S]")
        log(f"[MR] CLOSED {d.upper()} - Take profit | PnL={pct:+.2f}%")
        pos_mr["active"] = False
    elif pos_mr["bars_held"] >= TIME_EXIT_MR:
        sell_spot(pos_mr["qty"], "[MR-L]") if d == "long" else buy_cover(pos_mr["qty"], "[MR-S]")
        log(f"[MR] CLOSED {d.upper()} - Time exit | PnL={pct:+.2f}%")
        pos_mr["active"] = False
    else:
        log(f"[MR] Holding {d.upper()} | Bar {pos_mr['bars_held']}/{TIME_EXIT_MR} | PnL={pct:+.2f}% | SL={pos_mr['stop']:.4f} | TP={pos_mr['target']:.4f}")

# ── TF management ───────────────────────────────────────────────────────────
def open_tf_long(price, atr_val):
    qty = TRADE_AMOUNT_TF / price
    pos_tf.update({
        "active": True, "direction": "long",
        "entry": price, "qty": qty,
        "stop":   price - (ATR_STOP_TF * atr_val),
        "target": price + (ATR_TARGET_TF * atr_val),
        "bars_held": 0, "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_val,
    })
    log(f"[TF] LONG OPEN | Entry={price:.4f} | SL={pos_tf['stop']:.4f} | TP={pos_tf['target']:.4f} | Size=${TRADE_AMOUNT_TF}")

def open_tf_short(price, atr_val):
    qty = TRADE_AMOUNT_TF / price
    pos_tf.update({
        "active": True, "direction": "short",
        "entry": price, "qty": qty,
        "stop":   price + (ATR_STOP_TF_SH * atr_val),
        "target": price - (ATR_TARGET_TF * atr_val),
        "bars_held": 0, "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_val,
    })
    log(f"[TF] SHORT OPEN | Entry={price:.4f} | SL={pos_tf['stop']:.4f} | TP={pos_tf['target']:.4f} | Size=${TRADE_AMOUNT_TF}")

def manage_tf(price, ema9, ema21, prev_ema9, prev_ema21, rsi_15):
    global cooldown_tf
    pos_tf["bars_held"] += 1
    d    = pos_tf["direction"]
    gain = price - pos_tf["entry"] if d == "long" else pos_tf["entry"] - price
    pct  = (gain / pos_tf["entry"]) * 100
    # Activate trailing stop once gain reaches ATR_TRAIL_MULT
    if not pos_tf["trail_active"] and gain >= ATR_TRAIL_MULT * pos_tf["atr_at_entry"]:
        pos_tf["trail_active"] = True
        lock = ATR_TRAIL_LOCK * pos_tf["atr_at_entry"]
        pos_tf["trail_stop"] = pos_tf["entry"] + lock if d == "long" else pos_tf["entry"] - lock
        log(f"[TF] TRAILING STOP activated at {pos_tf['trail_stop']:.4f}")
    # Ratchet trail stop as price moves further in our favor
    if pos_tf["trail_active"]:
        atr = pos_tf["atr_at_entry"]
        if d == "long":
            new_trail = price - (ATR_TRAIL_MULT * atr)
            if new_trail > pos_tf["trail_stop"]:
                pos_tf["trail_stop"] = new_trail
                log(f"[TF] Trail ratcheted UP to {pos_tf['trail_stop']:.4f}")
        else:
            new_trail = price + (ATR_TRAIL_MULT * atr)
            if new_trail < pos_tf["trail_stop"]:
                pos_tf["trail_stop"] = new_trail
                log(f"[TF] Trail ratcheted DOWN to {pos_tf['trail_stop']:.4f}")
    trail_hit  = pos_tf["trail_active"] and ((d == "long" and price <= pos_tf["trail_stop"]) or (d == "short" and price >= pos_tf["trail_stop"]))
    stop_hit   = (d == "long" and price <= pos_tf["stop"]) or (d == "short" and price >= pos_tf["stop"])
    target_hit = (d == "long" and price >= pos_tf["target"]) or (d == "short" and price <= pos_tf["target"])
    ema_bearish = ema9 < ema21 and prev_ema9 >= prev_ema21 and rsi_15 > 55
    ema_bullish = ema9 > ema21 and prev_ema9 <= prev_ema21 and rsi_15 < 45
    def close_tf(reason):
        if d == "long":
            sell_spot(pos_tf["qty"], "[TF-L]")
        else:
            buy_cover(pos_tf["qty"], "[TF-S]")
        log(f"[TF] CLOSED {d.upper()} - {reason} | PnL={pct:+.2f}%")
        pos_tf["active"] = False
    if trail_hit:
        close_tf("Trailing stop")
    elif stop_hit:
        close_tf("Stop loss")
        cooldown_tf = COOLDOWN_TF
    elif target_hit:
        close_tf("Take profit")
    elif d == "long" and ema_bearish:
        close_tf(f"Bearish EMA cross RSI={rsi_15:.1f}")
    elif d == "short" and ema_bullish:
        close_tf(f"Bullish EMA cross RSI={rsi_15:.1f}")
    else:
        log(f"[TF] Holding {d.upper()} | Bar {pos_tf['bars_held']} | PnL={pct:+.2f}% | Trail={'ON' if pos_tf['trail_active'] else 'OFF'} | SL={pos_tf['stop']:.4f} | TP={pos_tf['target']:.4f}")

# ── SS management ───────────────────────────────────────────────────────────
def open_ss_long(price):
    qty = TRADE_AMOUNT_SS / price
    pos_ss.update({
        "active": True, "direction": "long",
        "entry": price, "qty": qty,
        "stop":   price * (1 - SS_STOP_PCT),
        "target": price * (1 + SS_TARGET_PCT),
        "bars_held": 0, "trail_active": False, "trail_stop": 0.0, "atr_at_entry": 0.0,
    })
    log(f"[SS] LONG OPEN | Entry={price:.4f} | SL={pos_ss['stop']:.4f} | TP={pos_ss['target']:.4f} | Size=${TRADE_AMOUNT_SS}")

def open_ss_short(price):
    qty = TRADE_AMOUNT_SS / price
    pos_ss.update({
        "active": True, "direction": "short",
        "entry": price, "qty": qty,
        "stop":   price * (1 + SS_STOP_PCT),
        "target": price * (1 - SS_TARGET_PCT),
        "bars_held": 0, "trail_active": False, "trail_stop": 0.0, "atr_at_entry": 0.0,
    })
    log(f"[SS] SHORT OPEN | Entry={price:.4f} | SL={pos_ss['stop']:.4f} | TP={pos_ss['target']:.4f} | Size=${TRADE_AMOUNT_SS}")

def manage_ss(price):
    pos_ss["bars_held"] += 1
    d   = pos_ss["direction"]
    pct = ((price - pos_ss["entry"]) / pos_ss["entry"]) * 100
    if d == "short":
        pct = -pct
    stop_hit   = (d == "long" and price <= pos_ss["stop"]) or (d == "short" and price >= pos_ss["stop"])
    target_hit = (d == "long" and price >= pos_ss["target"]) or (d == "short" and price <= pos_ss["target"])
    def close_ss(reason):
        if d == "long":
            sell_spot(pos_ss["qty"], "[SS-L]")
        else:
            buy_cover(pos_ss["qty"], "[SS-S]")
        log(f"[SS] CLOSED {d.upper()} - {reason} | PnL={pct:+.2f}%")
        pos_ss["active"] = False
    if stop_hit:
        close_ss("Stop loss")
    elif target_hit:
        close_ss("Take profit")
    elif pos_ss["bars_held"] >= TIME_EXIT_SS:
        close_ss(f"Time exit {TIME_EXIT_SS} bars")
    else:
        log(f"[SS] Holding {d.upper()} | Bar {pos_ss['bars_held']}/{TIME_EXIT_SS} | PnL={pct:+.2f}% | SL={pos_ss['stop']:.4f} | TP={pos_ss['target']:.4f}")

# ── BM management ───────────────────────────────────────────────────────────
def open_bm_long(price, atr_val):
    qty = TRADE_AMOUNT_BM / price
    pos_bm.update({
        "active": True, "direction": "long",
        "entry":      price,
        "qty":        qty,
        "stop":       price - (BM_ATR_STOP * atr_val),
        "target":     0.0,
        "trail_stop": price - (BM_ATR_TRAIL * atr_val),
        "trail_active": True,
        "bars_held":  0,
        "atr_at_entry": atr_val,
    })
    log(f"[BM] LONG OPEN | Entry={price:.4f} | SL={pos_bm['stop']:.4f} | Trail={pos_bm['trail_stop']:.4f} | Size=${TRADE_AMOUNT_BM}")

def manage_bm(price):
    pos_bm["bars_held"] += 1
    pct       = ((price - pos_bm["entry"]) / pos_bm["entry"]) * 100
    new_trail = price - (BM_ATR_TRAIL * pos_bm["atr_at_entry"])
    if new_trail > pos_bm["trail_stop"]:
        pos_bm["trail_stop"] = new_trail
    stop_hit  = price <= pos_bm["stop"]
    trail_hit = price <= pos_bm["trail_stop"] and pos_bm["bars_held"] > 3
    def close_bm(reason):
        sell_spot(pos_bm["qty"], "[BM]")
        log(f"[BM] CLOSED - {reason} | PnL={pct:+.2f}%")
        pos_bm["active"] = False
    if stop_hit:
        close_bm("Hard stop loss")
    elif trail_hit:
        close_bm(f"Trailing stop {pos_bm['trail_stop']:.4f}")
    else:
        log(f"[BM] Holding | Bar {pos_bm['bars_held']} | PnL={pct:+.2f}% | Trail={pos_bm['trail_stop']:.4f} | SL={pos_bm['stop']:.4f}")

# ── Main loop ───────────────────────────────────────────────────────────────
def run():
    global cooldown_mr, cooldown_tf

    log("XRP 4-STRATEGY PARALLEL BOT STARTED - KRAKEN")
    log(f"MR  Mean Reversion:    ${TRADE_AMOUNT_MR}/trade | Long+Short | ADX < {ADX_TREND}")
    log(f"TF  Trend Following:   ${TRADE_AMOUNT_TF}/trade | Long+Short | ADX > {ADX_TREND}")
    log(f"SS  Sentiment:         ${TRADE_AMOUNT_SS}/trade | Long+Short | FNG extremes")
    log(f"BM  Breakout Momentum: ${TRADE_AMOUNT_BM}/trade | Long only  | Post-consolidation")
    log(f"Max simultaneous exposure: ${TRADE_AMOUNT_MR + TRADE_AMOUNT_TF + TRADE_AMOUNT_SS + TRADE_AMOUNT_BM:.2f}")

    usd, xrp = get_balances()
    log(f"Account: ${usd:.2f} USD | {xrp:.4f} XRP")

    prev_fng  = None
    sleep_time = INTERVAL_SEC

    while True:
        try:
            df15  = compute(fetch_candles(15))
            df1h  = compute(fetch_candles(60))
            last  = df15.iloc[-1]
            prev  = df15.iloc[-2]
            price = get_price()
            fng_today, fng_yesterday, fng_label = get_fear_greed()

            adx        = last["adx"]
            atr_v      = last["atr"]
            atr_avg    = last["atr_avg"]
            atr_ratio  = atr_v / atr_avg if atr_avg > 0 else 1.0
            vol        = last["volume"]
            vol_avg    = last["vol_avg"]
            vol_ok     = vol > vol_avg
            vol_spike  = vol > (vol_avg * BM_VOL_MULT)
            rsi_1h     = df1h.iloc[-1]["rsi"]
            rsi_15     = last["rsi"]
            ema9_1h    = df1h.iloc[-1]["ema9"]
            regime     = "RANGING" if adx < ADX_TREND else "TRENDING"

            bb_lower   = last["bb_lower"]
            bb_upper   = last["bb_upper"]
            bb_mid     = last["bb_mid"]
            bb_width   = last["bb_width"]
            bb_w_avg   = last["bb_w_avg"]
            squeeze    = bb_width < bb_w_avg
            atr_ok     = atr_ratio < ATR_RATIO_MAX

            mr_oversold   = price <= bb_lower and rsi_15 < RSI_ENTRY_MR
            mr_overbought = price >= bb_upper and rsi_15 > RSI_EXIT_MR

            ema9       = last["ema9"]
            ema21      = last["ema21"]
            macd_v     = last["macd"]
            macd_sig_v = last["macd_sig"]
            prev_ema9  = prev["ema9"]
            prev_ema21 = prev["ema21"]
            prev_macd  = prev["macd"]
            prev_msig  = prev["macd_sig"]
            ema_bull   = ema9 > ema21 and prev_ema9 <= prev_ema21
            ema_bear   = ema9 < ema21 and prev_ema9 >= prev_ema21
            macd_bull  = macd_v > macd_sig_v and prev_macd <= prev_msig
            macd_bear  = macd_v < macd_sig_v and prev_macd >= prev_msig
            rsi_ok_tf  = RSI_TF_MIN <= rsi_15 <= RSI_TF_MAX

            fng_cross_fear  = prev_fng is not None and prev_fng >= FNG_SS_BUY   and fng_today < FNG_SS_BUY
            fng_cross_greed = prev_fng is not None and prev_fng <= FNG_SS_SHORT and fng_today > FNG_SS_SHORT

            consolidating  = last["adx_low"] >= BM_ADX_BARS
            atr_compressed = atr_ratio < BM_ATR_RATIO
            high20         = last["high20"]
            bm_breakout    = price > high20 and vol_spike and consolidating and atr_compressed

            any_short = (
                (pos_mr["active"] and pos_mr["direction"] == "short") or
                (pos_tf["active"] and pos_tf["direction"] == "short") or
                (pos_ss["active"] and pos_ss["direction"] == "short")
            )
            sleep_time = 300 if any_short else INTERVAL_SEC

            log(
                f"P={price:.4f} | {regime} (ADX={adx:.1f}) | "
                f"RSI15={rsi_15:.1f} RSI1H={rsi_1h:.1f} | "
                f"FNG={fng_today} ({fng_label}) | "
                f"ATR_ratio={atr_ratio:.2f} | Vol={'OK' if vol_ok else 'LOW'} | "
                f"ShortMode={'YES' if any_short else 'NO'} | "
                f"MR={'L' if pos_mr['active'] and pos_mr['direction']=='long' else 'S' if pos_mr['active'] else '-'} "
                f"TF={'L' if pos_tf['active'] and pos_tf['direction']=='long' else 'S' if pos_tf['active'] else '-'} "
                f"SS={'L' if pos_ss['active'] and pos_ss['direction']=='long' else 'S' if pos_ss['active'] else '-'} "
                f"BM={'L' if pos_bm['active'] else '-'}"
            )

            if cooldown_mr > 0:
                cooldown_mr -= 1
                log(f"[MR] Cooldown: {cooldown_mr} bars remaining")
            if cooldown_tf > 0:
                cooldown_tf -= 1
                log(f"[TF] Cooldown: {cooldown_tf} bars remaining")

            # ── STRATEGY 1: Mean Reversion ──────────────────────────────────
            if pos_mr["active"]:
                manage_mr(price)
            elif cooldown_mr == 0 and regime == "RANGING":
                log(f"[MR] BB_LOW={bb_lower:.4f} BB_HI={bb_upper:.4f} | RSI={rsi_15:.1f} | FNG={fng_today}")
                if not atr_ok:
                    log(f"[MR] SKIP - ATR expanding ratio={atr_ratio:.2f}")
                elif squeeze:
                    log("[MR] SKIP - BB squeeze")
                elif not vol_ok:
                    log("[MR] SKIP - Volume LOW")
                elif mr_oversold and rsi_1h > RSI_1H_LONG_MIN:
                    log(f"[MR] LONG SIGNAL | Price at lower band | RSI={rsi_15:.1f} | FNG={fng_today}")
                    buy_spot(TRADE_AMOUNT_MR, "[MR]")
                    open_mr_long(price, atr_v, bb_mid)
                elif mr_overbought and FNG_MR_MIN <= fng_today <= FNG_MR_MAX and rsi_1h < RSI_1H_SHORT_MAX:
                    log(f"[MR] SHORT SIGNAL | Price at upper band | RSI={rsi_15:.1f} | FNG={fng_today}")
                    sell_short(TRADE_AMOUNT_MR, "[MR]")
                    open_mr_short(price, atr_v, bb_mid)
                else:
                    log("[MR] Watching - no signal")
            elif cooldown_mr == 0:
                log(f"[MR] SKIP - Trending ADX={adx:.1f}")

            # ── STRATEGY 2: Trend Following ─────────────────────────────────
            if pos_tf["active"]:
                manage_tf(price, ema9, ema21, prev_ema9, prev_ema21, rsi_15)
            elif cooldown_tf == 0:
                log(f"[TF] EMA9={ema9:.4f} EMA21={ema21:.4f} | MACD={'up' if macd_v > macd_sig_v else 'dn'} | RSI={rsi_15:.1f} | FNG={fng_today}")
                if not vol_ok:
                    log("[TF] SKIP - Volume LOW")
                elif ema_bull and rsi_ok_tf and macd_bull and fng_today >= FNG_TF_MIN and regime == "TRENDING":
                    log(f"[TF] LONG SIGNAL | EMA bull cross | MACD bullish | RSI={rsi_15:.1f} | FNG={fng_today}")
                    buy_spot(TRADE_AMOUNT_TF, "[TF]")
                    open_tf_long(price, atr_v)
                elif ema_bull and regime != "TRENDING":
                    log(f"[TF] LONG SKIPPED - EMA bull cross but ADX={adx:.1f} not trending")
                elif ema_bear and rsi_ok_tf and macd_bear and fng_today <= FNG_TF_MAX_SH:
                    log(f"[TF] SHORT SIGNAL | EMA bear cross | MACD bearish | RSI={rsi_15:.1f} | FNG={fng_today} | ADX={adx:.1f}")
                    sell_short(TRADE_AMOUNT_TF, "[TF]")
                    open_tf_short(price, atr_v)
                elif ema_bull or ema_bear:
                    log(f"[TF] EMA cross seen but filters blocked | RSI={rsi_15:.1f} | FNG={fng_today}")
                else:
                    log("[TF] Watching - no EMA cross")

            # ── STRATEGY 3: Sentiment ───────────────────────────────────────
            if pos_ss["active"]:
                manage_ss(price)
            else:
                log(f"[SS] FNG={fng_today} prev={fng_yesterday} | Fear_cross={'YES' if fng_cross_fear else 'NO'} | Greed_cross={'YES' if fng_cross_greed else 'NO'}")
                if fng_cross_fear and vol_ok and price > ema9_1h:
                    log(f"[SS] LONG SIGNAL | FNG {fng_yesterday}->{fng_today} below {FNG_SS_BUY} | Above 1H EMA | Vol OK")
                    buy_spot(TRADE_AMOUNT_SS, "[SS]")
                    open_ss_long(price)
                elif fng_cross_fear and not vol_ok:
                    log(f"[SS] LONG SKIPPED - FNG cross confirmed but no volume climax")
                elif fng_cross_fear and price <= ema9_1h:
                    log(f"[SS] LONG SKIPPED - FNG cross confirmed but price below 1H EMA, still falling")
                elif fng_cross_greed and vol_ok and price < ema9_1h:
                    log(f"[SS] SHORT SIGNAL | FNG {fng_yesterday}->{fng_today} above {FNG_SS_SHORT} | Below 1H EMA | Vol OK")
                    sell_short(TRADE_AMOUNT_SS, "[SS]")
                    open_ss_short(price)
                elif fng_today < FNG_SS_BUY:
                    log(f"[SS] FNG={fng_today} in fear zone - no fresh cross, waiting")
                elif fng_today > FNG_SS_SHORT:
                    log(f"[SS] FNG={fng_today} in greed zone - no fresh cross, waiting")
                else:
                    log(f"[SS] Watching - FNG={fng_today} not at extreme")

            # ── STRATEGY 4: Breakout Momentum ───────────────────────────────
            if pos_bm["active"]:
                manage_bm(price)
            else:
                log(f"[BM] ADX={adx:.1f} | Consol={'YES' if consolidating else 'NO'} | ATR_ratio={atr_ratio:.2f} | VolSpike={'YES' if vol_spike else 'NO'} | High20={high20:.4f}")
                if fng_today < FNG_BM_MIN:
                    log(f"[BM] SKIP - FNG={fng_today} too fearful")
                elif bm_breakout:
                    log(f"[BM] LONG SIGNAL | Price {price:.4f} broke {high20:.4f} | Vol spike confirmed | ADX consolidation {BM_ADX_BARS}+ bars")
                    buy_spot(TRADE_AMOUNT_BM, "[BM]")
                    open_bm_long(price, atr_v)
                elif consolidating and atr_compressed:
                    log(f"[BM] Coiling - watching for breakout above {high20:.4f}")
                else:
                    log("[BM] Watching - no consolidation yet")

            prev_fng = fng_today

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(sleep_time)

if __name__ == "__main__":
    run()
