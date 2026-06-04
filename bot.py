"""
XRP Trading Bot — Optimized Strategy
=====================================
Indicators:
  - EMA 9 / 21 crossover  → trend direction
  - RSI(14)               → momentum filter
  - MACD(12,26,9)         → trend confirmation (reduces false crossovers)
  - ATR(14)               → dynamic stop loss (adapts to XRP volatility)
  - Volume filter         → only trade when volume > 20-period average

Entry (BUY):
  - EMA9 crosses above EMA21 (bullish crossover)
  - RSI between 40–60 (momentum building, not overbought)
  - MACD line > Signal line (trend confirmed)
  - Volume > 20-bar average (real move, not low-volume noise)

Exit (SELL) — first condition that triggers:
  1. ATR stop loss: entry - (2.0 x ATR)  → dynamic, respects volatility
  2. Take profit:  entry + (3.5 x ATR)   → 1.75:1 R/R ratio
  3. Trailing stop: once price moves 1.5x ATR in our favor,
     trail stop to entry + (0.5 x ATR)  → lock in profit
  4. EMA bearish crossover + RSI > 55   → trend reversal confirmed

Timeframe: 15-minute candles
Trade size: $20 per trade (configurable via TRADE_AMOUNT env var)
"""

import os
import time
import json
import base64
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

# ── Config ──────────────────────────────────────────────────────────────────────
API_KEY_ID   = os.environ["RH_API_KEY_ID"]
PRIVATE_KEY  = os.environ["RH_PRIVATE_KEY"]
SYMBOL       = "XRP-USD"
TRADE_AMOUNT = float(os.environ.get("TRADE_AMOUNT", "20"))
INTERVAL_SEC = int(os.environ.get("INTERVAL_SEC", "900"))

# Strategy parameters
ATR_PERIOD      = 14
ATR_STOP_MULT   = 2.0    # stop loss = entry - (2.0 x ATR)
ATR_TARGET_MULT = 3.5    # take profit = entry + (3.5 x ATR)  → ~1.75:1 R/R
ATR_TRAIL_MULT  = 1.5    # activate trailing stop after 1.5x ATR gain
ATR_TRAIL_LOCK  = 0.5    # trail stop to entry + (0.5 x ATR) once activated
VOLUME_PERIOD   = 20     # volume must exceed 20-bar average

BASE_URL = "https://trading.robinhood.com"

# ── Auth ────────────────────────────────────────────────────────────────────────
def get_pkey():
    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(PRIVATE_KEY))

def sign(method, path, body=""):
    ts  = str(int(datetime.now(timezone.utc).timestamp()))
    msg = f"{API_KEY_ID}{ts}{path}{method.upper()}{body}"
    sig = base64.b64encode(get_pkey().sign(msg.encode())).decode()
    return {"x-api-key": API_KEY_ID, "x-timestamp": ts, "x-signature": sig,
            "Content-Type": "application/json; charset=utf-8"}

def rh_get(path):
    r = requests.get(BASE_URL + path, headers=sign("GET", path), timeout=10)
    r.raise_for_status()
    return r.json()

def rh_post(path, body):
    s = json.dumps(body)
    r = requests.post(BASE_URL + path, headers=sign("POST", path, s), data=s, timeout=10)
    r.raise_for_status()
    return r.json()

# ── Market Data ─────────────────────────────────────────────────────────────────
def fetch_candles():
    path = f"/api/v1/crypto/marketdata/historical/?symbol={SYMBOL}&interval=15minute&span=week"
    pts  = rh_get(path).get("data_points", [])
    if len(pts) < 50:
        raise ValueError(f"Not enough candles: {len(pts)}")
    df = pd.DataFrame(pts)
    df["close"]  = df["close_price"].astype(float)
    df["high"]   = df["high_price"].astype(float)
    df["low"]    = df["low_price"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df.reset_index(drop=True)

def get_price():
    path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={SYMBOL}"
    d    = rh_get(path)["results"][0]
    return (float(d["ask_inclusive_of_buy_spread"]) + float(d["bid_inclusive_of_sell_spread"])) / 2

# ── Indicators ──────────────────────────────────────────────────────────────────
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + g / l))

def macd(s, fast=12, slow=26, sig=9):
    line   = ema(s, fast) - ema(s, slow)
    signal = ema(line, sig)
    return line, signal

def atr(df, n=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def compute(df):
    df["ema9"]       = ema(df["close"], 9)
    df["ema21"]      = ema(df["close"], 21)
    df["rsi"]        = rsi(df["close"], 14)
    df["macd"], df["macd_sig"] = macd(df["close"])
    df["atr"]        = atr(df, ATR_PERIOD)
    df["vol_avg"]    = df["volume"].rolling(VOLUME_PERIOD).mean()
    return df

# ── Orders ──────────────────────────────────────────────────────────────────────
def get_holdings():
    for h in rh_get("/api/v1/crypto/trading/holdings/").get("results", []):
        if h["asset_code"] == "XRP":
            return float(h["total_quantity"])
    return 0.0

def buy(usd):
    r = rh_post("/api/v1/crypto/trading/orders/", {
        "symbol": SYMBOL, "side": "buy", "type": "market",
        "market_order_config": {"asset_quantity": None, "quote_amount": str(round(usd, 2))}
    })
    log(f"BUY  ${usd} → order {r.get('id')}")
    return r

def sell(qty):
    r = rh_post("/api/v1/crypto/trading/orders/", {
        "symbol": SYMBOL, "side": "sell", "type": "market",
        "market_order_config": {"asset_quantity": str(round(qty, 6)), "quote_amount": None}
    })
    log(f"SELL {qty} XRP → order {r.get('id')}")
    return r

# ── State ───────────────────────────────────────────────────────────────────────
pos = {
    "active":        False,
    "entry":         0.0,
    "qty":           0.0,
    "stop":          0.0,
    "target":        0.0,
    "trail_active":  False,
    "trail_stop":    0.0,
    "atr_at_entry":  0.0,
}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def open_position(price, qty, atr_val):
    pos.update({
        "active":       True,
        "entry":        price,
        "qty":          qty,
        "stop":         price - (ATR_STOP_MULT * atr_val),
        "target":       price + (ATR_TARGET_MULT * atr_val),
        "trail_active": False,
        "trail_stop":   0.0,
        "atr_at_entry": atr_val,
    })
    log(f"POSITION OPEN | Entry={price:.4f} | SL={pos['stop']:.4f} | TP={pos['target']:.4f} | ATR={atr_val:.4f}")

def close_position(reason):
    sell(pos["qty"])
    log(f"POSITION CLOSED — {reason}")
    pos["active"] = False

# ── Main Loop ───────────────────────────────────────────────────────────────────
def run():
    log("=" * 60)
    log("XRP BOT STARTED — Optimized 5-Indicator Strategy")
    log(f"EMA 9/21 | RSI(14) | MACD(12,26,9) | ATR(14) | Volume Filter")
    log(f"SL: {ATR_STOP_MULT}x ATR | TP: {ATR_TARGET_MULT}x ATR | Trailing: {ATR_TRAIL_MULT}x ATR")
    log("=" * 60)

    # Sync on startup
    held = get_holdings()
    if held > 0.01:
        price = get_price()
        df    = compute(fetch_candles())
        atr_v = df["atr"].iloc[-1]
        open_position(price, held, atr_v)
        log(f"Resumed existing position of {held:.4f} XRP")

    while True:
        try:
            df   = compute(fetch_candles())
            prev = df.iloc[-2]
            last = df.iloc[-1]
            price = get_price()

            # Current values
            ema9, ema21   = last["ema9"], last["ema21"]
            rsi_v         = last["rsi"]
            macd_v        = last["macd"]
            macd_sig_v    = last["macd_sig"]
            atr_v         = last["atr"]
            vol           = last["volume"]
            vol_avg       = last["vol_avg"]

            # Previous bar values (for crossover detection)
            prev_ema9, prev_ema21 = prev["ema9"], prev["ema21"]
            prev_macd, prev_macd_sig = prev["macd"], prev["macd_sig"]

            log(
                f"P={price:.4f} | EMA9={ema9:.4f} EMA21={ema21:.4f} | "
                f"RSI={rsi_v:.1f} | MACD={'↑' if macd_v > macd_sig_v else '↓'} | "
                f"ATR={atr_v:.4f} | Vol={'✓' if vol > vol_avg else '✗'} | "
                f"Pos={'YES' if pos['active'] else 'NO'}"
            )

            if pos["active"]:
                # ── Update trailing stop ──────────────────────────────
                gain = price - pos["entry"]
                if not pos["trail_active"] and gain >= ATR_TRAIL_MULT * pos["atr_at_entry"]:
                    pos["trail_active"] = True
                    pos["trail_stop"]   = pos["entry"] + (ATR_TRAIL_LOCK * pos["atr_at_entry"])
                    log(f"TRAILING STOP activated at {pos['trail_stop']:.4f}")

                # ── Exit checks ───────────────────────────────────────
                if pos["trail_active"] and price <= pos["trail_stop"]:
                    close_position(f"Trailing stop hit at {price:.4f}")

                elif price <= pos["stop"]:
                    close_position(f"ATR stop loss hit at {price:.4f}")

                elif price >= pos["target"]:
                    close_position(f"Take profit hit at {price:.4f}")

                elif (ema9 < ema21 and prev_ema9 >= prev_ema21 and rsi_v > 55):
                    close_position(f"Bearish EMA cross + RSI={rsi_v:.1f}")

            else:
                # ── Entry checks ──────────────────────────────────────
                ema_cross_up  = ema9 > ema21 and prev_ema9 <= prev_ema21
                rsi_ok        = 40 <= rsi_v <= 60
                macd_bullish  = macd_v > macd_sig_v and prev_macd <= prev_macd_sig
                volume_ok     = vol > vol_avg

                if ema_cross_up and rsi_ok and macd_bullish and volume_ok:
                    log(f"BUY SIGNAL | EMA cross ✓ RSI={rsi_v:.1f} ✓ MACD bullish ✓ Volume ✓")
                    buy(TRADE_AMOUNT)
                    open_position(price, TRADE_AMOUNT / price, atr_v)

                elif ema_cross_up:
                    # Log why we didn't take the trade
                    reasons = []
                    if not rsi_ok:       reasons.append(f"RSI={rsi_v:.1f} out of 40-60")
                    if not macd_bullish: reasons.append("MACD not confirmed")
                    if not volume_ok:    reasons.append("Volume below average")
                    log(f"EMA cross seen but SKIPPED — {', '.join(reasons)}")

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    run()
