#!/usr/bin/env python3
"""
XRP DAILY TREND BOT - KRAKEN
============================
Single strategy. Long only. Spot only. Evaluates ONCE per day after the
00:00 UTC daily close. Replaces the 4-strategy 15-min bot entirely.

DO NOT DEPLOY until backtest_trend.py prints passing deploy gates.

Entry : yesterday's daily close > highest close of prior DON_LEN days
        AND close > SMA(SMA_FILTER)  -> market buy
Exit  : daily close < max(initial stop, chandelier trail) -> market sell
        initial stop = entry - ATR_STOP * ATR(14) at entry
        chandelier   = highest close since entry - ATR_TRAIL * ATR(14) current

Railway env vars required: KRAKEN_API_KEY, KRAKEN_PRIVATE_KEY
Optional: TRADE_NOTIONAL (default 26)
"""

import os
import time
import json
import hmac
import base64
import hashlib
import urllib.parse
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

# ── Config ───────────────────────────────────────────────────────────────────
KRAKEN_API_KEY     = os.environ["KRAKEN_API_KEY"]
KRAKEN_PRIVATE_KEY = os.environ["KRAKEN_PRIVATE_KEY"]
TRADE_NOTIONAL     = float(os.environ.get("TRADE_NOTIONAL", "26"))

DON_LEN    = 30
SMA_FILTER = 200
ATR_LEN    = 14
ATR_STOP   = 2.5
ATR_TRAIL  = 3.0

KRAKEN_BASE = "https://api.kraken.com"
STATE_PATH  = "/tmp/trend_pos.json"
WAKE_MINUTE = 3   # evaluate at 00:03 UTC, after the daily candle closes


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}Z] {msg}", flush=True)


# ── Monotonic nonce (same fix as old bot) ────────────────────────────────────
_nonce_counter = int(time.time() * 1000)

def next_nonce():
    global _nonce_counter
    _nonce_counter += 1
    return str(_nonce_counter)


# ── Kraken auth / transport ──────────────────────────────────────────────────
def kraken_sign(urlpath, data):
    post_data = urllib.parse.urlencode(data)
    encoded   = (str(data["nonce"]) + post_data).encode()
    message   = urlpath.encode() + hashlib.sha256(encoded).digest()
    mac       = hmac.new(base64.b64decode(KRAKEN_PRIVATE_KEY), message, hashlib.sha512)
    return base64.b64encode(mac.digest()).decode()

def kraken_post(urlpath, data):
    data["nonce"] = next_nonce()
    headers = {"API-Key": KRAKEN_API_KEY, "API-Sign": kraken_sign(urlpath, data)}
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


# ── Market data ──────────────────────────────────────────────────────────────
def fetch_daily():
    """~720 daily candles from Kraken — enough for SMA200 warmup."""
    data = kraken_get("/0/public/OHLC", params={"pair": "XRPUSD", "interval": 1440})
    pts  = data.get("XXRPZUSD") or data.get("XRPUSD")
    if pts is None:
        raise ValueError(f"Unexpected Kraken keys: {list(data.keys())}")
    df = pd.DataFrame(pts, columns=["time", "open", "high", "low", "close",
                                    "vwap", "volume", "count"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["time"] = df["time"].astype(int)
    return df.iloc[:-1].reset_index(drop=True)   # drop in-progress candle

def get_price():
    data   = kraken_get("/0/public/Ticker", params={"pair": "XRPUSD"})
    ticker = data.get("XXRPZUSD") or data.get("XRPUSD")
    return (float(ticker["b"][0]) + float(ticker["a"][0])) / 2

def get_balances():
    result = kraken_post("/0/private/Balance", {})
    return float(result.get("ZUSD", 0)), float(result.get("XXRP", 0))


# ── Orders ───────────────────────────────────────────────────────────────────
def query_fill(txid):
    for _ in range(5):
        res = kraken_post("/0/private/QueryOrders", {"txid": txid})
        o   = res.get(txid)
        if o and o["status"] == "closed":
            vol_exec  = float(o["vol_exec"])
            cost      = float(o.get("cost", 0))
            avg_price = (cost / vol_exec) if vol_exec > 0 and cost > 0 else float(o["price"])
            return avg_price, vol_exec
        time.sleep(1)
    raise ValueError(f"Order {txid} not closed after retries")

def place_order(side, volume):
    data = {"pair": "XRPUSD", "type": side, "ordertype": "market",
            "volume": str(round(volume, 6))}
    result = kraken_post("/0/private/AddOrder", data)
    txid   = result["txid"][0]
    fill_price, fill_qty = query_fill(txid)
    log(f"ORDER {side.upper()} filled {fill_qty:.6f} XRP @ {fill_price:.4f} txid={txid}")
    return fill_price, fill_qty


# ── Indicators ───────────────────────────────────────────────────────────────
def calc_atr(df, n=ATR_LEN):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


# ── Position state (persisted + mirrored to Railway logs) ───────────────────
def save_state(pos):
    print(f"[POSITION_RECORD] {json.dumps(pos)}", flush=True)
    try:
        with open(STATE_PATH, "w") as f:
            json.dump(pos, f)
    except Exception as e:
        log(f"WARNING: could not persist state: {e}")

def load_state():
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def log_trade(pos, exit_price, reason):
    pct = (exit_price - pos["entry"]) / pos["entry"] * 100
    record = {
        "timestamp":   datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy":    "TREND",
        "direction":   "long",
        "entry_price": round(pos["entry"], 6),
        "exit_price":  round(exit_price, 6),
        "pct_gain":    round(pct, 4),
        "exit_reason": reason,
        "days_held":   pos.get("days_held", 0),
        "win":         pct > 0,
    }
    print(f"[TRADE_RECORD] {json.dumps(record)}", flush=True)
    log(f"[TREND] CLOSED - {reason} | PnL={pct:+.2f}% (gross of fees)")


# ── Core daily evaluation ────────────────────────────────────────────────────
def evaluate(pos):
    df = fetch_daily()
    df["atr"] = calc_atr(df)
    df["sma"] = df["close"].rolling(SMA_FILTER).mean()
    df["don"] = df["close"].shift(1).rolling(DON_LEN).max()

    last     = df.iloc[-1]
    c        = float(last["close"])
    atr_v    = float(last["atr"])
    sma_v    = float(last["sma"])
    don_v    = float(last["don"])
    bar_time = int(last["time"])

    if pd.isna(sma_v) or pd.isna(don_v):
        log("Not enough history for SMA200 yet — skipping evaluation")
        return pos

    if pos is None:
        log(f"FLAT | close={c:.4f} | 30d-high={don_v:.4f} | SMA200={sma_v:.4f} | ATR={atr_v:.4f}")
        if c > don_v and c > sma_v:
            log(f"[TREND] ENTRY SIGNAL | close {c:.4f} > 30d high {don_v:.4f} and > SMA200")
            price  = get_price()
            volume = TRADE_NOTIONAL / price
            fill_price, fill_qty = place_order("buy", volume)
            pos = {
                "entry":        fill_price,
                "qty":          fill_qty,
                "stop":         fill_price - ATR_STOP * atr_v,
                "hi":           fill_price,
                "entry_time":   bar_time,
                "days_held":    0,
            }
            save_state(pos)
            log(f"[TREND] LONG OPEN | Entry={fill_price:.4f} | SL={pos['stop']:.4f} | Qty={fill_qty:.6f}")
        else:
            why = []
            if c <= don_v:
                why.append("no breakout")
            if c <= sma_v:
                why.append("below SMA200")
            log(f"[TREND] No entry ({', '.join(why)})")
        return pos

    # ── Manage open position on the daily close ──
    pos["hi"]        = max(pos["hi"], c)
    pos["days_held"] = max(0, int((bar_time - pos["entry_time"]) / 86400))
    trail = pos["hi"] - ATR_TRAIL * atr_v
    level = max(pos["stop"], trail)
    pnl   = (c - pos["entry"]) / pos["entry"] * 100
    log(f"[TREND] LONG | Day {pos['days_held']} | close={c:.4f} | PnL={pnl:+.2f}% | "
        f"exit-level={level:.4f} (stop={pos['stop']:.4f}, trail={trail:.4f})")

    if c < level:
        reason = "stop_loss" if level == pos["stop"] else "trailing_stop"
        fill_price, _ = place_order("sell", pos["qty"])
        log_trade(pos, fill_price, reason)
        pos = None
        save_state({"active": False})
    else:
        save_state(pos)
    return pos


# ── Restart reconciliation ───────────────────────────────────────────────────
def reconcile_on_boot():
    """Railway's /tmp resets on redeploy. If we hold XRP but lost state,
    adopt the position and re-anchor the trail at the current price."""
    pos = load_state()
    if pos and pos.get("entry"):
        log(f"Restored position from state file: entry={pos['entry']:.4f}")
        return pos
    usd, xrp = get_balances()
    price = get_price()
    log(f"Account: ${usd:.2f} USD | {xrp:.4f} XRP (~${xrp*price:.2f})")
    if xrp * price > 5:
        log("WARNING: holding XRP with no saved state (restart wiped /tmp).")
        log("Adopting position; trail RE-ANCHORED at current price. Entry/PnL unknown.")
        df = fetch_daily()
        df["atr"] = calc_atr(df)
        atr_v = float(df.iloc[-1]["atr"])
        pos = {
            "entry":      price,            # unknown true entry — PnL resets
            "qty":        xrp,
            "stop":       price - ATR_STOP * atr_v,
            "hi":         price,
            "entry_time": int(df.iloc[-1]["time"]),
            "days_held":  0,
        }
        save_state(pos)
        return pos
    return None


def seconds_until_next_wake():
    now = datetime.now(timezone.utc)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=WAKE_MINUTE,
                                            second=0, microsecond=0)
    return max(60, (nxt - now).total_seconds())


# ── Main ─────────────────────────────────────────────────────────────────────
def run():
    log("XRP DAILY TREND BOT STARTED - KRAKEN")
    log(f"Params: DON={DON_LEN} SMA={SMA_FILTER} STOP={ATR_STOP}xATR "
        f"TRAIL={ATR_TRAIL}xATR | ${TRADE_NOTIONAL} notional | long-only spot")
    pos = reconcile_on_boot()

    last_bar = 0
    while True:
        try:
            df_check = fetch_daily()
            newest = int(df_check.iloc[-1]["time"])
            if newest > last_bar:
                pos = evaluate(pos)
                last_bar = newest
            else:
                log("Daily candle unchanged — waiting")
        except Exception as e:
            log(f"ERROR: {e}")
        sleep_s = seconds_until_next_wake()
        log(f"Sleeping {sleep_s/3600:.1f}h until next daily close evaluation")
        time.sleep(sleep_s)


if __name__ == "__main__":
    run()
