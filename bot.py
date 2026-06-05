import os
import time
import json
import base64
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

API_KEY_ID   = os.environ["RH_API_KEY_ID"]
PRIVATE_KEY  = os.environ["RH_PRIVATE_KEY"]
SYMBOL       = "XRP-USD"
TRADE_AMOUNT = float(os.environ.get("TRADE_AMOUNT", "20"))
INTERVAL_SEC = int(os.environ.get("INTERVAL_SEC", "900"))

# Shared
ATR_PERIOD     = 14
VOLUME_PERIOD  = 20
ADX_PERIOD     = 14
ADX_TREND      = 25    # above = trending, below = ranging

# Mean Reversion (BB+RSI) -- ranging markets
BB_PERIOD      = 20
BB_STD         = 2.0
RSI_PERIOD     = 14
RSI_ENTRY_MR   = 35
RSI_1H_MIN     = 40
ATR_STOP_MR    = 1.5
ATR_RATIO_MAX  = 1.5   # skip entry if ATR expanding too fast
TIME_EXIT_BARS = 8
COOLDOWN_BARS  = 4

# Trend Following (EMA+MACD) -- trending markets
EMA_FAST       = 9
EMA_SLOW       = 21
ATR_STOP_TF    = 2.0
ATR_TARGET_TF  = 3.5
ATR_TRAIL_MULT = 1.5
ATR_TRAIL_LOCK = 0.5

RH_BASE_URL = "https://trading.robinhood.com"

# ── Auth ─────────────────────────────────────────────────────────────────────
def get_pkey():
    return Ed25519PrivateKey.from_private_bytes(base64.b64decode(PRIVATE_KEY))

def sign(method, path, body=""):
    ts  = str(int(datetime.now(timezone.utc).timestamp()))
    msg = f"{API_KEY_ID}{ts}{path}{method.upper()}{body}"
    sig = base64.b64encode(get_pkey().sign(msg.encode())).decode()
    return {
        "x-api-key":    API_KEY_ID,
        "x-timestamp":  ts,
        "x-signature":  sig,
        "Content-Type": "application/json; charset=utf-8",
    }

def rh_get(path):
    r = requests.get(RH_BASE_URL + path, headers=sign("GET", path), timeout=10)
    r.raise_for_status()
    return r.json()

def rh_post(path, body):
    s = json.dumps(body)
    r = requests.post(RH_BASE_URL + path, headers=sign("POST", path, s), data=s, timeout=10)
    r.raise_for_status()
    return r.json()

# ── Data ─────────────────────────────────────────────────────────────────────
def fetch_candles(interval=15):
    url = f"https://api.kraken.com/0/public/OHLC?pair=XRPUSD&interval={interval}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise ValueError(f"Kraken error: {data['error']}")
    pts = data["result"].get("XXRPZUSD") or data["result"].get("XRPUSD")
    if pts is None:
        raise ValueError(f"Unexpected Kraken keys: {list(data['result'].keys())}")
    df = pd.DataFrame(pts, columns=["time","open","high","low","close","vwap","volume","count"])
    for col in ["close","high","low","volume"]:
        df[col] = df[col].astype(float)
    return df.iloc[:-1].reset_index(drop=True)

def get_price():
    path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={SYMBOL}"
    d = rh_get(path)["results"][0]
    return (float(d["ask_inclusive_of_buy_spread"]) + float(d["bid_inclusive_of_sell_spread"])) / 2

# ── Indicators ───────────────────────────────────────────────────────────────
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
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    overlap  = (high.diff() < -low.diff()) | (high.diff() < 0)
    plus_dm[overlap] = 0
    minus_dm[~overlap & (high.diff() > -low.diff())] = 0
    atr_s    = calc_atr(df, n)
    plus_di  = 100 * plus_dm.ewm(span=n, adjust=False).mean() / atr_s
    minus_di = 100 * minus_dm.ewm(span=n, adjust=False).mean() / atr_s
    dx       = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9))
    adx      = dx.ewm(span=n, adjust=False).mean()
    return adx

def compute(df):
    close = df["close"]
    df["rsi"]      = calc_rsi(close, RSI_PERIOD)
    df["atr"]      = calc_atr(df, ATR_PERIOD)
    df["atr_avg"]  = df["atr"].rolling(BB_PERIOD).mean()
    df["adx"]      = calc_adx(df, ADX_PERIOD)
    df["vol_avg"]  = df["volume"].rolling(VOLUME_PERIOD).mean()
    # BB
    df["bb_mid"]   = close.rolling(BB_PERIOD).mean()
    df["bb_std"]   = close.rolling(BB_PERIOD).std()
    df["bb_lower"] = df["bb_mid"] - BB_STD * df["bb_std"]
    df["bb_upper"] = df["bb_mid"] + BB_STD * df["bb_std"]
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_w_avg"] = df["bb_width"].rolling(BB_PERIOD).mean()
    # EMA + MACD
    df["ema9"]  = calc_ema(close, EMA_FAST)
    df["ema21"] = calc_ema(close, EMA_SLOW)
    df["macd"], df["macd_sig"] = calc_macd(close)
    return df

# ── Orders ───────────────────────────────────────────────────────────────────
def get_holdings():
    for h in rh_get("/api/v1/crypto/trading/holdings/").get("results", []):
        if h["asset_code"] == "XRP":
            return float(h["total_quantity"])
    return 0.0

def buy(usd):
    r = rh_post("/api/v1/crypto/trading/orders/", {
        "symbol": SYMBOL, "side": "buy", "type": "market",
        "market_order_config": {"asset_quantity": None, "quote_amount": str(round(usd, 2))},
    })
    log(f"BUY ${usd} -> order {r.get('id')}")
    return r

def sell(qty):
    r = rh_post("/api/v1/crypto/trading/orders/", {
        "symbol": SYMBOL, "side": "sell", "type": "market",
        "market_order_config": {"asset_quantity": str(round(qty, 6)), "quote_amount": None},
    })
    log(f"SELL {qty} XRP -> order {r.get('id')}")
    return r

# ── State ────────────────────────────────────────────────────────────────────
pos = {
    "active":        False,
    "strategy":      "",
    "entry":         0.0,
    "qty":           0.0,
    "stop":          0.0,
    "target":        0.0,
    "bars_held":     0,
    "trail_active":  False,
    "trail_stop":    0.0,
    "atr_at_entry":  0.0,
}

state = {
    "cooldown_bars": 0,
}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def open_position_mr(price, qty, atr_val, bb_mid):
    pos.update({
        "active": True, "strategy": "MR",
        "entry": price, "qty": qty,
        "stop": price - (ATR_STOP_MR * atr_val),
        "target": bb_mid,
        "bars_held": 0,
        "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_val,
    })
    log(f"[MR] POSITION OPEN | Entry={price:.4f} | SL={pos['stop']:.4f} | TP={pos['target']:.4f} (BB mid) | ATR={atr_val:.4f}")

def open_position_tf(price, qty, atr_val):
    pos.update({
        "active": True, "strategy": "TF",
        "entry": price, "qty": qty,
        "stop":   price - (ATR_STOP_TF * atr_val),
        "target": price + (ATR_TARGET_TF * atr_val),
        "bars_held": 0,
        "trail_active": False, "trail_stop": 0.0, "atr_at_entry": atr_val,
    })
    log(f"[TF] POSITION OPEN | Entry={price:.4f} | SL={pos['stop']:.4f} | TP={pos['target']:.4f} | ATR={atr_val:.4f}")

def close_position(reason, hit_stop=False):
    sell(pos["qty"])
    log(f"[{pos['strategy']}] POSITION CLOSED - {reason}")
    pos["active"] = False
    if hit_stop:
        state["cooldown_bars"] = COOLDOWN_BARS
        log(f"COOLDOWN activated - skipping next {COOLDOWN_BARS} bars")

# ── Main Loop ────────────────────────────────────────────────────────────────
def run():
    log("XRP DUAL-STRATEGY BOT STARTED")
    log(f"Regime switch threshold: ADX {ADX_TREND}")
    log(f"ADX < {ADX_TREND} = Mean Reversion (BB+RSI) | ADX > {ADX_TREND} = Trend Following (EMA+MACD)")
    log(f"Trade size: ${TRADE_AMOUNT}")

    held = get_holdings()
    if held > 0.01:
        price = get_price()
        df    = compute(fetch_candles(15))
        last  = df.iloc[-1]
        adx   = last["adx"]
        if adx < ADX_TREND:
            open_position_mr(price, held, last["atr"], last["bb_mid"])
        else:
            open_position_tf(price, held, last["atr"])
        log(f"Resumed existing position of {held:.4f} XRP")

    while True:
        try:
            df15  = compute(fetch_candles(15))
            df1h  = compute(fetch_candles(60))
            last  = df15.iloc[-1]
            prev  = df15.iloc[-2]
            price = get_price()

            # Shared readings
            adx      = last["adx"]
            atr_v    = last["atr"]
            atr_avg  = last["atr_avg"]
            atr_ratio = atr_v / atr_avg if atr_avg > 0 else 1.0
            vol      = last["volume"]
            vol_avg  = last["vol_avg"]
            vol_ok   = vol > vol_avg
            rsi_1h   = df1h.iloc[-1]["rsi"]
            regime   = "RANGING" if adx < ADX_TREND else "TRENDING"

            # MR readings
            rsi_15   = last["rsi"]
            bb_lower = last["bb_lower"]
            bb_mid   = last["bb_mid"]
            bb_width = last["bb_width"]
            bb_w_avg = last["bb_w_avg"]
            squeeze  = bb_width < bb_w_avg
            atr_ok   = atr_ratio < ATR_RATIO_MAX
            oversold = price <= bb_lower and rsi_15 < RSI_ENTRY_MR

            # TF readings
            ema9       = last["ema9"]
            ema21      = last["ema21"]
            macd_v     = last["macd"]
            macd_sig_v = last["macd_sig"]
            prev_ema9  = prev["ema9"]
            prev_ema21 = prev["ema21"]
            prev_macd  = prev["macd"]
            prev_msig  = prev["macd_sig"]
            ema_cross  = ema9 > ema21 and prev_ema9 <= prev_ema21
            macd_bull  = macd_v > macd_sig_v and prev_macd <= prev_msig
            rsi_ok_tf  = 40 <= rsi_15 <= 60

            log(
                f"P={price:.4f} | Regime={regime} (ADX={adx:.1f}) | "
                f"RSI15={rsi_15:.1f} RSI1H={rsi_1h:.1f} | "
                f"ATR_ratio={atr_ratio:.2f} | Vol={'OK' if vol_ok else 'LOW'} | "
                f"Squeeze={'YES' if squeeze else 'NO'} | "
                f"Cooldown={state['cooldown_bars']} | Pos={'YES' if pos['active'] else 'NO'}"
            )

            if state["cooldown_bars"] > 0:
                state["cooldown_bars"] -= 1
                log(f"In cooldown - {state['cooldown_bars']} bars remaining")

            # ── Manage open position ─────────────────────────────────────────
            if pos["active"]:
                pos["bars_held"] += 1
                strat = pos["strategy"]

                if strat == "MR":
                    if price <= pos["stop"]:
                        close_position(f"Stop loss at {price:.4f}", hit_stop=True)
                    elif price >= pos["target"]:
                        close_position(f"Take profit at {price:.4f} (BB mid)")
                    elif pos["bars_held"] >= TIME_EXIT_BARS:
                        close_position(f"Time exit after {TIME_EXIT_BARS} bars")
                    else:
                        pct = ((price - pos["entry"]) / pos["entry"]) * 100
                        log(f"[MR] Holding | Bar {pos['bars_held']}/{TIME_EXIT_BARS} | PnL={pct:+.2f}% | SL={pos['stop']:.4f} | TP={pos['target']:.4f}")

                elif strat == "TF":
                    gain = price - pos["entry"]
                    if not pos["trail_active"] and gain >= ATR_TRAIL_MULT * pos["atr_at_entry"]:
                        pos["trail_active"] = True
                        pos["trail_stop"]   = pos["entry"] + (ATR_TRAIL_LOCK * pos["atr_at_entry"])
                        log(f"[TF] TRAILING STOP activated at {pos['trail_stop']:.4f}")

                    if pos["trail_active"] and price <= pos["trail_stop"]:
                        close_position(f"Trailing stop at {price:.4f}")
                    elif price <= pos["stop"]:
                        close_position(f"Stop loss at {price:.4f}", hit_stop=True)
                    elif price >= pos["target"]:
                        close_position(f"Take profit at {price:.4f}")
                    elif ema9 < ema21 and prev_ema9 >= prev_ema21 and rsi_15 > 55:
                        close_position(f"Bearish EMA cross + RSI={rsi_15:.1f}")
                    else:
                        pct = ((price - pos["entry"]) / pos["entry"]) * 100
                        log(f"[TF] Holding | PnL={pct:+.2f}% | Trail={'ON' if pos['trail_active'] else 'OFF'} | SL={pos['stop']:.4f} | TP={pos['target']:.4f}")

            # ── Look for new entries ─────────────────────────────────────────
            else:
                if state["cooldown_bars"] > 0:
                    pass

                elif regime == "RANGING":
                    log(f"[MR mode] BB_LOW={bb_lower:.4f} | Price vs BB: {((price-bb_lower)/bb_lower*100):+.2f}% | RSI={rsi_15:.1f}")
                    if not atr_ok:
                        log(f"[MR] SKIP - ATR expanding too fast (ratio={atr_ratio:.2f} > {ATR_RATIO_MAX})")
                    elif squeeze:
                        log("[MR] SKIP - BB squeeze, low volatility")
                    elif not vol_ok:
                        log("[MR] SKIP - Volume below average")
                    elif rsi_1h < RSI_1H_MIN:
                        log(f"[MR] SKIP - 1H RSI={rsi_1h:.1f} below {RSI_1H_MIN}, trend bearish")
                    elif oversold:
                        log(f"[MR] BUY SIGNAL | Price={price:.4f} <= BB_LOW={bb_lower:.4f} | RSI15={rsi_15:.1f} | 1H_RSI={rsi_1h:.1f}")
                        buy(TRADE_AMOUNT)
                        open_position_mr(price, TRADE_AMOUNT / price, atr_v, bb_mid)
                    else:
                        log(f"[MR] Watching - waiting for oversold signal")

                elif regime == "TRENDING":
                    log(f"[TF mode] EMA9={ema9:.4f} EMA21={ema21:.4f} | MACD={'up' if macd_v > macd_sig_v else 'dn'} | RSI={rsi_15:.1f}")
                    if not vol_ok:
                        log("[TF] SKIP - Volume below average")
                    elif ema_cross and rsi_ok_tf and macd_bull:
                        log(f"[TF] BUY SIGNAL | EMA cross | RSI={rsi_15:.1f} | MACD bullish | Vol=OK")
                        buy(TRADE_AMOUNT)
                        open_position_tf(price, TRADE_AMOUNT / price, atr_v)
                    elif ema_cross:
                        reasons = []
                        if not rsi_ok_tf: reasons.append(f"RSI={rsi_15:.1f} out of 40-60")
                        if not macd_bull: reasons.append("MACD not confirmed")
                        if not vol_ok:    reasons.append("Volume LOW")
                        log(f"[TF] EMA cross SKIPPED - {', '.join(reasons)}")
                    else:
                        log(f"[TF] Watching - no EMA cross yet")

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    run()
