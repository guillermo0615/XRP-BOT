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

BB_PERIOD      = 20
BB_STD         = 2.0
RSI_PERIOD     = 14
RSI_ENTRY      = 35
RSI_1H_MIN     = 40
ATR_PERIOD     = 14
ATR_STOP_MULT  = 1.5
VOLUME_PERIOD  = 20
TIME_EXIT_BARS = 8
COOLDOWN_BARS  = 4

RH_BASE_URL = "https://trading.robinhood.com"

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

def fetch_candles_15m():
    url = "https://api.kraken.com/0/public/OHLC?pair=XRPUSD&interval=15"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise ValueError(f"Kraken 15m error: {data['error']}")
    pts = data["result"].get("XXRPZUSD") or data["result"].get("XRPUSD")
    if pts is None:
        raise ValueError(f"Unexpected Kraken keys: {list(data['result'].keys())}")
    df = pd.DataFrame(pts, columns=["time","open","high","low","close","vwap","volume","count"])
    df["close"]  = df["close"].astype(float)
    df["high"]   = df["high"].astype(float)
    df["low"]    = df["low"].astype(float)
    df["volume"] = df["volume"].astype(float)
    return df.iloc[:-1].reset_index(drop=True)

def fetch_candles_1h():
    url = "https://api.kraken.com/0/public/OHLC?pair=XRPUSD&interval=60"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data.get("error"):
        raise ValueError(f"Kraken 1h error: {data['error']}")
    pts = data["result"].get("XXRPZUSD") or data["result"].get("XRPUSD")
    if pts is None:
        raise ValueError(f"Unexpected Kraken 1h keys: {list(data['result'].keys())}")
    df = pd.DataFrame(pts, columns=["time","open","high","low","close","vwap","volume","count"])
    df["close"] = df["close"].astype(float)
    return df.iloc[:-1].reset_index(drop=True)

def get_price():
    path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={SYMBOL}"
    d = rh_get(path)["results"][0]
    return (float(d["ask_inclusive_of_buy_spread"]) + float(d["bid_inclusive_of_sell_spread"])) / 2

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - (100 / (1 + g / l))

def calc_atr(df, n=14):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def compute_15m(df):
    close = df["close"]
    df["rsi"] = calc_rsi(close, RSI_PERIOD)
    df["atr"] = calc_atr(df, ATR_PERIOD)
    df["bb_mid"]   = close.rolling(BB_PERIOD).mean()
    df["bb_std"]   = close.rolling(BB_PERIOD).std()
    df["bb_lower"] = df["bb_mid"] - BB_STD * df["bb_std"]
    df["bb_upper"] = df["bb_mid"] + BB_STD * df["bb_std"]
    df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    df["bb_width_avg"] = df["bb_width"].rolling(BB_PERIOD).mean()
    df["vol_avg"]  = df["volume"].rolling(VOLUME_PERIOD).mean()
    return df

def compute_1h(df):
    df["rsi"] = calc_rsi(df["close"], RSI_PERIOD)
    return df

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

pos = {
    "active":        False,
    "entry":         0.0,
    "qty":           0.0,
    "stop":          0.0,
    "target":        0.0,
    "bars_held":     0,
}

state = {
    "cooldown_bars": 0,
}

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def open_position(price, qty, atr_val, bb_mid):
    pos.update({
        "active":    True,
        "entry":     price,
        "qty":       qty,
        "stop":      price - (ATR_STOP_MULT * atr_val),
        "target":    bb_mid,
        "bars_held": 0,
    })
    log(f"POSITION OPEN | Entry={price:.4f} | SL={pos['stop']:.4f} | TP={pos['target']:.4f} | ATR={atr_val:.4f}")

def close_position(reason, hit_stop=False):
    sell(pos["qty"])
    log(f"POSITION CLOSED - {reason}")
    pos["active"] = False
    if hit_stop:
        state["cooldown_bars"] = COOLDOWN_BARS
        log(f"COOLDOWN activated - skipping next {COOLDOWN_BARS} bars")

def run():
    log("XRP BOT STARTED - BB + RSI Mean Reversion Strategy")
    log(f"Entry: price below lower BB + RSI < {RSI_ENTRY} + 1H RSI > {RSI_1H_MIN} + volume OK + no squeeze")
    log(f"Exit: middle BB target | SL={ATR_STOP_MULT}x ATR | Time exit={TIME_EXIT_BARS} bars | Cooldown={COOLDOWN_BARS} bars after loss")
    log(f"Trade size: ${TRADE_AMOUNT}")

    held = get_holdings()
    if held > 0.01:
        price  = get_price()
        df15   = compute_15m(fetch_candles_15m())
        last   = df15.iloc[-1]
        open_position(price, held, last["atr"], last["bb_mid"])
        log(f"Resumed existing position of {held:.4f} XRP")

    while True:
        try:
            df15  = compute_15m(fetch_candles_15m())
            df1h  = compute_1h(fetch_candles_1h())
            last  = df15.iloc[-1]
            price = get_price()

            rsi_15   = last["rsi"]
            bb_lower = last["bb_lower"]
            bb_mid   = last["bb_mid"]
            bb_width = last["bb_width"]
            bb_w_avg = last["bb_width_avg"]
            atr_v    = last["atr"]
            vol      = last["volume"]
            vol_avg  = last["vol_avg"]
            rsi_1h   = df1h.iloc[-1]["rsi"]

            squeeze    = bb_width < bb_w_avg
            vol_ok     = vol > vol_avg
            oversold   = price <= bb_lower and rsi_15 < RSI_ENTRY
            trend_ok   = rsi_1h > RSI_1H_MIN
            in_cooldown = state["cooldown_bars"] > 0

            log(
                f"P={price:.4f} | BB_LOW={bb_lower:.4f} BB_MID={bb_mid:.4f} | "
                f"RSI15={rsi_15:.1f} RSI1H={rsi_1h:.1f} | "
                f"Vol={'OK' if vol_ok else 'LOW'} | Squeeze={'YES' if squeeze else 'NO'} | "
                f"Cooldown={state['cooldown_bars']} | Pos={'YES' if pos['active'] else 'NO'}"
            )

            if in_cooldown:
                state["cooldown_bars"] -= 1
                log(f"In cooldown - {state['cooldown_bars']} bars remaining")

            if pos["active"]:
                pos["bars_held"] += 1

                if price <= pos["stop"]:
                    close_position(f"Stop loss hit at {price:.4f}", hit_stop=True)
                elif price >= pos["target"]:
                    close_position(f"Take profit hit at {price:.4f} (middle BB)")
                elif pos["bars_held"] >= TIME_EXIT_BARS:
                    close_position(f"Time exit after {TIME_EXIT_BARS} bars")
                else:
                    pct = ((price - pos["entry"]) / pos["entry"]) * 100
                    log(f"Holding | Bars={pos['bars_held']}/{TIME_EXIT_BARS} | PnL={pct:+.2f}% | SL={pos['stop']:.4f} | TP={pos['target']:.4f}")

            else:
                if in_cooldown:
                    pass
                elif squeeze:
                    log("SKIP - Bollinger Band squeeze, low volatility")
                elif not vol_ok:
                    log("SKIP - Volume below average")
                elif not trend_ok:
                    log(f"SKIP - 1H RSI={rsi_1h:.1f} below {RSI_1H_MIN}, broader trend bearish")
                elif oversold:
                    log(f"BUY SIGNAL | Price={price:.4f} below BB_LOW={bb_lower:.4f} | RSI15={rsi_15:.1f} | RSI1H={rsi_1h:.1f} | Vol=OK")
                    buy(TRADE_AMOUNT)
                    open_position(price, TRADE_AMOUNT / price, atr_v, bb_mid)
                else:
                    log(f"WATCHING | Price vs BB_LOW: {((price - bb_lower) / bb_lower * 100):+.2f}% | RSI15={rsi_15:.1f}")

        except Exception as e:
            log(f"ERROR: {e}")

        time.sleep(INTERVAL_SEC)

if __name__ == "__main__":
    run()
