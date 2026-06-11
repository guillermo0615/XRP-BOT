#!/usr/bin/env python3
"""
DAILY TREND SYSTEM BACKTEST - XRP/USD
=====================================
Replaces the 4-strategy 15-min bot with ONE daily trend system.

Entry : daily close > highest close of prior DON_LEN days
        AND daily close > SMA(SMA_FILTER)
        -> buy at next daily open (long only, spot, no leverage)
Exit  : daily close < max(initial stop, chandelier trail)
        -> sell at next daily open
        initial stop = entry - ATR_STOP * ATR(14) at entry
        chandelier   = highest close since entry - ATR_TRAIL * ATR(14) current

Fees  : 0.26% per side (Kraken taker, round-trip 0.52%)

Run:
    pip install pandas requests
    python3 backtest_trend.py

DO NOT DEPLOY trend_bot.py until you have read this output.
Deploy gates are printed at the bottom of the report.
"""

import requests
import time
import json
import pandas as pd
from datetime import datetime, timezone, timedelta

# ── Strategy parameters (chosen a priori — do not curve-fit these) ──────────
DON_LEN    = 30      # breakout lookback (days)
SMA_FILTER = 200     # long-term trend filter (days)
ATR_LEN    = 14
ATR_STOP   = 2.5     # initial stop multiple
ATR_TRAIL  = 3.0     # chandelier trail multiple
FEE_SIDE   = 0.0026  # Kraken taker per side
NOTIONAL   = 26.0    # full allocation, one position
YEARS      = 5

BITSTAMP = "https://www.bitstamp.net/api/v2/ohlc/xrpusd/"


# ── Data ─────────────────────────────────────────────────────────────────────
def fetch_daily(years=YEARS):
    """Daily XRP/USD candles from Bitstamp, with extra history for SMA warmup."""
    step = 86400
    now  = int(datetime.now(timezone.utc).timestamp())
    cur  = int((datetime.now(timezone.utc)
                - timedelta(days=365 * years + SMA_FILTER + DON_LEN + 10)).timestamp())
    rows, seen = [], set()
    print("Fetching daily candles from Bitstamp...")
    while cur < now:
        r = requests.get(BITSTAMP, params={"step": step, "limit": 1000, "start": cur},
                         timeout=15)
        r.raise_for_status()
        ohlc = r.json()["data"]["ohlc"]
        if not ohlc:
            break
        new = 0
        for k in ohlc:
            ts = int(k["timestamp"])
            if ts not in seen:
                seen.add(ts)
                new += 1
                rows.append({
                    "time":   ts,
                    "open":   float(k["open"]),
                    "high":   float(k["high"]),
                    "low":    float(k["low"]),
                    "close":  float(k["close"]),
                    "volume": float(k["volume"]),
                })
        if new == 0:
            break
        cur = int(ohlc[-1]["timestamp"]) + step
        time.sleep(0.2)
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    df = df.iloc[:-1].reset_index(drop=True)   # drop in-progress candle
    print(f"  -> {len(df):,} daily candles | "
          f"{datetime.fromtimestamp(df.iloc[0]['time']).strftime('%Y-%m-%d')} -> "
          f"{datetime.fromtimestamp(df.iloc[-1]['time']).strftime('%Y-%m-%d')}")
    return df


# ── Indicators ───────────────────────────────────────────────────────────────
def calc_atr(df, n=ATR_LEN):
    h, l, pc = df["high"], df["low"], df["close"].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


# ── Backtest ─────────────────────────────────────────────────────────────────
def run_backtest(df):
    df = df.copy()
    df["atr"] = calc_atr(df)
    df["sma"] = df["close"].rolling(SMA_FILTER).mean()
    # Highest close of the PRIOR DON_LEN days (excludes today => true breakout)
    df["don"] = df["close"].shift(1).rolling(DON_LEN).max()

    trades = []
    pos = None
    start_i = SMA_FILTER + DON_LEN

    for i in range(start_i, len(df) - 1):
        row, nxt = df.iloc[i], df.iloc[i + 1]
        c = float(row["close"])

        if pos is None:
            if c > float(row["don"]) and c > float(row["sma"]):
                entry = float(nxt["open"])
                pos = {
                    "entry":    entry,
                    "entry_i":  i + 1,
                    "stop":     entry - ATR_STOP * float(row["atr"]),
                    "hi":       entry,
                }
        else:
            pos["hi"] = max(pos["hi"], c)
            trail = pos["hi"] - ATR_TRAIL * float(row["atr"])
            level = max(pos["stop"], trail)
            if c < level:
                exit_p = float(nxt["open"])
                gross = (exit_p - pos["entry"]) / pos["entry"] * 100
                net   = gross - FEE_SIDE * 2 * 100
                trades.append({
                    "entry_date": datetime.fromtimestamp(
                        int(df.iloc[pos["entry_i"]]["time"])).strftime("%Y-%m-%d"),
                    "exit_date":  datetime.fromtimestamp(
                        int(nxt["time"])).strftime("%Y-%m-%d"),
                    "entry":      round(pos["entry"], 4),
                    "exit":       round(exit_p, 4),
                    "days_held":  (i + 1) - pos["entry_i"],
                    "gross_pct":  round(gross, 2),
                    "net_pct":    round(net, 2),
                    "exit_reason": "stop" if level == pos["stop"] else "trail",
                })
                pos = None

    # Force-close any open position at final close
    if pos is not None:
        exit_p = float(df.iloc[-1]["close"])
        gross = (exit_p - pos["entry"]) / pos["entry"] * 100
        net   = gross - FEE_SIDE * 2 * 100
        trades.append({
            "entry_date": datetime.fromtimestamp(
                int(df.iloc[pos["entry_i"]]["time"])).strftime("%Y-%m-%d"),
            "exit_date":  datetime.fromtimestamp(
                int(df.iloc[-1]["time"])).strftime("%Y-%m-%d"),
            "entry":      round(pos["entry"], 4),
            "exit":       round(exit_p, 4),
            "days_held":  (len(df) - 1) - pos["entry_i"],
            "gross_pct":  round(gross, 2),
            "net_pct":    round(net, 2),
            "exit_reason": "end_of_data",
        })

    return trades, df, start_i


# ── Report ───────────────────────────────────────────────────────────────────
def report(trades, df, start_i):
    if not trades:
        print("\nNo trades generated — check data range.")
        return

    t = pd.DataFrame(trades)
    wins, losses = t[t.net_pct > 0], t[t.net_pct <= 0]
    wr = len(wins) / len(t) * 100
    pf = (wins.net_pct.sum() / abs(losses.net_pct.sum())
          if len(losses) and losses.net_pct.sum() != 0 else float("inf"))

    # Compounded equity (full notional redeployed each trade) + max drawdown
    eq, peak, maxdd = 1.0, 1.0, 0.0
    for n in t["net_pct"]:
        eq *= (1 + n / 100)
        peak = max(peak, eq)
        maxdd = max(maxdd, (peak - eq) / peak)

    span_days = (int(df.iloc[-1]["time"]) - int(df.iloc[start_i]["time"])) / 86400
    span_yrs  = span_days / 365.0
    cagr = eq ** (1 / span_yrs) - 1 if span_yrs > 0 else 0

    # Buy & hold benchmark over the same window
    bh = (float(df.iloc[-1]["close"]) / float(df.iloc[start_i]["close"]) - 1) * 100

    p0 = datetime.fromtimestamp(int(df.iloc[start_i]["time"])).strftime("%Y-%m-%d")
    p1 = datetime.fromtimestamp(int(df.iloc[-1]["time"])).strftime("%Y-%m-%d")

    print()
    print("=" * 64)
    print("DAILY TREND SYSTEM — BACKTEST RESULTS (XRP/USD)")
    print("=" * 64)
    print(f"Period          : {p0} -> {p1}  ({span_yrs:.1f} years)")
    print(f"Params          : DON={DON_LEN} SMA={SMA_FILTER} "
          f"STOP={ATR_STOP}xATR TRAIL={ATR_TRAIL}xATR")
    print(f"Fees            : {FEE_SIDE*100:.2f}%/side "
          f"(round-trip {FEE_SIDE*200:.2f}%)")
    print()
    print(f"Trades          : {len(t)}  (~{len(t)/span_yrs:.1f}/year)")
    print(f"Win rate        : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
    if len(wins):
        print(f"Avg win         : {wins.net_pct.mean():+.2f}%   "
              f"(best {wins.net_pct.max():+.2f}%)")
    if len(losses):
        print(f"Avg loss        : {losses.net_pct.mean():+.2f}%   "
              f"(worst {losses.net_pct.min():+.2f}%)")
    print(f"Profit factor   : {pf:.2f}")
    print(f"Sum net %       : {t.net_pct.sum():+.2f}%  (not compounded)")
    print(f"Compounded      : {(eq-1)*100:+.2f}%  total  |  CAGR {cagr*100:+.1f}%/yr")
    print(f"Max drawdown    : {maxdd*100:.1f}%  (trade-close basis; intraday is worse)")
    print(f"Avg hold        : {t.days_held.mean():.1f} days")
    print(f"Total fee drag  : {len(t) * FEE_SIDE * 200:.1f}% over the full period")
    print(f"Buy & hold      : {bh:+.2f}% over same window (benchmark)")

    print()
    print("── BY CALENDAR YEAR (exit year) ────────────────────────────")
    t["year"] = t["exit_date"].str[:4]
    for yr, grp in t.groupby("year"):
        ywr = len(grp[grp.net_pct > 0]) / len(grp) * 100
        print(f"  {yr}: {len(grp):2d} trades | {ywr:3.0f}% WR | net {grp.net_pct.sum():+8.2f}%")

    print()
    print("── EVERY TRADE ─────────────────────────────────────────────")
    for _, r in t.iterrows():
        print(f"  {r.entry_date} -> {r.exit_date} | {r.days_held:3d}d | "
              f"{r.entry:.4f} -> {r.exit:.4f} | net {r.net_pct:+7.2f}% | {r.exit_reason}")

    pos_years = sum(1 for _, g in t.groupby("year") if g.net_pct.sum() >= 0)
    n_years   = t["year"].nunique()
    top_trade = t.net_pct.max()
    concentration = top_trade / t.net_pct.sum() * 100 if t.net_pct.sum() > 0 else float("nan")

    print()
    print("── DEPLOY GATES ────────────────────────────────────────────")
    print(f"  [{'PASS' if pf >= 1.5 else 'FAIL'}] Profit factor >= 1.5        (got {pf:.2f})")
    print(f"  [{'PASS' if (eq-1) > 0 else 'FAIL'}] Compounded return positive  (got {(eq-1)*100:+.1f}%)")
    print(f"  [{'PASS' if pos_years >= max(3, n_years-2) else 'FAIL'}] Non-negative in most years  ({pos_years}/{n_years})")
    print(f"  [{'PASS' if maxdd <= 0.40 else 'FAIL'}] Max drawdown <= 40%         (got {maxdd*100:.1f}%)")
    if not pd.isna(concentration):
        print(f"  [INFO] Largest single trade = {concentration:.0f}% of total profit "
              f"(trend systems are concentrated — expect this to be high)")
    print()
    print("  If any gate FAILS: do not deploy. The fallback is no bot at all —")
    print("  a small spot position you simply hold costs 0%/yr in fees.")
    print("=" * 64)

    with open("backtest_trend_trades.json", "w") as f:
        json.dump(trades, f, indent=2)
    print("\nTrade log -> backtest_trend_trades.json")


def main():
    print("XRP DAILY TREND BACKTESTER")
    print("=" * 64)
    df = fetch_daily()
    trades, df, start_i = run_backtest(df)
    report(trades, df, start_i)


if __name__ == "__main__":
    main()
