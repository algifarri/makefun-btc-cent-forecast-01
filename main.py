# btc-cent-forecast-01 entry (Python 3.13, standard library only, offline, deterministic).
#
# Task: read one case at a time -- BTC/USDC 1-minute closes for the 72 hours ending at a
# checkpoint -- and write ONE number: the forecast of the close 240 minutes later.
# Scoring is mean absolute error, and a prize needs a loss strictly below the house
# baseline, which is pure persistence (the checkpoint's own close). A byte-identical
# copy of persistence can therefore never win anything: a deliberate, minimal,
# principled deviation is required.
#
# Model: anchor the forecast mostly on the last close, but shrink it a quarter of
# the way toward the median of the last 15 closes:
#
#     guess = 0.75 * last + 0.25 * median(prices[-15:])
#
# Rationale: the checkpoint close is a single print of a thin DEX pool (the median of
# the samples in its bar's last 10 seconds). One print carries pool microstructure
# noise that is not information about where the pool trades 4 hours later; the recent
# quarter-hour's median is a stabler estimate of the level the market is actually at.
# The shrinkage is kept small on purpose: at a 4-hour horizon Bitcoin is close to a
# martingale, and every stronger adjustment tried on the practice window (drift
# extrapolation from 15-minute to 24-hour trend bands, mean reversion to the 24-hour
# mean, heavier smoothing) cost more than it earned, because a single window's trend
# reverses about as often as it continues. Backtests on pseudo-cases cut from the
# practice recording put this light shrinkage within half a percent of persistence --
# a fair, cheap deviation that differs from the baseline on essentially every case.
#
# The guess is kept on the price grid (5 decimals of a dollar) and inside the scorer's
# bounds, exactly as the baseline does. Input directory and output directory come from
# MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR outside the grader's container, defaulting to
# /input and /output inside it. Every <id>.in is handled in name order, so two runs
# write the same bytes.

import json
import math
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# The smallest positive price on the grid: one unit of its 5 decimals.
MIN_PRICE = 0.00001
# The largest grid price below the scorer's exclusive 10,000,000-dollar cap.
MAX_PRICE = 9999999.99999

# Shrinkage: weight on the recent median, and how many trailing closes form it.
MEDIAN_WEIGHT = 0.25
MEDIAN_WINDOW = 15


def price5(x):
    """x kept on the grid's 5 decimals and inside the scorer's (0, 10,000,000)."""
    return math.floor(min(MAX_PRICE, max(MIN_PRICE, x)) * 1e5 + 0.5) / 1e5


def forecast(market):
    """The shrunk-level guess of one market: last close pulled toward the recent median."""
    prices = market.get("prices")
    if not isinstance(prices, list) or len(prices) == 0:
        raise ValueError("market %s has no prices" % market.get("id"))
    last = prices[-1]
    if not isinstance(last, (int, float)) or not math.isfinite(last):
        raise ValueError("market %s's last price is not a finite number" % market.get("id"))
    window = prices[-MEDIAN_WINDOW:]
    level = (1.0 - MEDIAN_WEIGHT) * last + MEDIAN_WEIGHT * statistics.median(window)
    return price5(level)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for name in sorted(os.listdir(INPUT_DIR)):
        if not name.endswith(".in"):
            continue
        case_id = name[: -len(".in")]
        with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        markets = case.get("markets")
        if not isinstance(markets, list) or len(markets) != 1:
            raise ValueError("%s does not hold exactly one market" % name)
        out = {"forecasts": [{"case": case_id, "price": forecast(markets[0])}]}
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
