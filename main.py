"""Forecast the BTC/USDC close 240 minutes after each checkpoint.

Entry for make.fun competition ``btc-cent-forecast-01``.

Python 3.13, standard library only, offline, deterministic.

Input
-----
Every file ``<id>.in`` in ``MAKEFUN_INPUT_DIR`` (default ``/input``) holds one
case in the ``makefun/btc-cent-case/v1`` JSON format: a single market with
4,320 one-minute closes covering the 72 hours that end at the checkpoint.
Each case is processed in file-name order.

Output
------
For every case, one file ``<id>.out`` in ``MAKEFUN_OUTPUT_DIR`` (default
``/output``) containing a single JSON line::

    {"forecasts":[{"case":"<id>","price":<number>}]}

Model
-----
The forecast is the checkpoint's last close, shrunk a quarter of the way
toward the median of the last 15 closes::

    guess = 0.75 * last_close + 0.25 * median(closes[-15:])

Why this shape: scoring is mean absolute error against a house baseline of
pure persistence (the checkpoint close itself), and a prize needs a loss
strictly below that baseline, so a byte-identical copy of persistence can
never win. The checkpoint close is one print of a thin DEX pool (the median
of the samples in its bar's last 10 seconds), so it carries pool
microstructure noise that says nothing about where the pool trades 4 hours
later; the median of the recent quarter-hour is a stabler estimate of the
level the market is actually at. The shrinkage is deliberately small: at a
4-hour horizon Bitcoin is close to a martingale, and every stronger
adjustment tried on the practice window (drift extrapolation over 15-minute
to 24-hour trend bands, mean reversion to the 24-hour mean, heavier
smoothing) cost more than it earned, because a single window's trend
reverses about as often as it continues. Backtests on pseudo-cases cut from
the practice recording put this light shrinkage within half a percent of
persistence -- a fair, cheap deviation that differs from the baseline on
essentially every case.

The guess is rounded to the price grid (5 decimals of a dollar) and clamped
inside the scorer's bounds ``(0, 10_000_000)``, exactly as the baseline does.
Two runs over the same inputs write the same bytes.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from typing import Any, Iterator, Sequence

#: Directory holding the ``<id>.in`` case files.
INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
#: Directory the ``<id>.out`` forecast files are written to.
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

#: Smallest positive price representable on the 5-decimal grid.
GRID_MIN = 0.00001
#: Largest grid price still below the scorer's exclusive $10,000,000 cap.
GRID_MAX = 9999999.99999
#: Number of decimals on the price grid.
GRID_DECIMALS = 5

#: Weight given to the recent median in the shrunk-level guess.
MEDIAN_WEIGHT = 0.25
#: How many trailing one-minute closes form the median window (a quarter-hour).
MEDIAN_WINDOW = 15

#: Fraction of the last close kept in the guess (``1 - MEDIAN_WEIGHT``).
LAST_WEIGHT = 1.0 - MEDIAN_WEIGHT


def load_cases(input_dir: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield ``(case_id, case_dict)`` for every ``.in`` file, in name order.

    Files that do not end in ``.in`` are skipped. Name order (not directory
    order) is what makes the output deterministic across runs.
    """
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".in"):
            continue
        case_id = filename[: -len(".in")]
        with open(os.path.join(input_dir, filename), encoding="utf-8") as handle:
            yield case_id, json.load(handle)


def sole_market(case: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Return the one market a case must hold, or raise ``ValueError``.

    A case is defined to hold exactly one market; anything else is a
    malformed input, not something to guess around.
    """
    markets = case.get("markets")
    if not isinstance(markets, list) or len(markets) != 1:
        raise ValueError(f"case {case_id} does not hold exactly one market")
    return markets[0]


def last_close(market: dict[str, Any]) -> float:
    """Return the checkpoint close of ``market``, validated as a finite number.

    Raises ``ValueError`` when the market has no prices, or when its last
    price is missing, non-numeric, or not finite.
    """
    closes = market.get("prices")
    if not isinstance(closes, list) or not closes:
        raise ValueError(f"market {market.get('id')} has no prices")
    close = closes[-1]
    if not isinstance(close, (int, float)) or not math.isfinite(close):
        raise ValueError(f"market {market.get('id')}'s last price is not a finite number")
    return float(close)


def trailing_median(closes: Sequence[float], window: int = MEDIAN_WINDOW) -> float:
    """Median of the last ``window`` closes from ``closes``.

    Only the tail of the series is sliced, so the median estimates the level
    at the checkpoint rather than an older regime -- and the full 4,320-close
    series is never copied just to look at its last 15 entries.
    """
    return float(statistics.median(closes[-window:]))


def shrunk_level(close: float, level: float) -> float:
    """Blend the last close with a stabler level estimate.

    ``MEDIAN_WEIGHT`` of the guess comes from ``level`` (the trailing
    median); the rest stays anchored on the checkpoint close itself.
    """
    return LAST_WEIGHT * close + MEDIAN_WEIGHT * level


def snap_to_grid(value: float) -> float:
    """Round ``value`` to the 5-decimal price grid and clamp to ``(0, 10_000_000)``.

    Rounding is half-up via ``floor(x * 1e5 + 0.5) / 1e5``; clamping happens
    before rounding so the result always lands on a valid grid price.
    """
    scale = 10 ** GRID_DECIMALS
    clamped = min(GRID_MAX, max(GRID_MIN, value))
    return math.floor(clamped * scale + 0.5) / scale


def forecast_price(market: dict[str, Any]) -> float:
    """Forecast one market's close 240 minutes after its checkpoint.

    Combines :func:`last_close`, :func:`trailing_median` and
    :func:`shrunk_level`, then snaps the result to the price grid.
    """
    closes = market.get("prices")
    close = last_close(market)
    median = trailing_median(closes)
    return snap_to_grid(shrunk_level(close, median))


def write_forecast(output_dir: str, case_id: str, price: float) -> None:
    """Write the single-line ``<case_id>.out`` file for one forecast."""
    payload = {"forecasts": [{"case": case_id, "price": price}]}
    with open(os.path.join(output_dir, case_id + ".out"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def run(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
    """Forecast every case in ``input_dir`` and write its ``.out`` file."""
    os.makedirs(output_dir, exist_ok=True)
    for case_id, case in load_cases(input_dir):
        market = sole_market(case, case_id)
        write_forecast(output_dir, case_id, forecast_price(market))


def main() -> None:
    """Entry point: forecast all cases using the configured directories."""
    run()


if __name__ == "__main__":
    main()
