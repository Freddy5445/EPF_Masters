"""
Test script for the ENTSO-E Transparency Platform RESTful API.

Fetches, for DK1 (Denmark West) and every hour of 2024:

* Day-ahead price [12.1.D]                       -- documentType=A44
* Day-ahead solar generation forecast [14.1.D]    -- documentType=A69, psrType=B16
* Day-ahead wind generation forecast [14.1.D]     -- documentType=A69, psrType=B18/B19

Wind and solar share the same document (A69); the platform returns one
TimeSeries per production type (psrType), which is why they come back from a
single request and are split apart here rather than fetched separately. Wind
offshore (B18) and onshore (B19) are summed into one "Wind" column; solar
(B16) is its own column.

Reuses the entsoe_tp package's HTTP client, XML parser and DST-safe hourly
grid (see entsoe_tp/README.md) instead of re-implementing request throttling,
pagination, A03 curve-type expansion or fall-back/spring-forward handling.

Usage:
    python test_entsoe_dk1_2024.py

Requires an ENTSO-E security token, either as the ENTSOE_API_TOKEN
environment variable or in a .env file in the project root -- see
entsoe_tp/README.md for how to obtain one.
"""

import os
import sys

import pandas as pd

from entsoe_tp import TransparencyClient, lookup
from entsoe_tp.hourly import assert_epftoolbox_grid, to_local_hourly_grid
from entsoe_tp.parser import parse_document, to_series

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.path.join(THIS_DIR, ".cache", "entsoe")

ZONE = "DK1"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
OUT_PATH = os.path.join(THIS_DIR, "DK1_2024_price_solar_wind.csv")

# psrType codes the wind & solar forecast document (A69) breaks generation
# down by. See https://transparency.entsoe.eu -> "Allowed values" for A75/A69.
PSR_SOLAR = "B16"
PSR_WIND_OFFSHORE = "B18"
PSR_WIND_ONSHORE = "B19"


def _filter_day_ahead(frame):
    """Keep only the day-ahead auction TimeSeries from a price document.

    A price query can also return intraday series covering the same hours;
    dropping them stops an intraday price from displacing the day-ahead one.
    """
    if frame.empty:
        return frame
    contract = frame["contract_MarketAgreement.type"]
    if not contract.notna().any():
        return frame
    day_ahead = frame[contract == "A01"]
    return day_ahead if not day_ahead.empty else frame


def _fetch(client, params, value_tag, start_utc, end_utc, label):
    def progress(number, total, chunk_start, _chunk_end):
        print(f"  [{label}] chunk {number}/{total}  {chunk_start:%Y-%m}", flush=True)

    documents = client.fetch(params, start_utc, end_utc, progress=progress)
    frames = [parse_document(doc, value_tag) for doc in documents]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["timestamp", "value", "psr_type"])
    return pd.concat(frames, ignore_index=True)


def main():
    area = lookup(ZONE)
    start_date = pd.Timestamp(START_DATE).normalize()
    end_date = pd.Timestamp(END_DATE).normalize()

    start_utc = start_date.tz_localize(area.tz).tz_convert("UTC")
    end_utc = (end_date + pd.Timedelta(days=1)).tz_localize(area.tz).tz_convert("UTC")

    print(f"Zone {area.code} ({area.name}, {area.eic}) in {area.tz}")
    print(f"Range {start_date.date()} to {end_date.date()} inclusive\n")

    client = TransparencyClient(cache_dir=DEFAULT_CACHE)

    print("Fetching day-ahead price [A44]...")
    price_frame = _filter_day_ahead(_fetch(
        client,
        {"documentType": "A44", "in_Domain": area.eic, "out_Domain": area.eic},
        "price.amount", start_utc, end_utc, "price",
    ))
    price_series = to_series(price_frame, aggregate="first")

    print("Fetching day-ahead wind & solar forecast [A69]...")
    res_frame = _fetch(
        client,
        {"documentType": "A69", "processType": "A01", "in_Domain": area.eic},
        "quantity", start_utc, end_utc, "wind+solar",
    )
    solar_series = to_series(
        res_frame[res_frame["psr_type"] == PSR_SOLAR], aggregate="sum"
    )
    wind_series = to_series(
        res_frame[res_frame["psr_type"].isin([PSR_WIND_OFFSHORE, PSR_WIND_ONSHORE])],
        aggregate="sum",
    )

    print("\nAligning to local hourly grid...")
    frame = pd.DataFrame({
        "Price": to_local_hourly_grid(price_series, area.tz, start_date, end_date),
        "Solar Forecast": to_local_hourly_grid(solar_series, area.tz, start_date, end_date),
        "Wind Forecast": to_local_hourly_grid(wind_series, area.tz, start_date, end_date),
    })
    frame.index.name = "Date"
    assert_epftoolbox_grid(frame)

    frame.to_csv(OUT_PATH, date_format="%Y-%m-%d %H:%M:%S")

    print(f"\nWrote {OUT_PATH}")
    print(f"  {len(frame)} rows ({len(frame) // 24} days), {frame.index[0]} to {frame.index[-1]}")
    print(f"  columns: {list(frame.columns)}")
    print(f"\n{frame.describe()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
