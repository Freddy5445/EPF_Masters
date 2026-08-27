"""
Installed generation capacity per production type [14.1.A].

Downloads ENTSO-E document type ``A68`` -- the aggregated installed capacity a
zone reports for a given year, broken down by production type -- and writes it
as a table.

    python -m entsoe_tp.installed_capacity
    python -m entsoe_tp.installed_capacity --zones DK1,DK2 --years 2015,2025
    python -m entsoe_tp.installed_capacity --long

Unlike the market-data documents this package usually fetches, A68 is a *stock*
reported once per year, not a time series. Each ``<TimeSeries>`` carries one
production type and one ``<Point>`` covering the whole year at resolution
``P1Y``, so it is parsed here rather than through :mod:`entsoe_tp.parser`, whose
period expansion is built for hourly and sub-hourly data.

The figure is what the TSO reported as installed at the start of the year, so
"2025" is capacity standing on 1 January 2025. It is a nameplate figure, not
availability: it does not account for outages, derating or seasonal capability.

Commands are given on one line: this project is developed from PowerShell, where
a trailing ``\`` is not a line continuation and truncates the command.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import pandas as pd

from .areas import lookup
from .client import TIME_FORMAT, TransparencyClient, TransparencyError
from .parser import _check_acknowledgement, _findtext, _iterfind, unit_label

# ENTSO-E's production-type codes, from the Transparency Platform code list.
# Kept in the platform's own order so a table's columns read the way its
# published breakdowns do.
PSR_TYPES = {
    "B01": "Biomass",
    "B02": "Fossil brown coal/lignite",
    "B03": "Fossil coal-derived gas",
    "B04": "Fossil gas",
    "B05": "Fossil hard coal",
    "B06": "Fossil oil",
    "B07": "Fossil oil shale",
    "B08": "Fossil peat",
    "B09": "Geothermal",
    "B10": "Hydro pumped storage",
    "B11": "Hydro run-of-river and poundage",
    "B12": "Hydro water reservoir",
    "B13": "Marine",
    "B14": "Nuclear",
    "B15": "Other renewable",
    "B16": "Solar",
    "B17": "Waste",
    "B18": "Wind offshore",
    "B19": "Wind onshore",
    "B20": "Other",
    "B21": "AC link",
    "B22": "DC link",
    "B23": "Substation",
    "B24": "Transformer",
    "B25": "Energy storage",
}

# The zones this study compares: the Nordic bidding zones it models, plus
# Germany-Luxembourg as the large thermal neighbour whose capacity mix the
# Nordic prices are increasingly coupled to.
DEFAULT_ZONES = ["DK1", "DK2", "DE_LU", "NO1", "NO2", "NO3", "NO4", "NO5", "SE"]
DEFAULT_YEARS = [2015, 2020, 2025]

DOCUMENT_TYPE = "A68"
# A33 is the process type the platform uses for installed capacity. Without it
# the query is rejected rather than defaulted.
PROCESS_TYPE = "A33"


def parse_installed_capacity(xml_text):
    """Parse one A68 document into ``[{psr_type, value, unit}, ...]``.

    Returns an empty list for an acknowledgement document -- the platform's way
    of saying it holds nothing for the query, which is a normal answer for a
    zone that did not report a given year, not an error.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise TransparencyError(f"Response is not valid XML: {exc}") from exc

    if _check_acknowledgement(root):
        return []

    rows = []
    for ts in _iterfind(root, "TimeSeries"):
        psr_type = None
        for mkt in _iterfind(ts, "MktPSRType"):
            psr_type = _findtext(mkt, "psrType")

        unit = unit_label(_findtext(ts, "quantity_Measure_Unit.name"))

        for period in _iterfind(ts, "Period"):
            for point in _iterfind(period, "Point"):
                quantity = _findtext(point, "quantity")
                if quantity is None:
                    continue
                rows.append({"psr_type": psr_type,
                             "value": float(quantity),
                             "unit": unit})
    return rows


def fetch_year(client, zone, year):
    """One zone, one year. Returns a list of row dicts."""
    area = lookup(zone)

    # A full calendar year in UTC. The platform wants the interval that contains
    # the year, and returns the single annual figure regardless of how the bounds
    # fall relative to local time.
    start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")

    # client.fetch splits a range into monthly chunks, which is right for time
    # series and wrong here: it would issue twelve identical requests and return
    # the same annual figure twelve times. One request, one year.
    params = {
        "documentType": DOCUMENT_TYPE,
        "processType": PROCESS_TYPE,
        "in_Domain": area.eic,
        "periodStart": start.strftime(TIME_FORMAT),
        "periodEnd": end.strftime(TIME_FORMAT),
    }
    body = client._get(params)

    rows = parse_installed_capacity(body)
    for row in rows:
        row.update({"zone": zone, "eic": area.eic, "area_name": area.name,
                    "year": year})
    return rows


def collect(zones=None, years=None, cache_dir=None, quiet=False):
    """Fetch every (zone, year). Returns a long DataFrame.

    A zone-year the platform has nothing for is reported and skipped rather than
    failing the run: coverage genuinely varies, particularly in the earlier years.
    """
    zones = zones or DEFAULT_ZONES
    years = years or DEFAULT_YEARS

    client = TransparencyClient(cache_dir=cache_dir)
    rows, empty = [], []

    for zone in zones:
        for year in years:
            got = fetch_year(client, zone, year)
            if got:
                rows.extend(got)
            else:
                empty.append((zone, year))
            if not quiet:
                total = sum(r["value"] for r in got)
                print(f"  {zone:6} {year}  {len(got):>2} production type(s)"
                      + (f"  {total:>12,.0f} MW total" if got else "  nothing published"))

    if not rows:
        raise TransparencyError(
            "The platform returned no installed capacity for any requested "
            "zone-year. Check the token and that the zones are right.")

    frame = pd.DataFrame(rows)
    frame["production_type"] = frame["psr_type"].map(PSR_TYPES).fillna(
        frame["psr_type"])
    frame = frame[["zone", "area_name", "eic", "year", "psr_type",
                   "production_type", "value", "unit"]]
    frame = frame.sort_values(["zone", "year", "psr_type"]).reset_index(drop=True)
    return frame, empty


def to_wide(frame):
    """One row per zone-year, one column per production type, values in MW.

    Columns follow the platform's own code order rather than appearing
    alphabetically, so the breakdown reads the way its published tables do.
    Absent production types are 0, not blank: a zone that reports a breakdown
    without lignite has no lignite, which is a number rather than a gap.
    """
    wide = frame.pivot_table(index=["zone", "year"], columns="production_type",
                             values="value", aggfunc="sum")

    order = [name for code, name in PSR_TYPES.items()
             if name in wide.columns]
    order += [c for c in wide.columns if c not in order]
    wide = wide[order].fillna(0.0)

    wide.insert(0, "Total", wide.sum(axis=1))
    return wide.reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download installed generation capacity per production type "
                    "[14.1.A] and write it as a table.")
    parser.add_argument("--zones", default=",".join(DEFAULT_ZONES),
                        help=f"Comma-separated zones (default: {','.join(DEFAULT_ZONES)})")
    parser.add_argument("--years", default=",".join(str(y) for y in DEFAULT_YEARS),
                        help=f"Comma-separated years (default: {','.join(str(y) for y in DEFAULT_YEARS)})")
    parser.add_argument("--out", default=os.path.join("datasets", "installed_capacity.csv"),
                        help="Where to write the table")
    parser.add_argument("--long", action="store_true",
                        help="Write tidy long format (one row per zone-year-type) "
                             "instead of the wide table")
    parser.add_argument("--cache-dir", default=".cache",
                        help="Cache raw API responses here (default: .cache)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    zones = [z.strip().upper() for z in args.zones.split(",") if z.strip()]
    try:
        years = [int(y) for y in args.years.split(",") if y.strip()]
    except ValueError:
        print(f"error: --years must be comma-separated integers, got {args.years!r}",
              file=sys.stderr)
        return 1

    for zone in zones:
        try:
            lookup(zone)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if not args.quiet:
        print(f"Installed capacity per production type [14.1.A], "
              f"{len(zones)} zone(s) x {len(years)} year(s)")

    try:
        frame, empty = collect(zones, years, cache_dir=args.cache_dir,
                               quiet=args.quiet)
    except TransparencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out = frame if args.long else to_wide(frame)
    directory = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(directory, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.3f")

    if not args.quiet:
        print(f"\nwritten: {args.out}  ({len(out):,} rows, "
              f"{'long' if args.long else 'wide'} format)")
        if empty:
            print(f"\nno data published for {len(empty)} zone-year(s): "
                  + ", ".join(f"{z} {y}" for z, y in empty))
            print("Coverage genuinely varies -- the earlier years are thinner, and "
                  "not every zone reports every year.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
