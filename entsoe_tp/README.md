# entsoe_tp

Builds `epftoolbox`-compatible datasets from the [ENTSO-E Transparency Platform][tp]
RESTful API, for any European bidding zone and date range.

Self-contained: nothing here imports or modifies the vendored `epftoolbox/` tree or
`get_nordpool_data.py`.

## Getting a security token

API access is not automatic — it takes a few days.

1. Register at <https://transparency.entsoe.eu/> and verify the email link.
2. Email `transparency@entsoe.eu` with **"RESTful API access"** as the subject and your
   registered address in the body.
3. Access is granted within about three working days; you'll get a confirmation email.
4. Log in, open **My Account**, and generate a security token.

Then put it in a `.env` file in the project root (already gitignored):

```
ENTSOE_API_TOKEN=your-token-here
```

or set the `ENTSOE_API_TOKEN` environment variable. The token is only ever sent as a
query parameter to the platform, and is stripped from any error message the client raises.

## Usage

```bash
python -m entsoe_tp.build_dataset --zone DK1 --start 2016-01-01 --end 2024-12-31
```

Writes `datasets/DK1.csv`, which loads straight into the toolbox:

```python
from epftoolbox.data import read_data
df_train, df_test = read_data(path='datasets', dataset='DK1', years_test=1)
```

`read_data` treats any dataset name outside its five built-ins as a plain local file, so
no registration step is needed.

| Flag | Meaning |
|---|---|
| `--zone` | Bidding zone code — `DK1`, `DK2`, `SE1`–`SE4`, `NO1`–`NO5`, `FI`, `DE_LU`, `FR`, `BE`, `NL`, … (see `areas.py`) |
| `--start`, `--end` | Inclusive local calendar days, `YYYY-MM-DD` |
| `--out` | Output path (default `datasets/<ZONE>.csv`) |
| `--exog` | Exogenous layout: `load-windsolar` (2 columns, default) or `load-wind-solar` (3 columns) |
| `--no-cache` | Re-download instead of reusing cached raw XML |
| `--max-gap` | Longest run of missing hours to interpolate (default 3) |

## What it fetches

With `--exog load-windsolar` (the default):

| Output column | Data item | Query |
|---|---|---|
| `Price` | Day-ahead prices [12.1.D] | `documentType=A44` |
| `Exogenous 1` | Day-ahead total load forecast [6.1.B] | `A65` + `processType=A01` |
| `Exogenous 2` | Day-ahead wind & solar forecast [14.1.D] | `A69` + `processType=A01` |

With `--exog load-wind-solar`, the renewables are split by production type:

| Output column | Data item | Query |
|---|---|---|
| `Price` | Day-ahead prices [12.1.D] | `documentType=A44` |
| `Exogenous 1` | Day-ahead total load forecast [6.1.B] | `A65` + `processType=A01` |
| `Exogenous 2` | Day-ahead wind forecast, on- + offshore | `A69`, `psrType` B18 + B19 |
| `Exogenous 3` | Day-ahead solar forecast | `A69`, `psrType` B16 |

Wind and solar arrive in the *same* A69 document as separate `TimeSeries`, so
both columns come from one query that is fetched once and split afterwards. If a
zone does not publish a requested production type the build fails loudly rather
than emitting a column of zeros.

`--include-reservoir` appends a further column from Water Reservoirs and Hydro
Storage Plants [16.1.D] (`A72` + `processType=A16`):

| Output column | Data item | Query |
|---|---|---|
| last | Weekly stored energy | `A72` + `processType=A16` |

Two things make this column different from the others. It is **weekly**, not
hourly, and it is a **stock** (a level) rather than a flow. It is therefore held
constant between publications rather than interpolated — interpolating would
invent a trajectory the data does not contain, and would read from the *next*
observation, which is future information.

It is also delayed: a value covering a week cannot be known before that week has
ended, so each observation only influences the grid from 7 days after the period
it describes. `--reservoir-lag-days` adds more on top, to model the platform's
own reporting delay. Hours before the first available publication stay empty
rather than being back-filled.

Note the exogenous count drives LEAR's feature count (`96 + 7 + 72·n`), which in
turn sets the shortest usable calibration window — see `lear_dk1/README.md`.
**Adding the reservoir to the 3-exogenous layout makes 4, so 391 features and a
minimum window of about 399 days**, which rules out the 364-day window in the
default LEAR ensemble.

## Re-running only downloads what is missing

Two caches make adding a column cheap:

- **Raw XML** is keyed on request parameters, so adding a data item fetches only
  that item; the others are served from `.cache/entsoe/`.
- **Finished columns** are cached under `.cache/entsoe/columns/`, keyed on
  everything that shapes them — zone, query, date range, aggregation, production
  types, gap handling and reservoir lag. Adding one column to an existing
  dataset therefore costs one download and one parse, instead of re-deriving
  every column from a decade of cached XML.

So this re-uses four columns and downloads only `A72`:

```
python -m entsoe_tp.build_dataset --zone NO2 --start 2016-01-01 --end 2025-04-07 --exog load-wind-solar --include-reservoir
```

Pass `--refresh-columns` to recompute columns from cached XML, or `--no-cache`
to re-download everything.

## Sub-hourly publication is folded, not truncated

Zones began publishing `PT15M` documents months before the day-ahead auction
actually cleared sub-hourly — NO2 in February 2025, DK1 in April, against an
EU-wide deadline of 2025-10-01. Throughout that period the four quarters of an
hour repeat a single value, so folding them to hourly is **lossless**, and
stopping the range at the first `PT15M` document would discard months of good
data.

So the values decide, not the declared resolution:

- quarters that are identical within the hour are folded, and the build
  continues;
- where they genuinely differ, the build stops and names the last clean day,
  because averaging real intra-hour variation is a modelling choice rather than
  a parsing one.

`--stop-at-resolution-change` truncates to that day automatically instead of
failing — useful across zones, since each has its own boundary.

Both exogenous series are *forecasts* published day-ahead, so they are genuinely
available when a day-ahead price forecast has to be made. Realised load and generation
are not, and are deliberately not used.

`read_data` assigns column names **positionally** — the first column after the index
becomes `Price`, the rest become `Exogenous 1..N` — so the header names written here are
for human readers only; the order is what matters.

## Raw multi-zone dump

`build_dataset` produces a *modelling* dataset and enforces what the epftoolbox
models need. To study the data itself — how resolution, availability and
completeness change across zones and over time — use the raw dump instead:

```
python -m entsoe_tp.raw_dump
```

Covers all 15 Nordic and Baltic zones (`DK1`, `DK2`, `NO1`–`NO5`, `SE1`–`SE4`,
`FI`, `EE`, `LV`, `LT`) from 2016-01-01 to 2025-10-01 inclusive, and deliberately
does the opposite of `build_dataset`:

| `build_dataset` | `raw_dump` |
|---|---|
| Strict hourly grid, 24 rows per local day | No grid; rows exist only where data was published |
| Refuses to mix `PT60M` and `PT15M` | Native resolution kept, recorded per observation |
| Naive local time, DST folded | UTC as published — no folded or missing hour |
| Interpolates or aborts on gaps | Neither; absence is absence |
| Aborts if a production type is missing | Records it in the coverage manifest and continues |

Output is one long/tidy Parquet file (`datasets/nordic_baltic_raw.parquet`) with
one row per observation, carrying `resolution`, `psr_type`, `curve_type`,
`business_type`, `unit` and `currency` alongside the value, plus a
`*_manifest.csv` recording per zone and data item what was returned, which
resolutions appeared, and what failed.

Per-zone parts are written under `<out>.parts/`, so an interrupted run resumes
by zone; pass `--refresh` to re-fetch. Needs `pyarrow`.

The full run is roughly 5,800 requests and several million observations — expect
tens of minutes on a cold cache.

## Things this handles that are easy to get wrong

**`curveType=A03`.** The platform's default encoding publishes a `Point` only when the
value *changes*; a gap in the `position` sequence means the previous value continues.
This is not an edge case — ENTSO-E's own documented example response for 12.1.D carries
95 points for a 96-slot day. Reading points as a dense list drops intervals and shifts
everything after the gap earlier. `parser.py` expands each period across the full slot
range implied by its `timeInterval` and `resolution`, carrying values forward.

**Rejections return HTTP 200.** "No matching data found" comes back as an
`Acknowledgement_MarketDocument` with reason code 999, not an HTTP error. Code 999 is
treated as an empty result so that a gap mid-range doesn't abort a multi-year download;
any other reason raises.

**DST.** ENTSO-E publishes in UTC, but `epftoolbox` needs naive local market time with
exactly 24 rows per calendar day. `hourly.py` reproduces the convention the shipped
`datasets/NP.csv` uses: the two fall-back 02:00 hours are averaged into one row, and the
spring-forward 02:00 is interpolated from its neighbours. The invariant is asserted
before writing, because the downstream symptoms — a `KeyError` deep in feature
construction, or a `reshape(-1, 24)` failure — are much harder to trace back.

**15-minute market time units.** Day-ahead resolution moved from 60 to 15 minutes on
2025-10-01. The parser reads `resolution` per period and raises on anything that isn't
`PT60M`; `--end` is refused at or past that date. Extending past it means deciding how to
aggregate to hourly, which is a modelling choice, not a parsing one.

**Rate limits.** 400 requests/minute per *token* (not per IP); exceeding it bans the
token for about ten minutes. The client throttles to ~6 requests/second, chunks requests
by month, and paginates with `offset` when a response returns a full 100 `TimeSeries`.

## Tests

Offline, no token or network needed:

```bash
.venv/Scripts/python -m unittest discover -s tests -v
```

## Caching

Raw XML responses are cached under `.cache/entsoe/` (gitignored), keyed by a hash of the
request parameters with the token excluded. This is purely a re-run accelerator — the CSV
is the only output. Use `--no-cache` to bypass it.

[tp]: https://transparency.entsoe.eu/
