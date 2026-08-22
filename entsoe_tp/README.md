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

Note the exogenous count drives LEAR's feature count (`96 + 7 + 72·n`), which in
turn sets the shortest usable calibration window — see `lear_dk1/README.md`.

Both exogenous series are *forecasts* published day-ahead, so they are genuinely
available when a day-ahead price forecast has to be made. Realised load and generation
are not, and are deliberately not used.

`read_data` assigns column names **positionally** — the first column after the index
becomes `Price`, the rest become `Exogenous 1..N` — so the header names written here are
for human readers only; the order is what matters.

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
