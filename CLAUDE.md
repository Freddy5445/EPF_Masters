# EPF_Masters

Electricity price forecasting — master's thesis code.

## Before touching the Python environment, read `SETUP.md`

This project sits in a OneDrive-synced folder. A `.venv/` appearing inside the
project is **stale debris synced from another machine** — venvs are not
relocatable and cannot be shared. Never activate or repair one; build a fresh
local venv outside the OneDrive folder instead.

`SETUP.md` has the full procedure and the reasoning. Key points:

- Python **3.11** (not 3.13). Check with `py -0p`.
- Venv lives outside OneDrive, e.g. `C:\Users\<you>\venvs\EPF_Masters`.
- `requirements.txt` is a full pin of direct and transitive dependencies.
- `epftoolbox` installs from the local `./epftoolbox` source tree, not PyPI —
  which is why it is absent from `requirements.txt`.

## ENTSO-E API token

`entsoe_tp/client.py` reads `ENTSOE_API_TOKEN` from the environment, falling
back to a `.env` file in the project root. Write it **unquoted** — the parser
does a plain `partition("=")` and does not strip quotes.

`.env` is gitignored and never committed.
