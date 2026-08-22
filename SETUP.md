# Environment setup (read this before touching the Python environment)

Note for a Claude Code session running on a machine other than the one where
this file was written. It explains how to get a working Python environment for
this project, and one trap to avoid.

## The trap: do not reuse a synced `.venv`

This project lives in a OneDrive-synced folder. **A virtual environment cannot
be shared between machines**, even though OneDrive will happily sync one.

A venv contains no interpreter. It contains a shim plus a `pyvenv.cfg` holding
an *absolute path* to the base interpreter that created it:

```
home = C:\Program Files\Python311
executable = C:\Program Files\Python311\python.exe
```

On a machine where Python 3.11 sits anywhere else, that shim fails with:

```
No Python at '"C:\Program Files\Python311\python.exe'
```

`Scripts\*.exe` entry points and `.dist-info/RECORD` files hardcode paths too,
so patching `pyvenv.cfg` alone is a partial fix at best. Python's own docs state
venvs are not relocatable.

**So:** if you find a `.venv/` inside this project folder, treat it as stale
debris from another machine. Do not activate it, do not repair it. Build a fresh
local one as below. `.venv/` is already in `.gitignore`.

## Build a local venv

Keep the venv **outside the OneDrive folder** so it is never synced. On the
machine this was written from, it lives at `C:\Users\Frede\venvs\EPF_Masters`.
Use the equivalent path for the current machine.

1. Confirm Python 3.11 is present and note its path:

   ```powershell
   py -0p
   ```

   The project pins Python 3.11 (originally built on 3.11.6; 3.11.9 works —
   patch versions are ABI-compatible, minor versions are not). If 3.11 is
   missing:

   ```powershell
   winget install --id Python.Python.3.11 --scope user --silent --accept-package-agreements --accept-source-agreements
   ```

2. Create the venv outside OneDrive:

   ```powershell
   py -3.11 -m venv C:\Users\<you>\venvs\EPF_Masters
   ```

3. Install dependencies. `requirements.txt` is a full pin of every package —
   direct and transitive — captured from a known-working environment:

   ```powershell
   & C:\Users\<you>\venvs\EPF_Masters\Scripts\python.exe -m pip install --upgrade pip
   & C:\Users\<you>\venvs\EPF_Masters\Scripts\python.exe -m pip install -r requirements.txt
   ```

   This pulls TensorFlow and SciPy, so expect a multi-hundred-MB download.

4. Install `epftoolbox` **from the local source tree**, not PyPI. The
   `epftoolbox/` directory in this repo is the package source, and the pinned
   `epftoolbox==1.0` refers to it:

   ```powershell
   & C:\Users\<you>\venvs\EPF_Masters\Scripts\python.exe -m pip install .\epftoolbox
   ```

   It is deliberately absent from `requirements.txt` for this reason.

5. Point VS Code at the interpreter — `.vscode/settings.json` sets
   `python.defaultInterpreterPath`, but that path is machine-specific and syncs
   through OneDrive. It is only a *default*, so selecting the interpreter
   explicitly in VS Code (`Python: Select Interpreter`) overrides it. Update the
   file to this machine's path, or select it in the UI.

## Verify

```powershell
& C:\Users\<you>\venvs\EPF_Masters\Scripts\python.exe -c "import tensorflow, epftoolbox, pandas; print('ok')"
```

## The ENTSO-E API token

`entsoe_tp/client.py` reads a security token from `ENTSOE_API_TOKEN`, checking
the environment variable first, then a `.env` file in the project root:

```
ENTSOE_API_TOKEN=<token>
```

No quotes — the parser does a plain `partition("=")` and does not strip them, so
quotes become part of the token and auth fails with HTTP 401.

`.env` is gitignored. It currently reaches other machines via OneDrive sync; if
this project ever moves to git-only sync, each machine needs its own `.env`.

## What is portable, and what is not

| Portable (commit / sync) | Machine-local (rebuild) |
| --- | --- |
| `requirements.txt` | `.venv/` — never share |
| `epftoolbox/` source | Interpreter install path |
| Project code | `.vscode/settings.json` interpreter path |

`.cache/`, `__pycache__/`, and `*.csv` are gitignored build/data artifacts and
do not need to travel either.
