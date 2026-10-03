#!/bin/bash
#
# Quant Research Platform — one-click dashboard launcher (macOS).
#
# Double-click this file. It sets everything up automatically and opens the
# dashboard in your browser. No Terminal commands needed.
#
# What it does, in plain English:
#   1. Installs "uv" (a fast, reliable Python installer) if you don't have it.
#   2. Creates a private Python environment for this project (in .venv).
#      It downloads its own Python — your Mac's Python is never touched.
#   3. Installs the project's libraries into that environment (first run only).
#   4. Starts the dashboard and opens it in your browser.
#
# To stop the dashboard: come back to this window and press Ctrl+C.

set -euo pipefail

say()  { printf '\n%s\n' "$1"; }
fail() {
  printf '\n%s\n' "❌ $1"
  [ -n "${2:-}" ] && printf '%s\n' "$2"
  printf '\nPress Enter to close this window.\n'
  read -r _
  exit 1
}

# Always run from the folder this file lives in (the project folder).
cd "$(cd "$(dirname "$0")" && pwd)"

# Never use the `curl` from PATH: package managers (notably Anaconda) can
# shadow it with a broken build (wrong CPU architecture), which crashes on
# launch. macOS always ships its own working curl at /usr/bin/curl.
if [ -x /usr/bin/curl ]; then
  CURL=/usr/bin/curl
else
  CURL=curl
fi

say "📈  Quant Research Platform — starting up…"

# --- Step 1: make sure "uv" is available -------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  [ -x "$HOME/.local/bin/uv" ] && export PATH="$HOME/.local/bin:$PATH"
fi
if ! command -v uv >/dev/null 2>&1; then
  say "First-time setup: downloading 'uv' (the official installer from astral.sh — safe to approve)…"
  "$CURL" -LsSf https://astral.sh/uv/install.sh | sh \
    || fail "I couldn't download the installer." \
            "Check your internet connection, then double-click 'Run Dashboard' again."
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 \
  || fail "I couldn't find 'uv' even after installing it." \
          "Try restarting your Mac, then double-click 'Run Dashboard' again."

# --- Step 2: private Python environment (never touches your Mac's Python) ----
if [ ! -x ".venv/bin/python" ]; then
  say "First-time setup: creating a private Python for this project (downloading — a minute or two)…"
  uv venv --python 3.12 .venv \
    || fail "I couldn't create the project's Python environment." \
            "Check your internet connection, then double-click 'Run Dashboard' again."
else
  say "Project Python found — reusing it."
fi

# --- Step 3: install the project's libraries (only when they changed) --------
REQ_HASH="$(shasum -a 256 requirements.txt | cut -d' ' -f1)"
if [ -f ".venv/.qrp-requirements.sha256" ] && [ "$(cat .venv/.qrp-requirements.sha256)" = "$REQ_HASH" ]; then
  say "Libraries are already installed — skipping."
else
  say "Installing the project's libraries (first run only — a couple of minutes)…"
  uv pip install --python .venv/bin/python -r requirements.txt \
    || fail "I couldn't install the project's libraries." \
            "Check your internet connection, then double-click 'Run Dashboard' again."
  printf '%s' "$REQ_HASH" > .venv/.qrp-requirements.sha256
fi

if [ ! -f "data/market.db" ]; then
  say "Note: you haven't downloaded any market data yet, so the dashboard will look empty at first. Use the 'Market data' page in the dashboard to download it."
fi

# --- Step 4: start the dashboard ----------------------------------------------
say "Starting the dashboard…"
PORT=8501
.venv/bin/python -m streamlit run qrp/ui/app.py --server.port "$PORT" --server.headless true \
  >/tmp/qrp-dashboard.log 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 60); do
  if "$CURL" -sf -o /dev/null "http://127.0.0.1:${PORT}/"; then
    break
  fi
  sleep 1
done
if ! "$CURL" -sf -o /dev/null "http://127.0.0.1:${PORT}/"; then
  kill "$SERVER_PID" 2>/dev/null || true
  fail "The dashboard didn't start." \
       "Something unexpected went wrong. The details are in /tmp/qrp-dashboard.log — send that file to Nene."
fi

say "✅  The dashboard is running! Opening your browser…"
open "http://127.0.0.1:${PORT}/"
say "Leave this window open while you use the dashboard. When you're done, press Ctrl+C here."
wait "$SERVER_PID"
