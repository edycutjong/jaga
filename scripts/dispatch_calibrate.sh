#!/usr/bin/env bash
# Nightly calibration Job (Nebius AI Cloud Serverless Job). Called by hand or by the launchd agent dev.jaga.calibrate.
# launchd starts with a bare environment: set PATH so the `nebius` CLI and this repo's venv are found.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.nebius/bin:$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
export NEBIUS_PROFILE="${NEBIUS_PROFILE:-default}"
cd "$REPO"
exec "$REPO/.venv/bin/python" -m jaga.cli calibrate "$@"
