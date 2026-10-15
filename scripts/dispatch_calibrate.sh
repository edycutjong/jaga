#!/usr/bin/env bash
# The calibration Job (Nebius AI Cloud Serverless Job): run by hand, or from a scheduler such as a launchd agent once
# labelled margins exist (none is installed by this repo). A scheduler starts with a bare environment, so PATH is set
# here so the `nebius` CLI and this repo's venv are found.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="$HOME/.nebius/bin:$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:$PATH"
export NEBIUS_PROFILE="${NEBIUS_PROFILE:-default}"
cd "$REPO"
exec "$REPO/.venv/bin/python" -m jaga.cli calibrate "$@"
