#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
exec python -m casemem --config "$root/config/casemem.json" train "$@"
