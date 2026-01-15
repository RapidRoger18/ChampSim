#!/usr/bin/env bash
set -euo pipefail

############################
# Usage
############################
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <config.json> <tlist> [tag]"
  echo "Example:"
  echo "  $0 configs/singlecore_rl.json ligra_full.tlist ligra_sc"
  exit 1
fi

CONFIG_JSON=$1
TLIST=$2
TAG=${3:-default}

############################
# Paths
############################
CHAMPSIM_BIN=${CHAMPSIM_BIN:-./bin/champsim}
CONFIG_SH=${CONFIG_SH:-./config.sh}
RESULT_DIR=${RESULT_DIR:-./results}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRACES_HOME="${TRACES_HOME:-$SCRIPT_DIR/../traces}"
export TRACES_HOME

############################
# Sanity checks
############################
[ -f "$CONFIG_JSON" ] || { echo "[ERROR] Config not found"; exit 1; }
[ -f "$TLIST" ] || { echo "[ERROR] Tlist not found"; exit 1; }
[ -x "$CONFIG_SH" ] || { echo "[ERROR] config.sh not executable"; exit 1; }

if [ ! -d "$TRACES_HOME" ]; then
  echo "[ERROR] TRACES_HOME not found: $TRACES_HOME"
  exit 1
fi

############################
# Generate config + build
############################
echo ">>> Generating config from $CONFIG_JSON"
"$CONFIG_SH" "$CONFIG_JSON"

echo ">>> Building ChampSim"
make -j$(nproc)

############################
# Run traces
############################
OUTDIR="$RESULT_DIR/$TAG"
mkdir -p "$OUTDIR"

echo ">>> Running tlist: $TLIST"

# Print header (first comment block)
echo ">>> Tlist header:"
sed -n '1,20p' "$TLIST" | sed -n '/^#/p'
echo "----------------------------------------"

unset NAME TRACE KNOBS

while IFS= read -r line || [ -n "$line" ]; do
  # Skip empty lines
  [[ -z "$line" ]] && continue

  # Skip comments
  [[ "$line" =~ ^# ]] && continue

  # Evaluate variable assignment (NAME / TRACE / KNOBS)
  eval "$line"

  # Run once we have a full block
  if [[ -n "${NAME:-}" && -n "${TRACE:-}" ]]; then
    echo "  -> Running $NAME"
    echo "     Trace: $TRACE"
    echo "     Knobs: ${KNOBS:-<none>}"

    "$CHAMPSIM_BIN" \
      ${KNOBS:-} \
      "$TRACE" \
      > "$OUTDIR/$NAME.log" 2>&1

    # Reset for next trace block
    unset NAME TRACE KNOBS
  fi
done < "$TLIST"

echo ">>> Done"
