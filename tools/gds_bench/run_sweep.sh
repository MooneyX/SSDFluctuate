#!/usr/bin/env bash
# One-shot raw GPU<->SSD latency sweep; run on the GPU server.
# Usage: run_sweep.sh <test_file_on_nvme> [out_dir]
# Env overrides (defaults in brackets):
#   PATHS[gds,host] OPS[read] PATTERNS[rand] IO_SIZES[4K,64K,256K,1M,4M,16M] QD_LIST[1,4,16]
#   DURATION[20] WARMUP[3] COOLDOWN[2] GPU[0] PREPARE_SIZE[64G] STRICT_GDS[1] GDSIO[1]
#   PYTHON[python3] GDS_TOOLS[/usr/local/cuda/gds/tools]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
FILE=${1:?usage: run_sweep.sh <test_file_on_nvme> [out_dir]}
OUT=${2:-$ROOT/results/$(date +%Y%m%d_%H%M%S)_gds_raw}
PATHS=${PATHS:-gds,host}
OPS=${OPS:-read}
PATTERNS=${PATTERNS:-rand}
IO_SIZES=${IO_SIZES:-4K,64K,256K,1M,4M,16M}
QD_LIST=${QD_LIST:-1,4,16}
DURATION=${DURATION:-20}
WARMUP=${WARMUP:-3}
COOLDOWN=${COOLDOWN:-2}
GPU=${GPU:-0}
PREPARE_SIZE=${PREPARE_SIZE:-64G}
STRICT_GDS=${STRICT_GDS:-1}
PY=${PYTHON:-python3}
GDS_TOOLS=${GDS_TOOLS:-/usr/local/cuda/gds/tools}
export PYTHON=$PY GDS_TOOLS

mkdir -p "$OUT"
echo "results -> $OUT"

if [ ! -f "$FILE" ]; then
  echo "test file missing, preparing $PREPARE_SIZE of random data ..."
  "$PY" "$HERE/gds_latency.py" prepare --file "$FILE" --size "$PREPARE_SIZE"
fi

# cuFile itself may silently fall back to compat (bounce-buffer) mode; forbid it so that
# "gds" numbers are guaranteed to be real GPUDirect transfers.
if [ "$STRICT_GDS" = 1 ]; then
  SRC=${CUFILE_ENV_PATH_JSON:-/etc/cufile.json}
  if [ -r "$SRC" ]; then
    sed -E 's/("allow_compat_mode"[[:space:]]*:[[:space:]]*)true/\1false/' "$SRC" >"$OUT/cufile.strict.json"
    grep -qE '"allow_compat_mode"[[:space:]]*:[[:space:]]*false' "$OUT/cufile.strict.json" \
      || echo "WARN: allow_compat_mode not found in $SRC; cuFile default applies" >&2
    export CUFILE_ENV_PATH_JSON="$OUT/cufile.strict.json"
  else
    echo "WARN: $SRC not readable; cuFile may silently use compat mode" >&2
  fi
fi
export CUFILE_LOGFILE_PATH="$OUT/cufile.log"

bash "$HERE/env_check.sh" "$FILE" "$OUT/env.txt"

"$PY" "$HERE/gds_latency.py" run --file "$FILE" --out "$OUT" --gpu "$GPU" \
  --paths "$PATHS" --ops "$OPS" --patterns "$PATTERNS" --io-sizes "$IO_SIZES" --qd "$QD_LIST" \
  --duration "$DURATION" --warmup "$WARMUP" --cooldown "$COOLDOWN" 2>&1 | tee "$OUT/run.log"

# Native cross-check without Python overhead (gdsio reports mean latency only).
if [ "${GDSIO:-1}" = 1 ] && [ -x "$GDS_TOOLS/gdsio" ] && [ "$OPS" = read ]; then
  case "$PATTERNS" in rand) IOTYPE=2 ;; seq) IOTYPE=0 ;; *) IOTYPE= ;; esac
  if [ -n "$IOTYPE" ]; then
    FILE_KB=$(( $(stat -c %s "$FILE") / 1024 ))
    for io in ${IO_SIZES//,/ }; do
      for qd in ${QD_LIST//,/ }; do
        for x in 0 2; do  # 0 = GPU_DIRECT, 2 = CPU_GPU (bounce)
          echo "### gdsio io=$io qd=$qd xfer=$x" | tee -a "$OUT/gdsio.log"
          "$GDS_TOOLS/gdsio" -f "$FILE" -d "$GPU" -w "$qd" -s "${FILE_KB}K" -i "$io" \
            -x "$x" -I "$IOTYPE" -T "$DURATION" 2>&1 | tee -a "$OUT/gdsio.log" || true
        done
      done
    done
  fi
fi

"$PY" "$HERE/analyze.py" "$OUT" | tee "$OUT/analyze.log"
