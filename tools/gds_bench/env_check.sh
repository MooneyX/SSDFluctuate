#!/usr/bin/env bash
# Collect GDS / NVMe / PCIe topology info needed to interpret raw GPU<->SSD latency.
# Usage: env_check.sh <test_file> [out_file]
set -u
FILE=${1:?usage: env_check.sh <test_file> [out_file]}
OUT=${2:-/dev/stdout}
GDS_TOOLS=${GDS_TOOLS:-/usr/local/cuda/gds/tools}
PY=${PYTHON:-python3}

section() { printf '\n===== %s =====\n' "$*"; }
run() { echo "\$ $*"; "$@" 2>&1 || echo "(exit $?)"; }

{
  section host
  run uname -a
  run date -Is

  section gpu
  run nvidia-smi
  run nvidia-smi topo -m
  run nvidia-smi -q -d MEMORY   # BAR1 size limits how much can be cuFileBufRegister'ed

  section nvidia-fs
  run bash -c 'lsmod | grep -E "^(nvidia_fs|nvme|nvme_core) "'
  [ -r /proc/driver/nvidia-fs/stats ] && run cat /proc/driver/nvidia-fs/stats

  section gdscheck
  if [ -x "$GDS_TOOLS/gdscheck" ]; then run "$GDS_TOOLS/gdscheck" -p
  elif command -v gdscheck >/dev/null; then run gdscheck -p
  elif [ -x "$GDS_TOOLS/gdscheck.py" ]; then run "$GDS_TOOLS/gdscheck.py" -p
  else echo "gdscheck not found (set GDS_TOOLS)"; fi

  section filesystem
  run findmnt -T "$FILE" -o TARGET,SOURCE,FSTYPE,OPTIONS
  DEV=$(findmnt -n -T "$FILE" -o SOURCE 2>/dev/null)
  run lsblk -o NAME,MODEL,SIZE,ROTA,TYPE,MOUNTPOINT
  command -v nvme >/dev/null && run nvme list
  BASE=$(lsblk -no PKNAME "$DEV" 2>/dev/null | head -n1)
  [ -z "$BASE" ] && BASE=$(basename "${DEV:-unknown}")
  echo "block device: $BASE"
  run readlink -f "/sys/block/$BASE"   # PCIe path of the NVMe device
  for f in scheduler nr_requests max_sectors_kb write_cache; do
    [ -r "/sys/block/$BASE/queue/$f" ] && echo "queue/$f: $(cat "/sys/block/$BASE/queue/$f")"
  done
  if command -v nvme >/dev/null && [ -e "/dev/$BASE" ]; then
    run nvme smart-log "/dev/$BASE"
  fi

  section cufile.json
  CJ=${CUFILE_ENV_PATH_JSON:-/etc/cufile.json}
  echo "CUFILE_ENV_PATH_JSON=${CUFILE_ENV_PATH_JSON:-<unset>} (using $CJ)"
  [ -r "$CJ" ] && grep -nE '"(allow_compat_mode|use_poll_mode|poll_mode_max_size_kb|max_direct_io_size_kb|max_device_cache_size_kb|max_device_pinned_mem_size_kb|use_pci_p2pdma|io_batchsize)"' "$CJ"

  section python
  run "$PY" - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
try:
    import kvikio
    print("kvikio", getattr(kvikio, "__version__", "?"))
except Exception as e:
    print("kvikio import failed:", e)
EOF
} >"$OUT"
[ "$OUT" != /dev/stdout ] && echo "env info -> $OUT"
exit 0
