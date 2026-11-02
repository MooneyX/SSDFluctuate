#!/usr/bin/env python3
"""Raw GPU<->SSD per-op latency microbenchmark (no sglang involved).

Two data paths are measured with identical offsets / sizes / concurrency:
  gds   SSD <-> GPU memory via cuFile (GPUDirect Storage), driven through KvikIO.
  host  SSD <-> pinned host memory (O_DIRECT) + cudaMemcpy. This is the bounce
        path HiCache uses today and serves as the baseline. Phase times
        (io / xfer) are recorded separately.

Every op is timed individually so latency distributions and jitter over time
can be analysed afterwards (see analyze.py).

Examples:
  python gds_latency.py prepare --file /mnt/nvme/gds_bench.dat --size 64G
  python gds_latency.py run --file /mnt/nvme/gds_bench.dat --out results/demo \
      --paths gds,host --ops read --patterns rand \
      --io-sizes 4K,64K,1M,8M --qd 1,4,16 --duration 20
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

ALIGN = 4096
_UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_size(text: str) -> int:
    t = text.strip().upper().removesuffix("IB").removesuffix("B")
    unit = t[-1] if t and t[-1].isalpha() else ""
    if unit not in _UNITS:
        raise argparse.ArgumentTypeError(f"bad size: {text!r}")
    return int(float(t[: len(t) - len(unit)]) * _UNITS[unit])


def fmt_size(n: int) -> str:
    for unit in ("T", "G", "M", "K"):
        f = _UNITS[unit]
        if n >= f and n % f == 0:
            return f"{n // f}{unit}"
    return str(n)


def csv_list(conv):
    return lambda s: [conv(x) for x in s.split(",") if x.strip()]


# --------------------------------------------------------------------------- prepare


def cmd_prepare(args: argparse.Namespace) -> None:
    size, chunk = parse_size(args.size), parse_size(args.chunk)
    if chunk % ALIGN or size % chunk:
        sys.exit(f"--size must be a multiple of --chunk, and --chunk a multiple of {ALIGN}")
    buf = mmap.mmap(-1, chunk)  # page aligned, required by O_DIRECT
    view = np.frombuffer(buf, dtype=np.uint8)
    rng = np.random.default_rng(args.seed)
    fd = os.open(args.file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
    t0, written, next_report = time.time(), 0, 0
    try:
        while written < size:
            # Real random data: unwritten extents / zero pages could be short-circuited
            # by the filesystem or compressed by the SSD and would distort reads.
            view[:] = np.frombuffer(rng.bytes(chunk), dtype=np.uint8)
            n = os.pwrite(fd, buf, written)
            if n != chunk:
                raise OSError(f"short write {n} at {written}")
            written += n
            if written >= next_report:
                rate = written / max(time.time() - t0, 1e-9) / 2**30
                print(f"\r  prepared {fmt_size(written)} / {fmt_size(size)} ({rate:.2f} GiB/s)",
                      end="", flush=True)
                next_report += 1 << 30
        os.fsync(fd)
    finally:
        os.close(fd)
    print(f"\nprepared {args.file}: {fmt_size(size)} in {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------- run


@dataclass
class Config:
    path: str
    op: str
    pattern: str
    io_size: int
    qd: int

    @property
    def name(self) -> str:
        return f"{self.path}_{self.op}_{self.pattern}_bs{fmt_size(self.io_size)}_qd{self.qd}"


def _kvikio_register(kvikio, buf):
    for name in ("buffer_register", "memory_register"):
        fn = getattr(kvikio, name, None)
        if callable(fn):
            fn(buf)
            return getattr(kvikio, name.replace("register", "deregister"), None)
    return None


def _kvikio_compat_info(kvikio) -> str:
    d = getattr(kvikio, "defaults", None)
    if d is None:
        return "unknown"
    for name in ("compat_mode", "is_compat_mode_preferred"):
        fn = getattr(d, name, None)
        if callable(fn):
            try:
                return f"{name}={fn()}"
            except Exception:  # noqa: BLE001
                pass
    getter = getattr(d, "get", None)
    if callable(getter):
        try:
            return f"compat_mode={getter('compat_mode')}"
        except Exception:  # noqa: BLE001
            pass
    return "unknown"


def build_gds_op(cfg: Config, args, kvikio, torch, dev):
    io = cfg.io_size
    buf = torch.empty(io, dtype=torch.uint8, device=dev)
    if cfg.op == "write":
        buf.random_(0, 256)
    dereg = None
    if not args.no_register:
        try:
            dereg = _kvikio_register(kvikio, buf)
        except Exception as e:  # noqa: BLE001  e.g. BAR1 exhausted on large qd*io
            print(f"  WARN: cuFileBufRegister failed ({e}); continuing unregistered", flush=True)
    fh = kvikio.CuFile(args.file, "r" if cfg.op == "read" else "r+")
    call = fh.pread if cfg.op == "read" else fh.pwrite

    def fn(off: int):
        # task_size=io -> exactly one cuFileRead/Write per op inside KvikIO.
        n = call(buf, io, off, task_size=io).get()
        if n != io:
            raise OSError(f"short gds {cfg.op}: {n} != {io} at {off}")
        return None

    def cleanup():
        fh.close()
        if dereg is not None:
            dereg(buf)

    return fn, cleanup, dereg is not None


def build_host_op(cfg: Config, args, torch, dev):
    io = cfg.io_size
    dbuf = torch.empty(io, dtype=torch.uint8, device=dev)
    if cfg.op == "write":
        dbuf.random_(0, 256)
    raw = torch.empty(io + ALIGN, dtype=torch.uint8, pin_memory=True)
    shift = (-raw.data_ptr()) % ALIGN
    hbuf = raw[shift: shift + io]
    mv = memoryview(hbuf.numpy())
    stream = torch.cuda.Stream(device=dev)
    flags = (os.O_RDONLY if cfg.op == "read" else os.O_RDWR) | os.O_DIRECT
    fd = os.open(args.file, flags)
    clock = time.perf_counter_ns

    if cfg.op == "read":
        def fn(off: int):
            t0 = clock()
            n = os.preadv(fd, [mv], off)
            t1 = clock()
            with torch.cuda.stream(stream):
                dbuf.copy_(hbuf, non_blocking=True)
            stream.synchronize()
            t2 = clock()
            if n != io:
                raise OSError(f"short host read: {n} != {io} at {off}")
            return t1 - t0, t2 - t1
    else:
        def fn(off: int):
            t0 = clock()
            with torch.cuda.stream(stream):
                hbuf.copy_(dbuf, non_blocking=True)
            stream.synchronize()
            t1 = clock()
            n = os.pwritev(fd, [mv], off)
            t2 = clock()
            if n != io:
                raise OSError(f"short host write: {n} != {io} at {off}")
            return t2 - t1, t1 - t0

    return fn, lambda: os.close(fd), False


def run_config(cfg: Config, args, file_size: int, kvikio, torch) -> dict:
    nblocks = file_size // cfg.io_size
    if nblocks < cfg.qd:
        raise ValueError(f"{cfg.name}: file too small for io_size*qd")
    max_ops = max(args.max_ops // cfg.qd, 1)
    times: dict[str, int] = {}

    def set_times():
        now = time.perf_counter_ns()
        times["record"] = now + int(args.warmup * 1e9)
        times["stop"] = times["record"] + int(args.duration * 1e9)

    barrier = threading.Barrier(cfg.qd + 1, action=set_times)
    results: list[dict | None] = [None] * cfg.qd
    errors: list[str] = []

    def worker(tid: int) -> None:
        try:
            torch.cuda.set_device(args.gpu)
            dev = torch.device("cuda", args.gpu)
            if cfg.path == "gds":
                fn, cleanup, registered = build_gds_op(cfg, args, kvikio, torch, dev)
            else:
                fn, cleanup, registered = build_host_op(cfg, args, torch, dev)
            rng = np.random.default_rng(args.seed * 1_000_003 + tid)
            region = nblocks // cfg.qd
            starts, lats, offs, ph_io, ph_xfer = [], [], [], [], []
            clock = time.perf_counter_ns
            seq_i = 0
            barrier.wait()
            t_record, t_stop = times["record"], times["stop"]
            while True:
                if cfg.pattern == "rand":
                    off = int(rng.integers(nblocks)) * cfg.io_size
                else:
                    off = (tid * region + seq_i % region) * cfg.io_size
                    seq_i += 1
                t0 = clock()
                if t0 >= t_stop or len(lats) >= max_ops:
                    break
                phases = fn(off)
                t1 = clock()
                if t0 >= t_record:
                    starts.append(t0 - t_record)
                    lats.append(t1 - t0)
                    offs.append(off)
                    if phases is not None:
                        ph_io.append(phases[0])
                        ph_xfer.append(phases[1])
            cleanup()
            results[tid] = dict(start=starts, lat=lats, off=offs, io=ph_io, xfer=ph_xfer,
                                registered=registered)
        except threading.BrokenBarrierError:
            pass
        except BaseException as e:  # noqa: BLE001
            errors.append(f"worker {tid}: {type(e).__name__}: {e}")
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(cfg.qd)]
    for t in threads:
        t.start()
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError("; ".join(errors))

    tid = np.concatenate([np.full(len(r["lat"]), i, dtype=np.int32) for i, r in enumerate(results)])
    start = np.concatenate([np.asarray(r["start"], dtype=np.int64) for r in results])
    order = np.argsort(start, kind="stable")
    out = dict(
        start_ns=start[order],
        lat_ns=np.concatenate([np.asarray(r["lat"], dtype=np.int64) for r in results])[order],
        offset=np.concatenate([np.asarray(r["off"], dtype=np.int64) for r in results])[order],
        tid=tid[order],
    )
    if cfg.path == "host":
        out["io_ns"] = np.concatenate([np.asarray(r["io"], dtype=np.int64) for r in results])[order]
        out["xfer_ns"] = np.concatenate([np.asarray(r["xfer"], dtype=np.int64) for r in results])[order]
    out["registered_workers"] = sum(bool(r["registered"]) for r in results)
    return out


def quick_stats(cfg: Config, lat_ns: np.ndarray, duration_s: float) -> dict:
    if lat_ns.size == 0:
        return dict(ops=0)
    us = lat_ns / 1e3
    p = np.percentile(us, [50, 99, 99.9])
    return dict(
        ops=int(us.size),
        gib_s=us.size * cfg.io_size / duration_s / 2**30,
        mean_us=float(us.mean()), p50_us=float(p[0]), p99_us=float(p[1]),
        p999_us=float(p[2]), max_us=float(us.max()),
    )


def probe_gds(kvikio, torch, args) -> None:
    buf = torch.empty(ALIGN, dtype=torch.uint8, device=torch.device("cuda", args.gpu))
    fh = kvikio.CuFile(args.file, "r")
    try:
        n = fh.read(buf, ALIGN, 0)
    finally:
        fh.close()
    if n != ALIGN:
        raise OSError(f"probe read returned {n}")


def cmd_run(args: argparse.Namespace) -> None:
    if not os.path.isfile(args.file):
        sys.exit(f"{args.file} not found; run `prepare` first")
    file_size = os.path.getsize(args.file)
    for p in args.paths:
        if p not in ("gds", "host"):
            sys.exit(f"unknown path {p}")
    bad = [s for s in args.io_sizes if s % ALIGN]
    if bad:
        sys.exit(f"io sizes must be multiples of {ALIGN}: {bad}")

    kvikio = None
    if "gds" in args.paths:
        if not args.allow_compat:
            # Fail loudly instead of silently falling back to POSIX + bounce buffers.
            os.environ["KVIKIO_COMPAT_MODE"] = "OFF"
        os.environ.setdefault("KVIKIO_NTHREADS", str(max(args.qd)))
    import torch  # noqa: E402

    torch.cuda.set_device(args.gpu)
    if "gds" in args.paths:
        import kvikio  # noqa: E402

        try:
            probe_gds(kvikio, torch, args)
        except Exception as e:  # noqa: BLE001
            sys.exit(f"GDS probe failed: {type(e).__name__}: {e}\n"
                     "Check `gdscheck -p` / nvidia-fs / filesystem mount options. "
                     "Use --allow-compat to measure anyway (that is NOT a GPUDirect path).")

    os.makedirs(args.out, exist_ok=True)
    env_meta = dict(
        host=socket.gethostname(), time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        file=os.path.abspath(args.file), file_size=file_size, gpu=args.gpu,
        gpu_name=torch.cuda.get_device_name(args.gpu), torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        kvikio=getattr(kvikio, "__version__", None) if kvikio else None,
        kvikio_compat=_kvikio_compat_info(kvikio) if kvikio else None,
        env={k: os.environ.get(k) for k in
             ("KVIKIO_COMPAT_MODE", "KVIKIO_NTHREADS", "CUFILE_ENV_PATH_JSON")},
        warmup_s=args.warmup, duration_s=args.duration, seed=args.seed,
        register=not args.no_register, tag=args.tag,
    )
    with open(os.path.join(args.out, "env_meta.json"), "w") as f:
        json.dump(env_meta, f, indent=2)
    print(json.dumps(env_meta, indent=2))

    configs = [Config(p, op, pat, io, qd)
               for op in args.ops for pat in args.patterns
               for io in args.io_sizes for qd in args.qd for p in args.paths]
    summary_path = os.path.join(args.out, "summary.jsonl")
    for i, cfg in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {cfg.name} ...", flush=True)
        res = run_config(cfg, args, file_size, kvikio, torch)
        stats = quick_stats(cfg, res["lat_ns"], args.duration)
        meta = dict(asdict(cfg), name=cfg.name, registered_workers=res.pop("registered_workers"),
                    warmup_s=args.warmup, duration_s=args.duration, file_size=file_size,
                    gpu_name=env_meta["gpu_name"], tag=args.tag, **stats)
        np.savez_compressed(os.path.join(args.out, f"{cfg.name}.npz"),
                            meta=np.array(json.dumps(meta)), **res)
        with open(summary_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        if stats["ops"]:
            print(f"    ops={stats['ops']} {stats['gib_s']:.2f} GiB/s  "
                  f"p50={stats['p50_us']:.1f}us p99={stats['p99_us']:.1f}us "
                  f"p99.9={stats['p999_us']:.1f}us max={stats['max_us']:.1f}us", flush=True)
        if args.cooldown > 0 and i < len(configs):
            time.sleep(args.cooldown)
    print(f"done -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="create a test file filled with random data (O_DIRECT)")
    p.add_argument("--file", required=True)
    p.add_argument("--size", default="64G", help="file size, should exceed SSD DRAM cache (default 64G)")
    p.add_argument("--chunk", default="64M")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser("run", help="run the latency sweep")
    r.add_argument("--file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--paths", type=csv_list(str), default=["gds", "host"])
    r.add_argument("--ops", type=csv_list(str), default=["read"], help="read,write")
    r.add_argument("--patterns", type=csv_list(str), default=["rand"], help="rand,seq")
    r.add_argument("--io-sizes", type=csv_list(parse_size),
                   default=[parse_size(s) for s in ("4K", "64K", "256K", "1M", "4M", "16M")])
    r.add_argument("--qd", type=csv_list(int), default=[1, 4, 16], help="concurrent outstanding ops")
    r.add_argument("--duration", type=float, default=20.0, help="recorded seconds per config")
    r.add_argument("--warmup", type=float, default=3.0, help="discarded seconds per config")
    r.add_argument("--cooldown", type=float, default=2.0, help="idle seconds between configs")
    r.add_argument("--max-ops", type=int, default=2_000_000, help="cap on recorded ops per config")
    r.add_argument("--gpu", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--tag", default="")
    r.add_argument("--allow-compat", action="store_true",
                   help="allow KvikIO compat (POSIX) fallback; results are then NOT GPUDirect")
    r.add_argument("--no-register", action="store_true", help="skip cuFileBufRegister")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    if getattr(args, "ops", None):
        for op in args.ops:
            if op not in ("read", "write"):
                sys.exit(f"unknown op {op}")
        for pat in args.patterns:
            if pat not in ("rand", "seq"):
                sys.exit(f"unknown pattern {pat}")
    args.func(args)


if __name__ == "__main__":
    main()
