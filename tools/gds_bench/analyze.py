#!/usr/bin/env python3
"""Summarize and plot results written by `gds_latency.py run`.

Outputs (into the result directory unless --plot-dir is given):
  summary.csv                      percentiles / throughput per config (+ host phase split)
  cdf_<op>_<pattern>_bs<size>.png  CDF and tail CCDF, gds vs host, one curve per qd
  tail_vs_size_<op>_<pattern>.png  p50 / p99 / p99.9 vs io size
  timeline/<config>.png            per-op latency over time with windowed p50 / p99

Usage: python analyze.py <result_dir> [--window-ms 100] [--no-timeline]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

from gds_latency import fmt_size

PCTS = [50, 90, 99, 99.9, 99.99]
PCT_KEYS = ["p50", "p90", "p99", "p99.9", "p99.99"]
PATH_COLOR = {"gds": "tab:red", "host": "tab:blue"}
QD_STYLE = ["-", "--", ":", "-."]


def load(path: str) -> dict:
    with np.load(path) as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def pct(us: np.ndarray) -> dict:
    if us.size == 0:
        return {k: float("nan") for k in PCT_KEYS}
    return dict(zip(PCT_KEYS, map(float, np.percentile(us, PCTS))))


def summarize(d: dict) -> dict:
    m = d["meta"]
    lat = d["lat_ns"] / 1e3
    elapsed = m["duration_s"]
    row = dict(path=m["path"], op=m["op"], pattern=m["pattern"], io_size=fmt_size(m["io_size"]),
               io_bytes=m["io_size"], qd=m["qd"], ops=int(lat.size),
               gib_s=lat.size * m["io_size"] / elapsed / 2**30 if lat.size else 0.0,
               mean_us=float(lat.mean()) if lat.size else float("nan"),
               std_us=float(lat.std()) if lat.size else float("nan"),
               max_us=float(lat.max()) if lat.size else float("nan"),
               registered_workers=m.get("registered_workers"))
    row.update({f"{k}_us": v for k, v in pct(lat).items()})
    row["p99_over_p50"] = row["p99_us"] / row["p50_us"] if lat.size else float("nan")
    for phase in ("io", "xfer"):
        key = f"{phase}_ns"
        if key in d and d[key].size:
            p = pct(d[key] / 1e3)
            row[f"{phase}_p50_us"], row[f"{phase}_p99_us"] = p["p50"], p["p99"]
    return row


def write_csv(rows: list[dict], path: str) -> None:
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def plot_cdf(group: list[dict], title: str, out: str, plt) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    qds = sorted({d["meta"]["qd"] for d in group})
    for d in sorted(group, key=lambda d: (d["meta"]["path"], d["meta"]["qd"])):
        m, lat = d["meta"], np.sort(d["lat_ns"] / 1e3)
        if lat.size == 0:
            continue
        n = lat.size
        style = dict(color=PATH_COLOR.get(m["path"]), linestyle=QD_STYLE[qds.index(m["qd"]) % 4],
                     label=f"{m['path']} qd={m['qd']}")
        ax1.plot(lat, np.arange(1, n + 1) / n, **style)
        ax2.plot(lat, 1.0 - np.arange(n) / n, **style)
    ax1.set(xscale="log", xlabel="latency (us)", ylabel="CDF", title=title)
    ax2.set(xscale="log", yscale="log", xlabel="latency (us)", ylabel="P(latency > x)",
            title="tail (CCDF)")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_tail_vs_size(rows: list[dict], title: str, out: str, plt) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    qds = sorted({r["qd"] for r in rows})
    for ax, key in zip(axes, ("p50_us", "p99_us", "p99.9_us")):
        series = defaultdict(list)
        for r in rows:
            series[(r["path"], r["qd"])].append((r["io_bytes"], r[key]))
        for (path, qd), pts in sorted(series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                    color=PATH_COLOR.get(path), linestyle=QD_STYLE[qds.index(qd) % 4],
                    label=f"{path} qd={qd}")
        sizes = sorted({r["io_bytes"] for r in rows})
        ax.set(xscale="log", yscale="log", xlabel="io size", title=f"{title}: {key[:-3]}")
        ax.set_xticks(sizes, [fmt_size(s) for s in sizes])
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("latency (us)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_timeline(d: dict, out: str, window_ms: float, max_points: int, plt) -> None:
    m = d["meta"]
    t = d["start_ns"] / 1e9
    lat = d["lat_ns"] / 1e3
    if lat.size == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    idx = np.arange(lat.size)
    if lat.size > max_points:
        idx = np.random.default_rng(0).choice(lat.size, max_points, replace=False)
    ax.scatter(t[idx], lat[idx], s=1, alpha=0.25, color="gray", rasterized=True, label="op")
    win = window_ms / 1e3
    bins = (t // win).astype(np.int64)
    edges = np.flatnonzero(np.diff(bins)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [lat.size]])
    centers = (bins[starts] + 0.5) * win
    p50 = [np.percentile(lat[s:e], 50) for s, e in zip(starts, ends)]
    p99 = [np.percentile(lat[s:e], 99) for s, e in zip(starts, ends)]
    ax.plot(centers, p50, color="tab:green", lw=1, label=f"p50 / {window_ms:g}ms")
    ax.plot(centers, p99, color="tab:red", lw=1, label=f"p99 / {window_ms:g}ms")
    ax.set(yscale="log", xlabel="time (s)", ylabel="latency (us)", title=m["name"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir")
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--window-ms", type=float, default=100.0)
    ap.add_argument("--max-points", type=int, default=200_000)
    ap.add_argument("--no-timeline", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.result_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no *.npz in {args.result_dir}")
    plot_dir = args.plot_dir or args.result_dir
    os.makedirs(plot_dir, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = [load(f) for f in files]
    rows = [summarize(d) for d in data]
    rows.sort(key=lambda r: (r["op"], r["pattern"], r["io_bytes"], r["qd"], r["path"]))
    write_csv(rows, os.path.join(args.result_dir, "summary.csv"))

    groups = defaultdict(list)
    for d in data:
        m = d["meta"]
        groups[(m["op"], m["pattern"], m["io_size"])].append(d)
    for (op, pat, io), group in groups.items():
        name = f"{op}_{pat}_bs{fmt_size(io)}"
        plot_cdf(group, f"{op} {pat} {fmt_size(io)}", os.path.join(plot_dir, f"cdf_{name}.png"), plt)

    by_op = defaultdict(list)
    for r in rows:
        by_op[(r["op"], r["pattern"])].append(r)
    for (op, pat), rs in by_op.items():
        plot_tail_vs_size(rs, f"{op} {pat}", os.path.join(plot_dir, f"tail_vs_size_{op}_{pat}.png"), plt)

    if not args.no_timeline:
        tdir = os.path.join(plot_dir, "timeline")
        os.makedirs(tdir, exist_ok=True)
        for d in data:
            plot_timeline(d, os.path.join(tdir, f"{d['meta']['name']}.png"),
                          args.window_ms, args.max_points, plt)

    cols = ["path", "op", "pattern", "io_size", "qd", "ops", "gib_s", "p50_us", "p99_us",
            "p99.9_us", "max_us", "p99_over_p50"]
    print("  ".join(f"{c:>10}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>10.2f}" if isinstance(r[c], float) else f"{r[c]!s:>10}" for c in cols))
    print(f"\nwrote {os.path.join(args.result_dir, 'summary.csv')} and plots in {plot_dir}")


if __name__ == "__main__":
    main()
