#!/usr/bin/env python3
"""Telemetry summarizer: parse logs/telemetry.log and output CSV and basic stats.

Usage:
  python scripts/telemetry_summary.py [--in logs/telemetry.log] [--out out.csv]
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Dict, Any


FIELDS = [
    "ts",
    "client",
    "model_dx",
    "model_dy",
    "scaled_model_dx",
    "scaled_model_dy",
    "proposal_scale",
    "proposal_jitter",
    "applied_delta",
    "sensitivity",
    "avg_rms",
    "c_real",
    "patch_mean",
]


def iter_entries(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            # payload may be nested under 'payload'
            if isinstance(entry, dict) and "payload" in entry:
                payload = entry["payload"]
                payload.setdefault("ts", entry.get("ts"))
                payload.setdefault("client", entry.get("client"))
                yield payload
            else:
                yield entry


def summarize(entries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for e in entries:
        row = {k: e.get(k, None) for k in FIELDS}
        # numeric conversions
        for k in [
            "model_dx",
            "model_dy",
            "scaled_model_dx",
            "scaled_model_dy",
            "applied_delta",
            "sensitivity",
            "avg_rms",
            "proposal_scale",
            "proposal_jitter",
            "c_real",
            "patch_mean",
        ]:
            v = row.get(k)
            try:
                row[k] = float(v) if v is not None else None
            except Exception:
                row[k] = None
        rows.append(row)

    if not rows:
        return {"count": 0}

    def stats_for(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return {"count": 0}
        return {
            "count": len(vals),
            "mean": mean(vals),
            "median": median(vals),
            "p90": sorted(vals)[int(0.9 * len(vals))],
        }

    out: Dict[str, Any] = {"count": len(rows)}
    for k in [
        "model_dx",
        "model_dy",
        "scaled_model_dx",
        "scaled_model_dy",
        "applied_delta",
        "sensitivity",
        "avg_rms",
        "proposal_scale",
    ]:
        out[k] = stats_for(k)
    return out


def write_csv(entries: Iterable[Dict[str, Any]], out_path: Path):
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for e in entries:
            row = {k: e.get(k, None) for k in FIELDS}
            w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", default="logs/telemetry.log")
    parser.add_argument("--out", dest="outfile", default=None)
    args = parser.parse_args()

    p = Path(args.infile)
    if not p.exists():
        print(f"Input file not found: {p}")
        return

    entries = list(iter_entries(p))
    stats = summarize(entries)
    print(json.dumps(stats, indent=2))

    if args.outfile:
        write_csv(entries, Path(args.outfile))
        print(f"Wrote CSV to {args.outfile}")


if __name__ == "__main__":
    main()
