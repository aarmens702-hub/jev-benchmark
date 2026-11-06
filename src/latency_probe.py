"""Latency probes (DESIGN.md, "cost and latency (no headline speed ratio)").

The method follows ickma2311/jev-baselines-eval at 3b828cb68992c00835b3f126a3d13534da875277 (its latency
scripts); nothing is copied. Requests go one at a time. Model load, the first (cold) request and the
warm-up requests are kept apart from the timed set. A minimal request to OpenRouter measures the network
alone. Swap in use is noted before and after each local probe. The report gives p50 and p95 per arm under
each condition (hardware, region, date, concurrency) and never a speed ratio between arms.

Probes run on validation rows. Local probes are free. API probes spend money, so they need
`--paid <stage>` and go through openrouter.run_arm like any paid call; they run under that stage's
approval, not during the build.

    PYTHONPATH=src uv run python -m latency_probe --local embed_sim --region ca-central
    PYTHONPATH=src uv run python -m latency_probe --api jev --paid stage1 --region ca-central
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

import guards
import openrouter
import predictions

DEFAULT_WARMUP = 5
DEFAULT_ROWS = 50
NETWORK_PROBES = 20
LOCAL_ARMS = ("embed_sim", "lr", "tfidf_lr", "nli")
API_ARMS = ("jev", "small_llm")


def percentiles(seconds: Sequence[float]) -> dict:
    """n, p50 and p95 of a list of wall-clock seconds (numpy's linear interpolation)."""
    values = np.asarray([s for s in seconds if s is not None and np.isfinite(s)], dtype=float)
    if len(values) == 0:
        return {"n": 0, "p50": None, "p95": None}
    p50, p95 = np.percentile(values, [50, 95])
    return {"n": int(len(values)), "p50": float(p50), "p95": float(p95)}


def hardware() -> str:
    """The machine, e.g. "Mac15,3 arm64". Falls back to the platform string."""
    try:
        model = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        model = ""
    return " ".join(x for x in (model, platform.machine()) if x) or platform.platform()


def condition(*, region: str, concurrency: int, hardware: Optional[str] = None, date: Optional[str] = None) -> dict:
    """The conditions a latency figure holds under. Figures from different conditions aren't compared."""
    return {
        "hardware": hardware if hardware is not None else globals()["hardware"](),
        "region": region,
        "date": date or _date.today().isoformat(),
        "concurrency": int(concurrency),
    }


def _split_timed(times: list, warmup: int) -> tuple[Optional[float], list]:
    """First request is cold; the next `warmup` are dropped; the rest are timed."""
    if not times:
        return None, []
    return times[0], times[1 + warmup :]


def probe_local(
    arm: str,
    load: Callable[[], Callable[[str], object]],
    texts: Sequence[str],
    *,
    warmup: int = DEFAULT_WARMUP,
    clock: Callable[[], float] = time.perf_counter,
    read_swap_gb: Callable[[], float] = guards.swap_used_gb,
) -> dict:
    """Time a local arm. `load` returns predict_one(text); each text is predicted alone, in order."""
    swap_before = read_swap_gb()
    started = clock()
    predict_one = load()
    load_s = clock() - started
    times = []
    for text in texts:
        t0 = clock()
        predict_one(text)
        times.append(clock() - t0)
    cold, timed = _split_timed(times, warmup)
    return {
        "arm": arm,
        "load_s": load_s,
        "cold_s": cold,
        "warmup": warmup,
        "timed": percentiles(timed),
        "swap_gb_before": swap_before,
        "swap_gb_after": read_swap_gb(),
    }


def probe_api(
    arm: str,
    rows: Sequence[dict],
    decide: Callable[[dict], object],
    *,
    client: openrouter.Client,
    results_path,
    warmup: int = DEFAULT_WARMUP,
    expected_per_row: Optional[float] = None,
    read_swap_gb: Callable[[], float] = guards.swap_used_gb,
) -> dict:
    """Time an API arm, one request at a time, through run_arm (ladder, budget and every spend guard).

    Each decision's own latency_s is used, in row order. The client must allow one request in flight.
    """
    if client.max_concurrency != 1:
        raise ValueError("a latency probe must be serial: build its client with max_concurrency=1")
    swap_before = read_swap_gb()
    report = openrouter.run_arm(rows, decide, client=client, results_path=results_path, expected_per_row=expected_per_row)
    ordered = [report.decisions.get(row["row_id"]) for row in rows]
    times = [d.latency_s for d in ordered if d is not None and not d.cached and not d.failed]
    cold, timed = _split_timed(times, warmup)
    return {
        "arm": arm,
        "concurrency": 1,
        "cold_s": cold,
        "warmup": warmup,
        "timed": percentiles(timed),
        "failed": sum(1 for d in ordered if d is not None and d.failed),
        "status": report.status,
        "swap_gb_before": swap_before,
        "swap_gb_after": read_swap_gb(),
    }


def network_probe(
    transport, *, n: int = NETWORK_PROBES, warmup: int = 1, clock: Callable[[], float] = time.perf_counter
) -> dict:
    """The smallest authenticated request (GET /api/v1/key, no tokens), to separate network time from model time."""
    times = []
    for _ in range(warmup + n):
        t0 = clock()
        transport.send("GET", openrouter.KEY_URL, None)
        times.append(clock() - t0)
    return {"what": "GET /api/v1/key", "warmup": warmup, "timed": percentiles(times[warmup:])}


def build_report(results: Sequence[dict]) -> dict:
    """Group probe results by condition. p50 and p95 per arm only; no ratio between arms."""
    groups: list[dict] = []
    for result in results:
        cond = result["condition"]
        group = next((g for g in groups if g["condition"] == cond), None)
        if group is None:
            group = {"condition": cond, "arms": []}
            groups.append(group)
        group["arms"].append({k: v for k, v in result.items() if k != "condition"})
    return {
        "note": "p50 and p95 wall-clock seconds per decision, by condition; not comparable across conditions",
        "by_condition": groups,
    }


# --------------------------------------------------------------------------- command line


@dataclass
class ProbeRuntime:
    """What the CLI touches, so tests can swap in fakes."""

    runs_root: Path = predictions.RUNS
    validation_texts: Optional[Callable[[str], list]] = None
    loaders: Optional[dict] = None  # arm -> load() for local probes
    deciders: Optional[dict] = None  # arm -> (client, rows) -> decide(row), for API probes


def _validation_texts(dataset: str) -> list:
    import data

    return data.split(data.load_train(dataset), dataset).val["text"].tolist()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="latency_probe", description="Serial latency probes on validation rows.")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--local", choices=LOCAL_ARMS)
    what.add_argument("--api", choices=API_ARMS)
    what.add_argument("--network", action="store_true", help="minimal-request probe; needs --paid as it uses the key")
    p.add_argument("--dataset", choices=["banking77", "ag_news"], default="banking77")
    p.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--region", default="unknown", help="where the requests leave from, e.g. ca-central")
    openrouter.add_paid_argument(p)
    return p


def main(argv=None, runtime: Optional[ProbeRuntime] = None) -> int:
    args = build_parser().parse_args(argv)
    rt = runtime or ProbeRuntime()
    if (args.api or args.network) and args.paid is None:
        raise openrouter.PaidFlagRequired("API and network probes spend money or use the key: pass --paid <stage>")
    texts = (rt.validation_texts or _validation_texts)(args.dataset)[: args.rows]
    out_dir = Path(rt.runs_root) / "latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.local:
        if not rt.loaders or args.local not in rt.loaders:
            raise SystemExit(f"no loader wired for {args.local}; build one with the frozen model and settings")
        result = probe_local(args.local, rt.loaders[args.local], texts, warmup=args.warmup)
        result["condition"] = condition(region=args.region, concurrency=1)
    elif args.network:
        transport = openrouter.RealTransport(args.paid)
        result = network_probe(transport)
        result["condition"] = condition(region=args.region, concurrency=1)
    else:
        if not rt.deciders or args.api not in rt.deciders:
            raise SystemExit(f"no decider wired for {args.api}; the stage runner passes the frozen prompt and options")
        client = openrouter.build_client(args.paid, max_concurrency=1)
        rows = [{"row_id": f"latency:{i}", "text": t} for i, t in enumerate(texts)]
        result = probe_api(
            args.api,
            rows,
            rt.deciders[args.api](client, rows),
            client=client,
            results_path=Path(rt.runs_root) / "private" / "latency" / f"{args.api}.jsonl",
            warmup=args.warmup,
        )
        result["condition"] = condition(region=args.region, concurrency=1)
    path = out_dir / "probes.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"latency probe written to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
