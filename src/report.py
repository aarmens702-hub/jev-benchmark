"""Writes the scorer's results: results.json, the CSV tables and the gap-vs-labels chart.

It gets computed results only. It never sees gold labels, predictions or probability vectors.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

GRID_LABELS = ("1", "2", "3", "5", "8", "16", "35", "all")
TOTAL_AXIS_TITLE = "total labels (all classes)"
DATASET_TITLES = {"banking77": "Banking77 (77 intents, sets the verdict)", "ag_news": "AG News (4 topics, a check only)"}


def _plain(value):
    """numpy and float edge cases to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def write_json(results: dict, path: Path) -> None:
    Path(path).write_text(json.dumps(_plain(results), indent=1, sort_keys=False), encoding="utf-8")


def _csv(rows: list, path: Path) -> None:
    pd.DataFrame([_plain(r) for r in rows]).to_csv(path, index=False)


def write_tables(results: dict, out_dir: Path) -> list:
    """One CSV per table. Every per-arm row carries the arm's failure flag; every D_s row lists flagged arms."""
    out_dir = Path(out_dir)
    written = []
    arms = [row for d in results["datasets"].values() for row in d.get("arms", [])]
    if arms:
        _csv(arms, out_dir / "arms.csv")
        written.append("arms.csv")
    for name, d in results["datasets"].items():
        if d.get("grid"):
            _csv([{"dataset": name, **r} for r in d["grid"]], out_dir / f"ds_{name}.csv")
            written.append(f"ds_{name}.csv")
        if d.get("coverages"):
            flags = ";".join(f for f in d.get("failure_flags", []))
            rows = [
                {"dataset": name, "coverage": c["coverage"], "s_star": c["s_star"], "s_j": c["s_j"], "verdict_row": c["verdict_row"],
                 "differs_from_80": c["differs_from_80"], "flags": flags}
                for c in d["coverages"]
            ]  # fmt: skip
            _csv(rows, out_dir / f"coverage_{name}.csv")
            written.append(f"coverage_{name}.csv")
        for family, sec in d.get("secondary_families", {}).items():
            if sec.get("grid"):
                _csv([{"dataset": name, "family": family, **r} for r in sec["grid"]], out_dir / f"ds_{family}_{name}.csv")
                written.append(f"ds_{family}_{name}.csv")
        rc = d.get("risk_coverage")
        if rc and rc["arms"]:
            flagged = set(d.get("failure_flags", []))
            frame = pd.DataFrame({"coverage": rc["coverage"], **{(f"{a} (flagged)" if a in flagged else a): v for a, v in rc["arms"].items()}})
            frame.to_csv(out_dir / f"risk_coverage_{name}.csv", index=False)
            written.append(f"risk_coverage_{name}.csv")
    adds = results.get("additions", {})
    if adds.get("his_claim"):
        flags = results.get("additions_flags", {})
        rows = [{"arm": arm, **vals, "flagged": flags.get(arm, False), "jev_flagged": flags.get("jev", False)} for arm, vals in adds["his_claim"].items()]
        _csv(rows, out_dir / "his_claim.csv")
        written.append("his_claim.csv")
    return written


def tick_labels(bank_sizes: dict) -> tuple[list, list]:
    """x labels (per class, with "all" given its real counts) and the total-labels axis labels."""
    per_class = []
    totals = []
    for s in GRID_LABELS:
        size = bank_sizes[s]
        totals.append(f"{size['total']:,.0f}")
        if s == "all":
            per_class.append(f"all\n{size['per_class']:.1f}/class avg\n{size['total']:,.0f} total")
        else:
            per_class.append(s)
    return per_class, totals


def chart(results: dict, path: Path) -> dict:
    """D_s with its 95% interval at each s, one panel per dataset that ran, with a total-labels axis on top.

    Returns the labels actually drawn, read back from the axes.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [(name, d) for name, d in results["datasets"].items() if d.get("grid") and d.get("bank_sizes")]
    fig, axes = plt.subplots(1, max(1, len(panels)), figsize=(6.5 * max(1, len(panels)), 4.8), squeeze=False)
    drawn = {}
    for ax, (name, d) in zip(axes[0], panels):
        x = np.arange(len(GRID_LABELS))
        grid = {row["s"]: row for row in d["grid"]}
        mid = np.array([grid[s]["d_s"] for s in GRID_LABELS])
        lo = np.array([grid[s]["lower"] for s in GRID_LABELS])
        hi = np.array([grid[s]["upper"] for s in GRID_LABELS])
        ax.errorbar(x, mid, yerr=[mid - lo, hi - mid], fmt="o-", color="#1f4e79", ecolor="#7f9fbf", capsize=4, label="D_s, 95% interval")
        ax.axhline(0, color="black", linewidth=0.8)
        for m in (-2, 2):
            ax.axhline(m, color="grey", linewidth=0.8, linestyle="--")
        per_class, totals = tick_labels(d["bank_sizes"])
        ax.set_xticks(x, per_class, fontsize=8)
        ax.set_xlabel("labels per class (s)")
        ax.set_ylabel("embedding LR minus zero-label Jev, acc@80 (points)")
        title = DATASET_TITLES.get(name, name)
        if d.get("failure_flags"):
            title += "\nflagged for failures: " + ", ".join(d["failure_flags"][:3])
        ax.set_title(title, fontsize=10)
        top = ax.secondary_xaxis("top")
        top.set_xticks(x, totals, fontsize=7)
        top.set_xlabel(TOTAL_AXIS_TITLE)
        ax.legend(fontsize=8, loc="best")
        fig.canvas.draw()
        drawn[name] = {
            "x_labels": [t.get_text() for t in ax.get_xticklabels()],
            "total_labels": [t.get_text() for t in top.get_xticklabels()],
            "total_axis_title": top.get_xlabel(),
            "title": ax.get_title(),
        }
    if results.get("dry_run"):
        fig.suptitle("DRY RUN: gold permuted, no real comparison", color="red")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return drawn


def write_all(results: dict, out_dir: Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = write_tables(results, out_dir)
    labels = chart(results, out_dir / "chart.png")
    results["chart"] = labels
    results["tables"] = tables
    write_json(results, out_dir / "results.json")
    return labels
