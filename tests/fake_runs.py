"""A fake set of prediction files for the scorer's tests and the dry run on the build machine.

Synthetic only: random scores over made-up rows. Nothing here reads a dataset, a model or an API.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import arms_local
import predictions
import score

CLASSES = {"banking77": ["card_arrival", "lost_card", "top_up_failed"], "ag_news": ["World", "Sports", "Business", "Sci/Tech"]}


def _softmax(z):
    return arms_local.softmax(z)


def make_fake_runs(
    root: Path,
    stage: str,
    *,
    per_class: dict | None = None,
    row_ids: dict | None = None,
    families=score.FAMILIES,
    skip=(),
    failures: dict | None = None,
    seed: int = 0,
    gold_as_ids: bool = False,
    gold: dict | None = None,
    classes_by_dataset: dict | None = None,
    blind: bool = False,
    int_ids: bool = False,
) -> dict:
    """Write every expected arm for each dataset; return {dataset: gold frame}.

    `skip` holds (arm or family, dataset) pairs to leave out. `failures` maps (arm, dataset) to a count of
    failed rows. Families not in `families` are logged not run. Gold frames use label ids when
    `gold_as_ids`, as the firewall gives them.
    """
    per_class = per_class or {"banking77": 6, "ag_news": 4}
    if gold is not None:  # given rows and labels (the build machine's dry run); scores can ignore them
        per_class = {d: None for d in gold}
    failures = failures or {}
    rng = np.random.default_rng(seed)
    golds = {}
    sizes = {}
    for dataset, n_per in per_class.items():
        classes = (classes_by_dataset or CLASSES)[dataset]
        k = len(classes)
        if gold is not None:
            labels = gold[dataset]["label"].to_numpy()
            ids = gold[dataset]["row_id"].tolist()
            n = len(ids)
        else:
            n = n_per * k
            labels = np.array([classes[i % k] for i in range(n)])
            ids = row_ids[dataset] if row_ids and dataset in row_ids else [f"{dataset}:train:v{i}" for i in range(n)]
            if int_ids:  # the firewall's row ids: positions in the official file, as integers
                ids = list(range(n))
        gold_idx = np.array([classes.index(c) for c in labels])
        golds[dataset] = pd.DataFrame({"row_id": ids, "label": gold_idx if gold_as_ids else labels})

        def write(arm, vec, *, to_probs=lambda v: v, api=False, keep_vectors=True):
            if (arm, dataset) in skip:
                return
            n_failed = failures.get((arm, dataset), 0)
            failed = np.zeros(n, dtype=bool)
            failed[:n_failed] = True
            vec = np.where(failed[:, None], np.nan, vec)
            probs = to_probs(np.where(failed[:, None], 0.0, vec))
            frame = pd.DataFrame(
                {
                    "row_id": ids,
                    "pred": np.where(failed, "", np.array(classes, dtype=object)[probs.argmax(axis=1)]),
                    "top_prob": np.where(failed, 0.0, probs.max(axis=1)),
                    "failed": failed,
                    "latency_s": rng.uniform(0.01, 0.5, n),
                }
            )
            if api:
                frame["model"] = "typesafe/jev-1.14-20261014" if arm.startswith("jev") else "openai/gpt-oss-20b"
                frame["input_tokens"] = pd.array(rng.integers(500, 2500, n), dtype="Int64")
                frame["output_tokens"] = pd.array(rng.integers(1, 5, n), dtype="Int64")
                frame["cost"] = rng.uniform(1e-5, 1e-4, n)
                frame["cached"] = False
            predictions.write(arm, dataset, frame, stage, vec if keep_vectors else None, classes if keep_vectors else None, root)

        def scores(strength):
            z = rng.normal(size=(n, k))
            if not blind:
                z[np.arange(n), gold_idx] += strength
            return z

        write("jev", np.round(_softmax(scores(2.0) * 1.5), 2), api=True)
        write("jev_bare", np.round(_softmax(scores(1.0) * 1.5), 2), api=True)
        if dataset == "banking77":
            write("jev_examples", np.round(_softmax(scores(2.5) * 1.5), 2), api=True)
        write("small_llm", _softmax(scores(1.5)), api=True, keep_vectors=False)
        write("nli", _softmax(scores(1.2)))
        cos = np.tanh(scores(1.0) * 0.2)
        write("embed_sim", cos, to_probs=lambda v: arms_local.softmax(v / 0.05))
        write("embed_sim_t0.01", arms_local.softmax(cos / 0.01))
        write("embed_sim_t0.2", arms_local.softmax(cos / 0.2))
        write("embed_sim_raw", cos)
        bare = np.tanh(scores(0.5) * 0.2)
        write("embed_sim_bare", bare, to_probs=lambda v: arms_local.softmax(v / 0.05))
        for family in score.FAMILIES:
            if (family, dataset) in skip:
                continue
            if family not in families:
                predictions.mark_not_run(family, dataset, stage, "not part of this fake", root)
                continue
            for draw, s, name in score.family_files(family):
                strength = {1: 0.5, 2: 0.8, 3: 1.0, 5: 1.3, 8: 1.6, 16: 2.0, 35: 2.5, "all": 3.0}[s]
                write(name, _softmax(scores(strength)))
                total = (k * s if s != "all" else k * 40) - (1 if dataset == "banking77" and s in (35, "all") else 0)
                sizes.setdefault(dataset, {})[f"d{draw}_s{s}"] = {"rows": total, "classes": k}
    path = Path(root) / stage / "bank_sizes.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sizes))
    return golds
