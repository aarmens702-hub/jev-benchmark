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


def make_fake_routing(root: Path, stage: str, gold_b77: pd.DataFrame, n: int, *, skip=(), not_run=(), fingerprint=True,
                      validity_path: Path | None = None, checks=None, seed: int = 0, strong=("Jev",)) -> pd.DataFrame:
    """Write the 9 routing files under runs/<stage>_routing/ for the first n rows by routing hash.

    Text is made up from the row id. Returns the text frame the scorer samples from. Arms in `strong` are
    right on most rows; the rest guess. `skip` leaves files out; `not_run` logs them not run.
    """
    import routing

    rng = np.random.default_rng(seed)
    text = pd.DataFrame({"row_id": gold_b77["row_id"].to_numpy(), "text": [f"message number {r}" for r in gold_b77["row_id"]]})
    sample = routing.sample_rows(text, n)
    by_id = dict(zip(gold_b77["row_id"].astype(str), gold_b77["label"]))
    keyed = sample.assign(label=[by_id[str(r)] for r in sample["row_id"]])
    golds = {rule: routing.gold_queues(keyed, rule) for rule in routing.RULES}
    rstage = score.routing_stage(stage)
    for name, (arm, rule) in score.ROUTING_FILES.items():
        if name in skip:
            continue
        if name in not_run:
            predictions.mark_not_run(name, "banking77", rstage, "not part of this fake", root)
            continue
        right = golds[rule or "A"]
        guess = rng.choice(list(routing.QUEUES), size=len(sample))
        pred = np.where(rng.random(len(sample)) < (0.9 if arm in strong else 0.3), right, guess)
        frame = pd.DataFrame({"row_id": sample["row_id"].to_numpy(), "pred": pred.astype(str), "top_prob": rng.uniform(0.3, 1.0, len(sample)),
                              "failed": np.zeros(len(sample), dtype=bool), "latency_s": rng.uniform(0.01, 0.2, len(sample))})
        predictions.write(name, "banking77", frame, rstage, root=root)
    folder = Path(root) / rstage
    folder.mkdir(parents=True, exist_ok=True)
    if fingerprint:
        (folder / score.FINGERPRINT_FILE).write_text(json.dumps({"fingerprint": routing.rule_fingerprint(), "first_arm": "jev_rule_A"}))
    if validity_path is not None:
        checks = checks or {k: True for k in score.VALIDITY_KEYS}
        Path(validity_path).write_text(json.dumps({"checks": checks, "fingerprint": routing.rule_fingerprint()}))
    return text
