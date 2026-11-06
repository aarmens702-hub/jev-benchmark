"""The freeze manifest: SHA-256 of every frozen input, so any later change is detectable (VALIDATION.md, Manifest).

`build_manifest()` writes manifest.json with:
- `files`: SHA-256 of every tracked file under src/, scripts/, routing/, prompts/, third_party/, stage0/ and
  tests/, plus the lock file, pyproject.toml, .python-version, .gitignore, the design and validation documents,
  the plan, the deviations table and the registration document. manifest.json itself is never in this list.
- `train_parquet`: SHA-256 of each pinned train parquet if it is in the Hugging Face cache; otherwise its
  repo, revision and filename with the hash marked "pending". A pending hash must be resolved before the tag:
  `verify_manifest` and `load_test` both refuse it.
- `routing_rule_fingerprint`: `routing.rule_fingerprint()`.
- `seeds`, `thresholds` and `models`: values recorded as the code uses them, read from the modules. They are
  recorded, not hashed; the files that hold them are hashed under `files`.
- `repo_url`: the public repo the freeze is registered in. It is known in advance, so it is written at build.

Registration is the public tag `freeze-v1` on the commit holding this manifest, in the repo named by
`repo_url`. The tag records the commit; the manifest cannot, since a commit cannot contain its own SHA. The
freeze token `load_test` needs is SHA-256 of manifest.json's bytes at that tag (`firewall.freeze_token`).

`verify_manifest()` recomputes all of it, compares canonical bytes, and lists every difference, plus anything
that makes the tree unclean or hides a change from git. Every git call uses the firewall's hardened git.

CLI (from the repo root): `scripts/freeze.sh` builds, `scripts/freeze.sh --verify` checks. Nothing here tags,
commits, pushes or touches the network.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import yaml

import firewall
from firewall import FREEZE_TAG, MANIFEST_NAME, PENDING, REGISTRATION_REPO_URL, REPO_URL_KEY, canonical_bytes

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "jev-benchmark-manifest/2"
FROZEN_DIRS = ("src", "scripts", "routing", "prompts", "third_party", "stage0", "tests")
FROZEN_FILES = (
    "uv.lock",
    "pyproject.toml",
    ".python-version",
    ".gitignore",
    "DESIGN.md",
    "VALIDATION.md",
    "PLAN.md",
    "deviations.md",
    "registration.md",
    "stage0/frozen.yaml",
    "stage0/stage0_report.md",
)
HASH_NOTE = (
    "The freeze token is SHA-256 of this file's bytes at tag freeze-v1. seeds, thresholds and models are recorded "
    "values, not hashes; the files that hold them are hashed under files."
)


class FreezeError(Exception):
    """The manifest cannot be built or verified as asked."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _z(root: Path, *args: str) -> list[str]:
    return [p for p in firewall.git_bytes(root, *args, error=FreezeError).decode("utf-8", errors="replace").split("\0") if p]


def tag_commit(root: Path) -> str | None:
    """Full commit SHA of the freeze tag, or None if the tag does not exist."""
    return firewall.tag_commit(root, FREEZE_TAG, error=FreezeError)


def tree_problems(root: Path) -> list[str]:
    """Hardened clean-tree check (see firewall.worktree_problems); changes to manifest.json itself are allowed."""
    return firewall.worktree_problems(root, FROZEN_DIRS, allow_changed=(MANIFEST_NAME,), error=FreezeError)


# --------------------------------------------------------------------------- what is frozen


def frozen_paths(root: Path) -> list[str]:
    """Repo-relative paths of every frozen file: tracked files in FROZEN_DIRS plus FROZEN_FILES."""
    tracked = set(_z(Path(root), "ls-files", "-z", "--cached", "--", *FROZEN_DIRS, *FROZEN_FILES))
    missing = [f for f in FROZEN_FILES if f not in tracked]
    if missing:
        raise FreezeError(f"frozen files must exist and be tracked by git: {missing}")
    return sorted(p for p in tracked if p != MANIFEST_NAME)


def file_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    hashes = {}
    for rel in frozen_paths(root):
        path = root / rel
        if path.is_symlink() or not path.is_file():
            raise FreezeError(f"tracked frozen file is missing or a symlink in the working tree: {rel}")
        hashes[rel] = sha256_file(path)
    return hashes


def train_parquet_hashes(hf_cache_dir: Path | None = None) -> dict[str, dict]:
    """Hash each pinned train parquet if it is in the HF cache. Never downloads; a missing file is "pending"."""
    import data
    from huggingface_hub import try_to_load_from_cache

    out = {}
    for name, (repo_id, revision) in sorted(data.PINNED.items()):
        cached = try_to_load_from_cache(
            repo_id, data.TRAIN_FILE, cache_dir=hf_cache_dir, revision=revision, repo_type="dataset"
        )
        found = isinstance(cached, str) and Path(cached).is_file()
        out[name] = {
            "repo_id": repo_id,
            "revision": revision,
            "filename": data.TRAIN_FILE,
            "sha256": sha256_file(Path(cached)) if found else PENDING,
        }
    return out


def _small_llm(root: Path) -> dict:
    path = Path(root) / "stage0" / "frozen.yaml"
    try:
        frozen = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise FreezeError(f"stage0/frozen.yaml cannot be read: {exc.__class__.__name__}") from exc
    llm = (frozen or {}).get("small_llm") if isinstance(frozen, dict) else None
    if not isinstance(llm, dict):
        raise FreezeError("stage0/frozen.yaml has no small_llm section")
    keep = ("status", "model", "provider", "provider_options", "reasoning_field_sent", "reply_model_suffix_pattern",
            "price_per_million_tokens_usd")
    return {k: llm.get(k) for k in keep}


def live_settings(root: Path = REPO) -> dict:
    """Seeds, thresholds and model ids exactly as the code uses them, plus the small LLM frozen by Stage 0."""
    import arms_local
    import data
    import jev_client
    import ledger
    import llm_client
    import metrics
    import openrouter
    import predict_local
    import retrieval
    import routing
    import score
    import stage0
    import verdict

    seeds = {
        "split": data.SEED,
        "option_order": openrouter.ORDER_SEED,
        "bootstrap": score.RECORD.seed,
        "dry_run_gold_permutation": score.GOLD_SEED,
        "routing_sample_prefix": routing.PREFIX,
        "routing_bootstrap": routing.SEED,
        "lr": arms_local.LR_SEED,
        "lr_cv_folds": arms_local.CV_SEED,
    }
    thresholds = {
        "coverage": score.RECORD.coverage,
        "bootstrap_replicates": score.RECORD.b,
        "grid": list(metrics.GRID),
        "draws": metrics.N_DRAWS,
        "margin_points": verdict.MARGIN,
        "lr_sequence": list(verdict.LR_SEQUENCE),
        "jev_sequence": list(verdict.JEV_SEQUENCE),
        "claim_ci_level": verdict.CLAIM_LEVEL,
        "claim_match_limit": verdict.MATCH_LIMIT,
        "examples_ci_level": verdict.EXAMPLES_LEVEL,
        "examples_limit": verdict.EXAMPLES_LIMIT,
        "ag_news_ceiling_at": verdict.AG_CEILING_AT,
        "failure_flag_at": dict(score.FAIL_FLAG_AT),
        "secondary_coverages": list(score.SECONDARY_COVERAGES),
        "headline_coverages": list(score.HEADLINE_COVERAGES),
        "realised_coverage_keep": score.VALIDATION_KEEP,
        "nll_floor": metrics.NLL_FLOOR,
        "ece_bins": inspect.signature(metrics.ece_with_floor).parameters["bins"].default,
        "embed_temperature": arms_local.EMBED_TEMPERATURE,
        "embed_sensitivity_temperatures": list(arms_local.SENSITIVITY_TEMPERATURES),
        "lr_c": arms_local.LR_C,
        "lr_max_iter": arms_local.LR_MAX_ITER,
        "lr_cv_grid": list(arms_local.CV_GRID),
        "lr_cv_max_folds": arms_local.CV_MAX_FOLDS,
        "nli_batch_size": arms_local.NLI_BATCH_SIZE,
        "nli_max_len": arms_local.NLI_MAX_LEN,
        "nli_bare_template": arms_local.BARE_TEMPLATE,
        "nli_limit_hours": predict_local.NLI_LIMIT_HOURS,
        "bm25_k1": retrieval.K1,
        "bm25_b": retrieval.B,
        "retrieval_max_per_class": retrieval.MAX_PER_CLASS,
        "retrieval_max_total": retrieval.MAX_TOTAL,
        "small_llm_max_tokens": llm_client.MAX_TOKENS,
        "small_llm_select_pass_of_50": stage0.SELECT_PASS,
        "max_request_bytes": openrouter.MAX_BODY_BYTES,
        "hard_cap_usd": ledger.HARD_CAP_USD,
        "stage_caps_usd": dict(ledger.STAGE_CAPS_USD),
        "routing_n_sample": routing.N_SAMPLE,
        "routing_margin_points": routing.MARGIN,
        "routing_ci_level": routing.LEVEL,
        "routing_agreement_range": list(routing.AGREEMENT_RANGE),
        "routing_blind_limit": routing.BLIND_LIMIT,
        "routing_lr_a_floor": routing.LR_A_FLOOR,
        "routing_lr_a_s": routing.LR_A_S,
        "routing_intent_lr_s": routing.INTENT_LR_S,
        "routing_draw": routing.DRAW,
    }
    models = {
        "jev": {"model": jev_client.JEV_MODEL, "provider": jev_client.JEV_PROVIDER, "build": jev_client.EXPECTED_JEV_BUILD},
        "small_llm": _small_llm(root),
        "nli": {k: {"model": m, "revision": r} for k, (m, r) in arms_local.NLI_MODELS.items()},
        "embedder": {"model": arms_local.EMBED_MODEL, "revision": arms_local.EMBED_REVISION},
        "datasets": {k: {"repo_id": r, "revision": rev} for k, (r, rev) in data.PINNED.items()},
        "test_files": {k: dict(v) for k, v in firewall.PINNED_TEST_FILES.items()},
    }
    # Round-trip through JSON so recorded and recomputed values compare equal (tuples become lists).
    return json.loads(json.dumps({"seeds": seeds, "thresholds": thresholds, "models": models}))


def rule_fingerprint() -> str:
    import routing

    return routing.rule_fingerprint()


# --------------------------------------------------------------------------- build and verify


def _parts(root: Path, hf_cache_dir: Path | None):
    """The manifest's sections, each computed lazily so verify can report one failing section and go on."""
    yield "header", lambda: {"schema": SCHEMA, "hash_note": HASH_NOTE, "freeze_tag": FREEZE_TAG, REPO_URL_KEY: REGISTRATION_REPO_URL}
    yield "files", lambda: {"files": file_hashes(root)}
    yield "train_parquet", lambda: {"train_parquet": train_parquet_hashes(hf_cache_dir)}
    yield "routing_rule_fingerprint", lambda: {"routing_rule_fingerprint": rule_fingerprint()}
    yield "settings", lambda: live_settings(root)


def compute_manifest(root: Path = REPO, *, hf_cache_dir: Path | None = None) -> dict:
    """The manifest as it would be written now."""
    manifest = {}
    for _, compute in _parts(Path(root), hf_cache_dir):
        manifest.update(compute())
    return manifest


def pending_hashes(manifest: dict) -> list[str]:
    return sorted(k for k, v in manifest.get("train_parquet", {}).items() if v.get("sha256") == PENDING)


def build_manifest(root: Path = REPO, *, hf_cache_dir: Path | None = None) -> dict:
    """Write manifest.json for the current tree.

    Refuses once the freeze tag exists (the frozen manifest is final) and refuses a dirty tree, since the
    manifest must hash exactly what gets committed. Only manifest.json itself may differ from HEAD.
    """
    root = Path(root)
    if tag_commit(root) is not None:
        raise FreezeError(f"freeze tag {FREEZE_TAG} exists; the frozen manifest cannot be rebuilt")
    problems = tree_problems(root)
    if problems:
        raise FreezeError("tree is not clean; commit or remove changes first: " + "; ".join(problems))
    manifest = compute_manifest(root, hf_cache_dir=hf_cache_dir)
    (root / MANIFEST_NAME).write_bytes(canonical_bytes(manifest))
    return manifest


def _same(a, b) -> bool:
    """Equal as canonical JSON, so 1, 1.0 and true are all different."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _diff(recorded, current, path: str, out: list[str]) -> None:
    if isinstance(recorded, dict) and isinstance(current, dict):
        for key in sorted(set(recorded) | set(current)):
            where = f"{path}.{key}" if path else str(key)
            if key not in current:
                out.append(f"{where}: in the manifest but no longer present")
            elif key not in recorded:
                out.append(f"{where}: present now but not in the manifest")
            else:
                _diff(recorded[key], current[key], where, out)
    elif not _same(recorded, current):
        out.append(f"{path}: changed (manifest {recorded!r}, now {current!r})")


def verify_manifest(root: Path = REPO, *, hf_cache_dir: Path | None = None) -> list[str]:
    """Every difference between manifest.json and the tree now. An empty list means nothing frozen changed.

    Compares canonical bytes, so a type change (1 to true, 0 to 0.0) counts. Also reported: a manifest not in
    canonical form or a symlink, a pending train parquet hash or one not in the cache, and anything that makes
    the tree unclean or hides a change from git (changes to manifest.json itself excepted).
    """
    root = Path(root)
    path = root / MANIFEST_NAME
    if path.is_symlink() or not path.is_file():
        return [f"no {MANIFEST_NAME} file at {root}"]
    raw = path.read_bytes()
    try:
        recorded = firewall.read_manifest(raw)
    except firewall.FirewallError as exc:
        return [str(exc)]
    try:
        problems = tree_problems(root)
    except FreezeError as exc:
        problems = [str(exc)]
    current, failed_keys = {}, set()
    for name, compute in _parts(root, hf_cache_dir):
        try:
            current.update(compute())
        except Exception as exc:  # report and keep checking the other sections
            problems.append(f"{name}: cannot be recomputed: {exc}")
            failed_keys |= {"seeds", "thresholds", "models"} if name == "settings" else {name}
    before = len(problems)
    _diff({k: v for k, v in recorded.items() if k not in failed_keys}, current, "", problems)
    if not failed_keys and len(problems) == before and canonical_bytes(current) != raw:
        problems.append("manifest.json differs from the recomputed manifest")  # backstop; _diff should name it
    for name in pending_hashes(recorded):
        problems.append(f"train_parquet.{name}: hash is pending; cache the file and rebuild before the tag")
    for name in pending_hashes(current):
        if name not in pending_hashes(recorded):
            problems.append(f"train_parquet.{name}: not in the Hugging Face cache, so it cannot be verified")
    return problems


# --------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="list every difference from manifest.json")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            problems = verify_manifest(REPO)
            for p in problems:
                print(p)
            if problems:
                print(f"manifest does NOT verify: {len(problems)} problem(s)")
                return 1
            print(f"manifest verifies; SHA-256 of manifest.json {sha256_file(REPO / MANIFEST_NAME)}")
            return 0
        manifest = build_manifest(REPO)
        print(f"wrote {MANIFEST_NAME}: {len(manifest['files'])} files hashed")
        for name in pending_hashes(manifest):
            print(f"train parquet {name}: hash PENDING (not in the Hugging Face cache); the freeze cannot proceed")
        print(f"SHA-256 of manifest.json {sha256_file(REPO / MANIFEST_NAME)} (the freeze token once tagged)")
        return 0
    except (FreezeError, firewall.FirewallError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
