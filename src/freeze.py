"""The freeze manifest: SHA-256 of every frozen input, so any later change is detectable (VALIDATION.md, Manifest).

`build_manifest()` writes manifest.json with:
- `files`: SHA-256 of every tracked file under src/, scripts/, routing/, prompts/ and third_party/, plus the lock
  file, pyproject.toml, .python-version, the design and validation documents, the plan, the deviations table,
  the registration document and Stage 0's outputs in stage0/. manifest.json itself is never in this list.
- `train_parquet`: SHA-256 of each pinned train parquet if it is in the Hugging Face cache; otherwise its
  repo, revision and filename with the hash marked "pending", for the controller to fill before the tag.
- `routing_rule_fingerprint`: `routing.rule_fingerprint()`.
- `seeds`, `thresholds` and `models`: the values the code actually uses, read from the modules.
- `registration`: null until the freeze tag exists.

`verify_manifest()` recomputes all of it and lists every difference. `register()` (CLI `--registration`) fills
the registration with the public repo URL and the full commit SHA of the freeze tag.

The manifest's own hash (`firewall.manifest_hash`, the token `load_test` needs) is SHA-256 of the canonical JSON
with the registration field set to null. A commit cannot contain its own SHA, so the registration can only be
written after the tag; leaving it out of the hash means filling it changes neither the manifest's hash nor any
frozen hash recorded in the manifest. The frozen manifest at the tag has registration null, so its hash is just
the SHA-256 of that file's bytes.

CLI (from the repo root): `scripts/freeze.sh` builds, `scripts/freeze.sh --verify` checks, and
`scripts/freeze.sh --registration` fills the registration once the tag exists. Nothing here tags, commits,
pushes or touches the network.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
from pathlib import Path

import yaml

import firewall
from firewall import FREEZE_TAG, MANIFEST_NAME, REGISTRATION_KEY, REGISTRATION_REPO_URL, canonical_bytes, manifest_hash

REPO = Path(__file__).resolve().parents[1]
SCHEMA = "jev-benchmark-manifest/1"
FROZEN_DIRS = ("src", "scripts", "routing", "prompts", "third_party")
FROZEN_FILES = (
    "uv.lock",
    "pyproject.toml",
    ".python-version",
    "DESIGN.md",
    "VALIDATION.md",
    "PLAN.md",
    "deviations.md",
    "registration.md",
    "stage0/frozen.yaml",
    "stage0/stage0_report.md",
)
PENDING = "pending"
HASH_NOTE = (
    "The manifest's hash is SHA-256 of this file in canonical form (sorted keys, 2-space indent, final newline) "
    "with registration set to null. registration is filled after the freeze tag exists and is not part of that hash."
)


class FreezeError(Exception):
    """The manifest cannot be built, verified or registered as asked."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=check)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise FreezeError(f"git {' '.join(args)} failed in {root}") from exc


def _git_lines(root: Path, *args: str) -> list[str]:
    return [line for line in _git(root, *args).stdout.split("\0") if line]


def tag_commit(root: Path) -> str | None:
    """Full commit SHA of the freeze tag, or None if the tag does not exist."""
    done = _git(root, "rev-parse", "-q", "--verify", f"refs/tags/{FREEZE_TAG}^{{commit}}", check=False)
    return done.stdout.strip() if done.returncode == 0 else None


# --------------------------------------------------------------------------- what is frozen


def frozen_paths(root: Path) -> list[str]:
    """Repo-relative paths of every frozen file: tracked files in FROZEN_DIRS plus FROZEN_FILES."""
    root = Path(root)
    tracked = set(_git_lines(root, "ls-files", "-z", "--cached", "--", *FROZEN_DIRS, *FROZEN_FILES))
    missing = [f for f in FROZEN_FILES if f not in tracked]
    if missing:
        raise FreezeError(f"frozen files must exist and be tracked by git: {missing}")
    return sorted(p for p in tracked if p != MANIFEST_NAME)


def untracked_frozen(root: Path) -> list[str]:
    """Untracked, not ignored files inside the frozen folders: code that would run but is not hashed."""
    return sorted(_git_lines(Path(root), "ls-files", "-z", "--others", "--exclude-standard", "--", *FROZEN_DIRS, "stage0"))


def file_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    hashes = {}
    for rel in frozen_paths(root):
        path = root / rel
        if not path.is_file():
            raise FreezeError(f"tracked frozen file is missing from the working tree: {rel}")
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
    yield "header", lambda: {"schema": SCHEMA, "hash_note": HASH_NOTE, "freeze_tag": FREEZE_TAG}
    yield "files", lambda: {"files": file_hashes(root)}
    yield "train_parquet", lambda: {"train_parquet": train_parquet_hashes(hf_cache_dir)}
    yield "routing_rule_fingerprint", lambda: {"routing_rule_fingerprint": rule_fingerprint()}
    yield "settings", lambda: live_settings(root)


def compute_manifest(root: Path = REPO, *, hf_cache_dir: Path | None = None) -> dict:
    """The manifest as it would be written now, with registration null."""
    manifest = {}
    for _, compute in _parts(Path(root), hf_cache_dir):
        manifest.update(compute())
    manifest[REGISTRATION_KEY] = None
    return manifest


def pending_hashes(manifest: dict) -> list[str]:
    return sorted(k for k, v in manifest.get("train_parquet", {}).items() if v.get("sha256") == PENDING)


def build_manifest(root: Path = REPO, *, hf_cache_dir: Path | None = None) -> dict:
    """Write manifest.json for the current tree. Refuses once the freeze tag exists: the frozen manifest is final."""
    root = Path(root)
    if tag_commit(root) is not None:
        raise FreezeError(f"freeze tag {FREEZE_TAG} exists; the frozen manifest cannot be rebuilt")
    manifest = compute_manifest(root, hf_cache_dir=hf_cache_dir)
    (root / MANIFEST_NAME).write_bytes(canonical_bytes(manifest))
    return manifest


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
    elif recorded != current:
        out.append(f"{path}: changed (manifest {recorded!r}, now {current!r})")


def verify_manifest(root: Path = REPO, *, hf_cache_dir: Path | None = None) -> list[str]:
    """Every difference between manifest.json and the tree now. An empty list means nothing frozen changed.

    Also reported: a manifest not in canonical form, a train parquet hash still pending or not in the cache,
    untracked files in the frozen folders, and a registration that is filled but wrong.
    """
    root = Path(root)
    path = root / MANIFEST_NAME
    if not path.is_file():
        return [f"no {MANIFEST_NAME} at {root}"]
    try:
        recorded = firewall.read_manifest(path.read_bytes())
    except firewall.FirewallError as exc:
        return [str(exc)]
    problems = []
    current, failed_keys = {}, set()
    for name, compute in _parts(root, hf_cache_dir):
        try:
            current.update(compute())
        except Exception as exc:  # report and keep checking the other sections
            problems.append(f"{name}: cannot be recomputed: {exc}")
            failed_keys |= {"seeds", "thresholds", "models"} if name == "settings" else {name}
    frozen_part = {k: v for k, v in recorded.items() if k != REGISTRATION_KEY and k not in failed_keys}
    _diff(frozen_part, current, "", problems)
    for name in pending_hashes(recorded):
        problems.append(f"train_parquet.{name}: hash is pending; cache the file and rebuild before the tag")
    for name in pending_hashes(current):
        if name not in pending_hashes(recorded):
            problems.append(f"train_parquet.{name}: not in the Hugging Face cache, so it cannot be verified")
    for rel in untracked_frozen(root):
        problems.append(f"{rel}: untracked file in a frozen folder is not covered by the manifest")
    registration = recorded.get(REGISTRATION_KEY)
    if registration is not None:
        commit = tag_commit(root)
        try:
            if commit is None:
                raise firewall.FirewallError(f"registration is filled but freeze tag {FREEZE_TAG} does not exist")
            firewall.check_registration(recorded, commit)
        except firewall.FirewallError as exc:
            problems.append(str(exc))
    return problems


def register(root: Path = REPO, *, hf_cache_dir: Path | None = None) -> dict:
    """Fill the registration (repo URL and the freeze tag's full commit SHA) in manifest.json. Run once the tag exists.

    Needs HEAD at the tag, a clean tree and a manifest that verifies. The manifest's hash is unchanged by this
    (see the module docstring). Commit the result on top of the tag; the firewall accepts exactly that commit.
    """
    root = Path(root)
    commit = tag_commit(root)
    if commit is None:
        raise FreezeError(f"freeze tag {FREEZE_TAG} does not exist yet; tag first, then register")
    path = root / MANIFEST_NAME
    manifest = firewall.read_manifest(path.read_bytes())
    target = {"repo_url": REGISTRATION_REPO_URL, "commit": commit}
    if manifest.get(REGISTRATION_KEY) == target:
        return target
    if manifest.get(REGISTRATION_KEY) is not None:
        raise FreezeError(f"manifest already holds a different registration: {manifest[REGISTRATION_KEY]!r}")
    if _git(root, "rev-parse", "HEAD").stdout.strip() != commit:
        raise FreezeError(f"HEAD must be at freeze tag {FREEZE_TAG} to register")
    if _git(root, "status", "--porcelain").stdout.strip():
        raise FreezeError("git tree must be clean to register")
    problems = verify_manifest(root, hf_cache_dir=hf_cache_dir)
    if problems:
        raise FreezeError("manifest does not verify; not registering:\n" + "\n".join(problems))
    before = manifest_hash(manifest)
    manifest[REGISTRATION_KEY] = target
    if manifest_hash(manifest) != before:  # cannot happen by construction; refuse rather than write
        raise FreezeError("filling the registration would change the manifest's hash")
    path.write_bytes(canonical_bytes(manifest))
    return target


# --------------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true", help="list every difference from manifest.json")
    mode.add_argument("--registration", action="store_true", help="fill the registration once the freeze tag exists")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            problems = verify_manifest(REPO)
            for p in problems:
                print(p)
            if problems:
                print(f"manifest does NOT verify: {len(problems)} problem(s)")
                return 1
            print(f"manifest verifies; manifest hash {firewall.freeze_token(REPO / MANIFEST_NAME)}")
            return 0
        if args.registration:
            reg = register(REPO)
            print(f"registration: {reg['repo_url']} at {reg['commit']}")
            print(f"manifest hash unchanged: {firewall.freeze_token(REPO / MANIFEST_NAME)}")
            return 0
        manifest = build_manifest(REPO)
        print(f"wrote {MANIFEST_NAME}: {len(manifest['files'])} files hashed")
        for name in pending_hashes(manifest):
            print(f"train parquet {name}: hash PENDING (not in the Hugging Face cache)")
        print(f"manifest hash {manifest_hash(manifest)}")
        return 0
    except (FreezeError, firewall.FirewallError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
