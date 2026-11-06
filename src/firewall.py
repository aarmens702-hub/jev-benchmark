"""The only module that may load the official test split (VALIDATION.md, Guards).

`load_test` returns text and gold as separate frames, and only after four conditions hold, all
checked before any download:

1. The token equals the frozen manifest's hash: SHA-256 of manifest.json in canonical form with its
   `registration` field set to null. The frozen manifest at the freeze tag has registration null, so
   this is exactly the SHA-256 of that file's bytes. Filling the registration later does not change it.
2. The manifest holds a registration: the public repo URL, exactly REGISTRATION_REPO_URL, and the
   full 40-character commit SHA of the freeze tag, which must equal the local tag's commit.
3. The git tree is clean (untracked files count as dirty).
4. HEAD is at the freeze tag. A commit cannot contain its own SHA, so the registration is written
   after tagging and committed on top. HEAD may therefore be the tag's commit or a single registration
   commit whose only parent is the tag's commit and whose only change is the registration field of
   manifest.json. Any other commit after the tag is refused.

Files come by `hf_hub_download` of a named file at a pinned revision, never by a dataset loader.
"""
import hashlib
import hmac
import json
import re
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_TAG = "freeze-v1"
MANIFEST_NAME = "manifest.json"
REGISTRATION_KEY = "registration"
REGISTRATION_REPO_URL = "https://github.com/aarmens702-hub/jev-benchmark"
_FULL_SHA = re.compile(r"[0-9a-f]{40}")

# Revisions are pinned in DESIGN.md. The file names are the repos' standard parquet layout and
# have not been checked against the Hub; confirm them when the freeze is prepared.
PINNED_TEST_FILES = {
    "PolyAI/banking77": {
        "revision": "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6",
        "filename": "data/test-00000-of-00001.parquet",
    },
    "fancyzhx/ag_news": {
        "revision": "eb185aade064a813bc0b7f42de02595523103ca4",
        "filename": "data/test-00000-of-00001.parquet",
    },
}
_ALIASES = {"banking77": "PolyAI/banking77", "ag_news": "fancyzhx/ag_news"}


class FirewallError(Exception):
    """A condition for reading the test split does not hold."""


def canonical_bytes(manifest: dict) -> bytes:
    """The one byte form a manifest is written in: sorted keys, 2-space indent, UTF-8, final newline."""
    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def manifest_hash(manifest: dict) -> str:
    """SHA-256 of the manifest in canonical form with the registration field set to null.

    The registration (repo URL and tagged commit) can only be written after the tag exists, so it is left
    out of the hash: filling it changes neither this hash nor any frozen hash in the manifest. For the
    frozen manifest, whose registration is null, this equals the SHA-256 of the file's bytes.
    """
    if not isinstance(manifest, dict):
        raise FirewallError("manifest.json is not a JSON object")
    return hashlib.sha256(canonical_bytes({**manifest, REGISTRATION_KEY: None})).hexdigest()


def read_manifest(raw: bytes) -> dict:
    """Parse manifest bytes; refuse anything that is not a JSON object written in canonical form."""
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise FirewallError("manifest.json is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise FirewallError("manifest.json is not a JSON object")
    if canonical_bytes(manifest) != raw:
        raise FirewallError("manifest.json is not in canonical form (sorted keys, 2-space indent, one final newline)")
    return manifest


def freeze_token(manifest_path: Path) -> str:
    """The frozen manifest's hash, as hex: see `manifest_hash`."""
    return manifest_hash(read_manifest(Path(manifest_path).read_bytes()))


def _git(repo_dir: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError) as exc:
        raise FirewallError(f"git {' '.join(args)} failed in {repo_dir}") from exc
    return done.stdout.strip()


def _git_bytes(repo_dir: Path, *args: str) -> bytes:
    try:
        done = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError) as exc:
        raise FirewallError(f"git {' '.join(args)} failed in {repo_dir}") from exc
    return done.stdout


def _tag_commit(repo_dir: Path, freeze_tag: str) -> str:
    try:
        return _git(repo_dir, "rev-parse", "--verify", "-q", f"refs/tags/{freeze_tag}^{{commit}}")
    except FirewallError as exc:
        raise FirewallError(f"freeze tag {freeze_tag} does not exist") from exc


def check_registration(manifest: dict, tag_commit: str) -> None:
    """The manifest's registration must be exactly the public repo URL and the freeze tag's full commit SHA."""
    reg = manifest.get(REGISTRATION_KEY)
    if not isinstance(reg, dict) or set(reg) != {"repo_url", "commit"}:
        raise FirewallError("manifest has no registration (repo_url and commit of the freeze tag)")
    if reg["repo_url"] != REGISTRATION_REPO_URL:
        raise FirewallError(f"registration repo_url must be exactly {REGISTRATION_REPO_URL}")
    if not isinstance(reg["commit"], str) or not _FULL_SHA.fullmatch(reg["commit"]):
        raise FirewallError("registration commit must be a full 40-character commit SHA")
    if reg["commit"] != tag_commit:
        raise FirewallError("registration commit does not match the freeze tag's commit")


def _check_head(repo_dir: Path, manifest: dict, tag_commit: str, freeze_tag: str) -> None:
    """HEAD is the tag's commit, or one registration commit directly on it that changes nothing else."""
    head = _git(repo_dir, "rev-parse", "HEAD")
    if head == tag_commit:
        return
    not_at_tag = FirewallError(f"HEAD is not at freeze tag {freeze_tag} or its registration commit")
    parents = _git(repo_dir, "rev-list", "--parents", "-n", "1", "HEAD").split()[1:]
    if parents != [tag_commit]:
        raise not_at_tag
    if _git(repo_dir, "diff", "--no-renames", "--name-only", tag_commit, head).splitlines() != [MANIFEST_NAME]:
        raise not_at_tag
    try:
        frozen = read_manifest(_git_bytes(repo_dir, "show", f"{tag_commit}:{MANIFEST_NAME}"))
    except FirewallError as exc:
        raise FirewallError(f"the manifest at freeze tag {freeze_tag} is missing or not canonical") from exc
    if frozen.get(REGISTRATION_KEY, "missing") is not None:
        raise FirewallError(f"the manifest at freeze tag {freeze_tag} must have registration null")
    if {**manifest, REGISTRATION_KEY: None} != frozen:
        raise FirewallError("the registration commit changed more than the manifest's registration field")


def _check_conditions(token, repo_dir: Path, manifest_path: Path, freeze_tag: str) -> None:
    # 1. The token must equal the frozen manifest's hash.
    if not isinstance(token, str) or not token:
        raise FirewallError("a freeze token is required")
    if not manifest_path.is_file():
        raise FirewallError(f"no manifest at {manifest_path}")
    manifest = read_manifest(manifest_path.read_bytes())
    if not hmac.compare_digest(token.encode("utf-8"), manifest_hash(manifest).encode("utf-8")):
        raise FirewallError("token does not match the frozen manifest's hash")

    # 2. The manifest must hold a registration naming the public repo and the freeze tag's commit.
    tag_commit = _tag_commit(repo_dir, freeze_tag)
    check_registration(manifest, tag_commit)

    # 3. The git tree must be clean (untracked files count as dirty).
    if _git(repo_dir, "status", "--porcelain"):
        raise FirewallError("git tree is dirty; it must be clean")

    # 4. HEAD must be at the freeze tag (or its registration commit, see the module docstring).
    _check_head(repo_dir, manifest, tag_commit, freeze_tag)


def _hub_download(repo_id: str, filename: str, revision: str) -> str:
    """Fetch one named pinned test file. Refuses anything else; reached only through `_load_test`."""
    if PINNED_TEST_FILES.get(repo_id, {}).get("filename") != filename or PINNED_TEST_FILES[repo_id]["revision"] != revision:
        raise FirewallError(f"refusing to fetch {repo_id}/{filename}@{revision}: not a pinned test file")
    import huggingface_hub

    return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)


def _load_test(
    dataset: str,
    token: str,
    *,
    repo_dir: Path,
    manifest_path: Path | None = None,
    freeze_tag: str = FREEZE_TAG,
    downloader=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parameterised core of `load_test`. Private: tests call it with a temp repo and a fake downloader."""
    repo_id = _ALIASES.get(dataset, dataset)
    spec = PINNED_TEST_FILES.get(repo_id)
    if spec is None:
        raise FirewallError(f"unknown dataset {dataset!r}; pinned datasets are {sorted(PINNED_TEST_FILES)}")
    repo_dir = Path(repo_dir)
    manifest_path = Path(manifest_path) if manifest_path else repo_dir / MANIFEST_NAME
    _check_conditions(token, repo_dir, manifest_path, freeze_tag)

    path = (downloader or _hub_download)(repo_id, spec["filename"], spec["revision"])
    frame = pd.read_parquet(path, columns=["text", "label"])
    row_id = pd.RangeIndex(len(frame), name="row_id")
    texts = pd.DataFrame({"row_id": row_id, "text": frame["text"].to_numpy()})
    gold = pd.DataFrame({"row_id": row_id, "label": frame["label"].to_numpy()})
    return texts, gold


def load_test(dataset: str, token: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row_id, text) and (row_id, label) for the official test split, as separate frames.

    Always checks this repo, its manifest.json and the freeze tag; none of that can be redirected.
    row_id is the row's position in the official file.
    """
    return _load_test(dataset, token, repo_dir=REPO_ROOT, manifest_path=REPO_ROOT / MANIFEST_NAME, freeze_tag=FREEZE_TAG)
