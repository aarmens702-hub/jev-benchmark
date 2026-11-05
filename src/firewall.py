"""The only module that may load the official test split (VALIDATION.md, Guards).

`load_test` returns text and gold as separate frames, and only after four conditions hold:
the token equals SHA-256 of the frozen manifest.json, the manifest holds an OSF id, the git
tree is clean, and HEAD is at the freeze tag. All four are checked before any download.
Files come by `hf_hub_download` of a named file at a pinned revision, never by a dataset loader.
"""
import hashlib
import hmac
import json
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_TAG = "freeze-v1"

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


def freeze_token(manifest_path: Path) -> str:
    """SHA-256 of the frozen manifest.json, as hex."""
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def _git(repo_dir: Path, *args: str) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError) as exc:
        raise FirewallError(f"git {' '.join(args)} failed in {repo_dir}") from exc
    return done.stdout.strip()


def _check_conditions(token, repo_dir: Path, manifest_path: Path, freeze_tag: str) -> None:
    # 1. The token must equal the hash of the frozen manifest.
    if not isinstance(token, str) or not token:
        raise FirewallError("a freeze token is required")
    if not manifest_path.is_file():
        raise FirewallError(f"no manifest at {manifest_path}")
    if not hmac.compare_digest(token, freeze_token(manifest_path)):
        raise FirewallError("token does not match the SHA-256 of manifest.json")

    # 2. The manifest must hold an OSF id.
    try:
        osf_id = json.loads(manifest_path.read_text()).get("osf_id")
    except (ValueError, AttributeError) as exc:
        raise FirewallError("manifest.json is not a JSON object") from exc
    if not isinstance(osf_id, str) or not osf_id.strip():
        raise FirewallError("manifest has no OSF id")

    # 3. The git tree must be clean (untracked files count as dirty).
    if _git(repo_dir, "status", "--porcelain"):
        raise FirewallError("git tree is dirty; it must be clean")

    # 4. HEAD must be at the freeze tag.
    try:
        tag_commit = _git(repo_dir, "rev-parse", "--verify", "-q", f"refs/tags/{freeze_tag}^{{commit}}")
    except FirewallError as exc:
        raise FirewallError(f"freeze tag {freeze_tag} does not exist") from exc
    if _git(repo_dir, "rev-parse", "HEAD") != tag_commit:
        raise FirewallError(f"HEAD is not at freeze tag {freeze_tag}")


def _hub_download(repo_id: str, filename: str, revision: str) -> str:
    import huggingface_hub

    return huggingface_hub.hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", revision=revision)


def load_test(
    dataset: str,
    token: str,
    *,
    repo_dir: Path = REPO_ROOT,
    manifest_path: Path | None = None,
    freeze_tag: str = FREEZE_TAG,
    downloader=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row_id, text) and (row_id, label) for the official test split, as separate frames.

    row_id is the row's position in the official file. `downloader(repo_id, filename, revision)`
    returns a local parquet path; it defaults to hf_hub_download and exists so tests can inject a fake.
    """
    repo_id = _ALIASES.get(dataset, dataset)
    spec = PINNED_TEST_FILES.get(repo_id)
    if spec is None:
        raise FirewallError(f"unknown dataset {dataset!r}; pinned datasets are {sorted(PINNED_TEST_FILES)}")
    repo_dir = Path(repo_dir)
    manifest_path = Path(manifest_path) if manifest_path else repo_dir / "manifest.json"
    _check_conditions(token, repo_dir, manifest_path, freeze_tag)

    path = (downloader or _hub_download)(repo_id, spec["filename"], spec["revision"])
    frame = pd.read_parquet(path, columns=["text", "label"])
    row_id = pd.RangeIndex(len(frame), name="row_id")
    texts = pd.DataFrame({"row_id": row_id, "text": frame["text"].to_numpy()})
    gold = pd.DataFrame({"row_id": row_id, "label": frame["label"].to_numpy()})
    return texts, gold
