"""The only module that may load the official test split (VALIDATION.md, Guards).

`load_test(dataset, token)` returns text and gold as separate frames, and only after every condition
below holds. All are checked before any download.

1. The token equals `freeze_token()`: SHA-256 of the bytes of manifest.json at the freeze tag
   (`git show freeze-v1:manifest.json`).
2. HEAD is the commit the freeze tag points to (peeled).
3. The tagged manifest is canonical JSON, records the public repo URL REGISTRATION_REPO_URL, has no
   pending hash, equals manifest.json on disk, and `freeze.verify_manifest` finds no difference between
   it and the tree, the settings in the code and the pinned train parquets.
4. The tree is clean, untracked files included, and nothing hides a change from git: no assume-unchanged
   or skip-worktree entries, no replace refs, no untracked file hidden by info/exclude or a global
   excludes file, no ignored file inside a frozen folder (bytecode of tracked sources excepted), and the
   repo's top level is this directory.
5. The public repo shows the tag: a credential-free `git ls-remote` of REGISTRATION_REPO_URL reports the
   peeled freeze tag and main both at HEAD. That works only once the repo is public, so registration is the
   public tag, and the manifest at the tag names the repo it is published in.

Every git call is hardened: all GIT_* variables removed, global and system config ignored, replace
objects off, fsmonitor off and untracked files always listed.

Files come by `hf_hub_download` of a named file at a pinned revision, never by a dataset loader.
After the first load the scorer may keep gold in gitignored runs/ (`cache_gold`, `read_cached_gold`),
so a rescore after results are committed never needs `load_test` again.
"""
import hashlib
import hmac
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_TAG = "freeze-v1"
MANIFEST_NAME = "manifest.json"
REPO_URL_KEY = "repo_url"
REGISTRATION_REPO_URL = "https://github.com/aarmens702-hub/jev-benchmark"
PENDING = "pending"
LS_REMOTE_TIMEOUT_SECONDS = 60

# Revisions are pinned in DESIGN.md. The file names are the repos' standard parquet layout; the HF API
# listed data/test-00000-of-00001.parquet at both pinned revisions on 2026-11-04.
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


# --------------------------------------------------------------------------- hardened git

_GIT_CONFIG = (
    "-c", "core.fsmonitor=false",
    "-c", "core.untrackedCache=false",
    "-c", "status.showUntrackedFiles=all",
    "-c", "core.ignoreStat=false",
)


def git_env() -> dict:
    """The environment for every git call: no GIT_* variable, no global or system config, no replace objects."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_TERMINAL_PROMPT="0")
    return env


def git_bytes(repo_dir: Path, *args: str, error=FirewallError) -> bytes:
    try:
        done = subprocess.run(["git", *_GIT_CONFIG, *args], cwd=repo_dir, capture_output=True, check=True, env=git_env())
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError) as exc:
        raise error(f"git {' '.join(args)} failed in {repo_dir}") from exc
    return done.stdout


def git_text(repo_dir: Path, *args: str, error=FirewallError) -> str:
    return git_bytes(repo_dir, *args, error=error).decode("utf-8", errors="replace").strip()


def _z(raw: bytes) -> list[str]:
    return [p for p in raw.decode("utf-8", errors="replace").split("\0") if p]


def tag_commit(repo_dir: Path, freeze_tag: str = FREEZE_TAG, error=FirewallError) -> str | None:
    """Full SHA of the commit the freeze tag points to, or None if there is no such tag."""
    try:
        return git_text(repo_dir, "rev-parse", "--verify", "-q", f"refs/tags/{freeze_tag}^{{commit}}", error=error)
    except error:
        return None


def _allowed_ignored(rel: str, tracked: set) -> bool:
    """Ignored files that may sit in a frozen folder: bytecode of a tracked source, and Finder's .DS_Store."""
    parts = rel.split("/")
    if parts[-1] == ".DS_Store":
        return True
    if len(parts) >= 2 and parts[-2] == "__pycache__" and parts[-1].endswith(".pyc"):
        stem = parts[-1].split(".")[0]
        return "/".join([*parts[:-2], stem + ".py"]) in tracked
    return False


def worktree_problems(repo_dir: Path, frozen_dirs, *, allow_changed=(), error=FirewallError) -> list[str]:
    """Everything that makes the tree not clean, or could hide a change from `git status`. Empty means clean.

    `allow_changed`: paths whose own changes are tolerated (freeze.py rebuilding manifest.json).
    """
    repo_dir = Path(repo_dir)
    problems = []
    top = git_text(repo_dir, "rev-parse", "--show-toplevel", error=error)
    if Path(top).resolve() != repo_dir.resolve():
        problems.append(f"git top level is {top}, not {repo_dir}")
        return problems
    if git_text(repo_dir, "replace", "-l", error=error):
        problems.append("git replace refs exist")
    flagged = [line for line in git_text(repo_dir, "ls-files", "-v", error=error).splitlines()
               if line and (line[0].islower() or line[0] == "S")]
    if flagged:
        problems.append(f"assume-unchanged or skip-worktree entries: {flagged[:5]}")
    status = [line for line in _z(git_bytes(repo_dir, "status", "--porcelain", "-z", "--untracked-files=all", error=error))]
    status = [line for line in status if line[3:] not in allow_changed]
    if status:
        problems.append(f"git tree is dirty; it must be clean: {status[:5]}")
    # Untracked files hidden by .git/info/exclude or a global excludes file: only the repo's .gitignore counts.
    hidden = [p for p in _z(git_bytes(repo_dir, "ls-files", "-z", "--others", "--exclude-per-directory=.gitignore", error=error))
              if p not in allow_changed]
    if hidden:
        problems.append(f"untracked files not ignored by .gitignore: {hidden[:5]}")
    tracked = set(_z(git_bytes(repo_dir, "ls-files", "-z", error=error)))
    ignored = [p for p in _z(git_bytes(repo_dir, "ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--", *frozen_dirs, error=error))
               if not _allowed_ignored(p, tracked)]
    if ignored:
        problems.append(f"ignored files inside frozen folders: {ignored[:5]}")
    return problems


# --------------------------------------------------------------------------- manifest bytes and token


def canonical_bytes(manifest: dict) -> bytes:
    """The one byte form a manifest is written in: sorted keys, 2-space indent, UTF-8, final newline."""
    return (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


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


def tagged_manifest_bytes(repo_dir: Path = REPO_ROOT, freeze_tag: str = FREEZE_TAG) -> bytes:
    commit = tag_commit(repo_dir, freeze_tag)
    if commit is None:
        raise FirewallError(f"freeze tag {freeze_tag} does not exist")
    return git_bytes(repo_dir, "cat-file", "blob", f"{commit}:{MANIFEST_NAME}")


def freeze_token(repo_dir: Path = REPO_ROOT, freeze_tag: str = FREEZE_TAG) -> str:
    """SHA-256 of manifest.json's bytes at the freeze tag, as hex. This is the token `load_test` needs.

    Passing the path of a manifest.json (the older call form) uses the repo it sits in.
    """
    repo_dir = Path(repo_dir)
    if repo_dir.name == MANIFEST_NAME:
        repo_dir = repo_dir.parent
    return hashlib.sha256(tagged_manifest_bytes(repo_dir, freeze_tag)).hexdigest()


def pending_entries(manifest: dict) -> list[str]:
    """Paths of every value in the manifest that is still the string "pending"."""
    out = []

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif node == PENDING:
            out.append(path)

    walk(manifest, "")
    return out


# --------------------------------------------------------------------------- public repo check


def ls_remote(url: str, refs: list[str]) -> str:
    """Credential-free `git ls-remote`, run outside any repo so no local config can redirect the URL."""
    with tempfile.TemporaryDirectory() as empty:
        try:
            done = subprocess.run(
                ["git", "-c", "credential.helper=", "-c", "http.sslVerify=true", "ls-remote", url, *refs],
                cwd=empty, capture_output=True, text=True, check=True, env=git_env(), timeout=LS_REMOTE_TIMEOUT_SECONDS,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise FirewallError(f"git ls-remote {url} failed: is the repo public?") from exc
    return done.stdout


def check_public(head: str, freeze_tag: str = FREEZE_TAG, lister=ls_remote) -> None:
    """The public repo's peeled freeze tag and main must both be HEAD."""
    tag_ref, peeled_ref, main_ref = f"refs/tags/{freeze_tag}", f"refs/tags/{freeze_tag}^{{}}", "refs/heads/main"
    seen = {}
    for line in lister(REGISTRATION_REPO_URL, [tag_ref, peeled_ref, main_ref]).splitlines():
        parts = line.split()
        if len(parts) == 2:
            seen[parts[1]] = parts[0]
    public_tag = seen.get(peeled_ref, seen.get(tag_ref))  # a lightweight tag has no peeled line
    if public_tag != head:
        raise FirewallError(f"the public repo's {freeze_tag} is not at HEAD (public: {public_tag})")
    if seen.get(main_ref) != head:
        raise FirewallError(f"the public repo's main is not at HEAD (public: {seen.get(main_ref)})")


# --------------------------------------------------------------------------- the conditions


def _default_verifier(repo_dir: Path) -> list[str]:
    import freeze  # lazy: freeze imports the whole pipeline

    return freeze.verify_manifest(repo_dir)


def _frozen_dirs():
    import freeze

    return freeze.FROZEN_DIRS


def _check_conditions(token, repo_dir: Path, freeze_tag: str, *, verifier, lister, frozen_dirs) -> None:
    repo_dir = Path(repo_dir)
    if not isinstance(token, str) or not token:
        raise FirewallError("a freeze token is required")

    # Hardened state first: a lying git would make every later answer worthless.
    problems = worktree_problems(repo_dir, frozen_dirs)
    if problems:
        raise FirewallError("; ".join(problems))

    # 1. Token equals SHA-256 of the tagged manifest's bytes.
    raw = tagged_manifest_bytes(repo_dir, freeze_tag)
    if not hmac.compare_digest(token.encode("utf-8"), hashlib.sha256(raw).hexdigest().encode("utf-8")):
        raise FirewallError("token does not match the SHA-256 of manifest.json at the freeze tag")

    # 2. HEAD is the tag's commit.
    head = git_text(repo_dir, "rev-parse", "HEAD")
    if head != tag_commit(repo_dir, freeze_tag):
        raise FirewallError(f"HEAD is not at freeze tag {freeze_tag}")

    # 3. The tagged manifest: canonical, names the public repo, nothing pending, same as on disk, and verifies.
    manifest = read_manifest(raw)
    if manifest.get(REPO_URL_KEY) != REGISTRATION_REPO_URL:
        raise FirewallError(f"manifest repo_url must be exactly {REGISTRATION_REPO_URL}")
    pending = pending_entries(manifest)
    if pending:
        raise FirewallError(f"manifest has pending hashes: {pending}")
    disk = repo_dir / MANIFEST_NAME
    if disk.is_symlink() or not disk.is_file() or disk.read_bytes() != raw:
        raise FirewallError("manifest.json on disk is not the tagged manifest")
    problems = verifier(repo_dir)
    if problems:
        raise FirewallError("manifest does not verify: " + "; ".join(problems[:5]))

    # 4 was checked first (clean, nothing hidden). 5. The public repo shows the tag and main at HEAD.
    check_public(head, freeze_tag, lister)


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
    freeze_tag: str = FREEZE_TAG,
    downloader=None,
    verifier=None,
    lister=None,
    frozen_dirs=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parameterised core of `load_test`. Private: tests inject a temp repo, a fake downloader, verifier and ls-remote."""
    repo_id = _ALIASES.get(dataset, dataset)
    spec = PINNED_TEST_FILES.get(repo_id)
    if spec is None:
        raise FirewallError(f"unknown dataset {dataset!r}; pinned datasets are {sorted(PINNED_TEST_FILES)}")
    _check_conditions(
        token, Path(repo_dir), freeze_tag,
        verifier=verifier or _default_verifier,
        lister=lister or ls_remote,
        frozen_dirs=_frozen_dirs() if frozen_dirs is None else frozen_dirs,
    )

    path = (downloader or _hub_download)(repo_id, spec["filename"], spec["revision"])
    frame = pd.read_parquet(path, columns=["text", "label"])
    row_id = pd.RangeIndex(len(frame), name="row_id")
    texts = pd.DataFrame({"row_id": row_id, "text": frame["text"].to_numpy()})
    gold = pd.DataFrame({"row_id": row_id, "label": frame["label"].to_numpy()})
    return texts, gold


def load_test(dataset: str, token: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (row_id, text) and (row_id, label) for the official test split, as separate frames.

    Always checks this repo, its tagged manifest and the public repo; none of that can be redirected.
    row_id is the row's position in the official file.
    """
    return _load_test(dataset, token, repo_dir=REPO_ROOT, freeze_tag=FREEZE_TAG)


# --------------------------------------------------------------------------- gold kept for the scorer

GOLD_CACHE = REPO_ROOT / "runs" / "private" / "test_gold"


def cache_gold(dataset: str, gold: pd.DataFrame, cache_dir: Path = GOLD_CACHE) -> Path:
    """Keep the gold frame from the first load in gitignored runs/, with its SHA-256 beside it. Never overwrites.

    Call it only after every prediction file is written (the scorer's first run). A rescore then reads it
    with `read_cached_gold` and needs no `load_test`, which refuses once results are committed past the tag.
    """
    if list(gold.columns) != ["row_id", "label"]:
        raise FirewallError("gold must have exactly the columns row_id, label")
    repo_id = _ALIASES.get(dataset, dataset)
    if repo_id not in PINNED_TEST_FILES:
        raise FirewallError(f"unknown dataset {dataset!r}")
    name = repo_id.replace("/", "--")
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path, digest_path = cache_dir / f"{name}.gold.json", cache_dir / f"{name}.gold.sha256"
    raw = json.dumps({"row_id": [int(x) for x in gold["row_id"]], "label": [int(x) for x in gold["label"]]}).encode()
    if path.exists() or digest_path.exists():
        raise FirewallError(f"gold for {repo_id} is already cached; it is never overwritten")
    path.write_bytes(raw)
    digest_path.write_text(hashlib.sha256(raw).hexdigest() + "\n")
    return path


def read_cached_gold(dataset: str, cache_dir: Path = GOLD_CACHE) -> pd.DataFrame:
    """The gold frame kept by `cache_gold`, refused if its bytes no longer match the recorded SHA-256."""
    repo_id = _ALIASES.get(dataset, dataset)
    name = repo_id.replace("/", "--")
    path, digest_path = Path(cache_dir) / f"{name}.gold.json", Path(cache_dir) / f"{name}.gold.sha256"
    if not path.is_file() or not digest_path.is_file():
        raise FirewallError(f"no cached gold for {repo_id}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest_path.read_text().strip():
        raise FirewallError(f"cached gold for {repo_id} does not match its recorded SHA-256")
    data = json.loads(raw)
    return pd.DataFrame({"row_id": data["row_id"], "label": data["label"]})
