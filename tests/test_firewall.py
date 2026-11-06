"""Test firewall (VALIDATION.md, Guards). Nothing here reads, fetches or downloads test data.

This file is allowed to contain the forbidden strings because it defines them.
"""
import ast
import hashlib
import inspect
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest
from huggingface_hub import constants as hf_constants

import firewall
from firewall import (
    REGISTRATION_REPO_URL,
    FirewallError,
    _hub_download,
    _load_test,
    canonical_bytes,
    freeze_token,
    load_test,
)

REPO = Path(__file__).resolve().parents[1]
PINNED = ("PolyAI/banking77", "fancyzhx/ag_news")
FREEZE_TAG = "freeze-v1"

# --------------------------------------------------------------------------- helpers


def git(repo, *args):
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args]
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True).stdout


def frozen(repo=REPO):
    """Once the freeze tag exists, test files and result files are expected."""
    return subprocess.run(
        ["git", "rev-parse", "-q", "--verify", f"refs/tags/{FREEZE_TAG}"], cwd=repo, capture_output=True
    ).returncode == 0


def repo_files(repo):
    """Tracked plus untracked-but-not-ignored files, as repo-relative posix paths."""
    out = git(repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    return sorted(f for f in out.split("\0") if f and (Path(repo) / f).is_file())


def cache_dirs(dataset):
    org, name = dataset.split("/")
    return [f"datasets--{org}--{name}", f"datasets--{name}"]  # current and legacy canonical names


def find_test_parquets(data_dir, hf_cache_dir):
    """Test parquets under data/ (project-owned, so any) and in the HF cache snapshots of the two pinned datasets."""
    found = []
    if Path(data_dir).exists():
        found += sorted(Path(data_dir).rglob("test-*.parquet"))
    for dataset in PINNED:
        for d in cache_dirs(dataset):
            snapshots = Path(hf_cache_dir) / d / "snapshots"
            if snapshots.exists():
                found += sorted(snapshots.rglob("test-*.parquet"))
    return found


FORBIDDEN = [re.compile(r"""split\s*=\s*["']test"""), re.compile(r"test-0000"), re.compile(r"load_dataset\(")]
ALLOWED_FORBIDDEN = {"src/firewall.py", "tests/test_firewall.py"}  # plus *.md docs


def forbidden_string_hits(repo):
    hits = []
    for f in repo_files(repo):
        if f in ALLOWED_FORBIDDEN or f.endswith(".md"):
            continue
        try:
            text = (Path(repo) / f).read_text()
        except UnicodeDecodeError:
            continue
        hits += [f"{f}: {p.pattern}" for p in FORBIDDEN if p.search(text)]
    return hits


def forbidden_artifacts(repo, is_frozen):
    bad = []
    for f in repo_files(repo):
        parts = PurePosixPath(f).parts
        if ".cache" in parts or parts[0] == "runs":
            bad.append(f)
        elif not is_frozen and (parts[0] == "results" or parts[-1] == "scored.lock"):
            bad.append(f)
    return bad


# Every firewall name that reaches test text or gold, now or in an earlier version. A predictor using any of
# them (as a name, an attribute or a string for getattr/import) after importing firewall is flagged.
FIREWALL_GOLD_NAMES = {"load_test", "_load_test", "_hub_download", "_check_conditions", "PINNED_TEST_FILES",
                       "cache_gold", "read_cached_gold"}


def reads_gold(source):
    """True if the source imports a module named gold, or imports firewall and uses any gold-reading name."""
    imports_firewall = uses_gold_name = False
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if "gold" in parts:
                    return True
                imports_firewall |= parts[-1] == "firewall"
        elif isinstance(node, ast.ImportFrom):
            parts = (node.module or "").split(".")
            names = [a.name for a in node.names]
            if "gold" in parts or "gold" in names:
                return True
            if parts[-1] == "firewall":
                imports_firewall = True
                if "*" in names:
                    return True
            imports_firewall |= "firewall" in names
            uses_gold_name |= bool(FIREWALL_GOLD_NAMES & set(names))
        elif isinstance(node, ast.Attribute):
            uses_gold_name |= node.attr in FIREWALL_GOLD_NAMES
        elif isinstance(node, ast.Name):
            uses_gold_name |= node.id in FIREWALL_GOLD_NAMES
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "gold":
                return True  # importlib.import_module("gold"), __import__("gold")
            uses_gold_name |= node.value in FIREWALL_GOLD_NAMES
    return imports_firewall and uses_gold_name


def test_gold_names_cover_every_public_firewall_reader():
    # Any firewall function whose source downloads or reads parquet must be listed.
    for name, obj in vars(firewall).items():
        if inspect.isfunction(obj) and obj.__module__ == "firewall":
            src = inspect.getsource(obj)
            if "read_parquet" in src or "hf_hub_download" in src:
                assert name in FIREWALL_GOLD_NAMES, name


def predict_files(repo):
    return [f for f in repo_files(repo) if f.endswith(".py") and PurePosixPath(f).name.startswith("predict")]


def write_manifest(path, manifest):
    Path(path).write_bytes(canonical_bytes(manifest))


FROZEN_MANIFEST = {"files": {"src/a.py": "0" * 64}, "repo_url": REGISTRATION_REPO_URL, "train_parquet": {"banking77": {"sha256": "ab" * 32}}}
TEST_DIRS = ("src",)


def make_repo(tmp_path, *, manifest=None, tag=True, annotated=False):
    """A temp repo frozen like the real one: canonical manifest naming the public repo, committed and tagged."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    write_manifest(repo / "manifest.json", FROZEN_MANIFEST if manifest is None else manifest)
    (repo / "src").mkdir()
    (repo / "src" / "a.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text("runs/\n__pycache__/\n*.log\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "freeze")
    if tag:
        if annotated:
            git(repo, "tag", "-a", FREEZE_TAG, "-m", "freeze")
        else:
            git(repo, "tag", FREEZE_TAG)
    return repo


def head_of(repo):
    return git(repo, "rev-parse", "HEAD").strip()


def public_lister(tag_sha, main_sha, peeled=None):
    """A fake ls-remote: what the public repo would report."""
    calls = []

    def lister(url, refs):
        calls.append((url, list(refs)))
        lines = [f"{main_sha}\trefs/heads/main", f"{tag_sha}\trefs/tags/{FREEZE_TAG}"]
        if peeled:
            lines.append(f"{peeled}\trefs/tags/{FREEZE_TAG}^{{}}")
        return "\n".join(lines) + "\n"

    lister.calls = calls
    return lister


def published(repo):
    return public_lister(head_of(repo), head_of(repo))


def token_for(repo):
    return freeze_token(repo)


def load(repo, tmp_path, *, token=None, lister=None, verifier=None, dataset=None):
    dl = FakeDownloader(tmp_path)
    out = _load_test(
        dataset or PINNED[0], token_for(repo) if token is None else token, repo_dir=repo, downloader=dl,
        verifier=verifier or (lambda r: []), lister=lister or published(repo), frozen_dirs=TEST_DIRS,
    )
    return out, dl


def assert_refused(repo, tmp_path, match, **kw):
    dl = FakeDownloader(tmp_path)
    token = kw.pop("token") if "token" in kw else token_for(repo)
    with pytest.raises(FirewallError, match=match):
        _load_test(
            PINNED[0], token, repo_dir=repo, downloader=dl,
            verifier=kw.pop("verifier", lambda r: []), lister=kw.pop("lister", None) or published(repo),
            frozen_dirs=TEST_DIRS,
        )
    assert dl.calls == []


class FakeDownloader:
    """Stands in for hf_hub_download: writes a small synthetic parquet, never touches the network."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.calls = []

    def __call__(self, repo_id, filename, revision):
        self.calls.append((repo_id, filename, revision))
        path = self.tmp_path / "fake.parquet"
        pd.DataFrame({"text": ["alpha", "beta", "gamma"], "label": [2, 0, 1]}).to_parquet(path)
        return str(path)


# ------------------------------------------------- no test parquet before the freeze


def test_no_test_parquet_on_disk_before_freeze():
    if frozen():
        pytest.skip("freeze tag exists; the test files are allowed after the freeze")
    found = find_test_parquets(REPO / "data", hf_constants.HF_HUB_CACHE)
    assert not found, f"test parquet on disk before the freeze: {found}"


def test_detects_test_parquet_under_data_dir(tmp_path):
    (tmp_path / "data" / "banking77").mkdir(parents=True)
    bad = tmp_path / "data" / "banking77" / "test-00000-of-00001.parquet"
    bad.touch()
    (tmp_path / "data" / "banking77" / "train-00000-of-00001.parquet").touch()
    assert find_test_parquets(tmp_path / "data", tmp_path / "hub") == [bad]


@pytest.mark.parametrize("dataset", PINNED)
def test_detects_test_parquet_in_injected_hf_cache(tmp_path, dataset):
    snap = tmp_path / "hub" / cache_dirs(dataset)[0] / "snapshots" / "abc123" / "data"
    snap.mkdir(parents=True)
    bad = snap / "test-00000-of-00001.parquet"
    bad.touch()
    assert find_test_parquets(tmp_path / "data", tmp_path / "hub") == [bad]


def test_detects_dangling_symlink_in_hf_cache(tmp_path):
    snap = tmp_path / "hub" / cache_dirs(PINNED[1])[0] / "snapshots" / "abc123" / "data"
    snap.mkdir(parents=True)
    link = snap / "test-00000-of-00001.parquet"
    link.symlink_to(tmp_path / "missing-blob")
    assert find_test_parquets(tmp_path / "data", tmp_path / "hub") == [link]


def test_ignores_train_files_and_other_datasets_in_cache(tmp_path):
    other = tmp_path / "hub" / "datasets--someone--else" / "snapshots" / "abc" / "data"
    other.mkdir(parents=True)
    (other / "test-00000-of-00001.parquet").touch()
    mine = tmp_path / "hub" / cache_dirs(PINNED[0])[0] / "snapshots" / "abc" / "data"
    mine.mkdir(parents=True)
    (mine / "train-00000-of-00001.parquet").touch()
    assert find_test_parquets(tmp_path / "data", tmp_path / "hub") == []


# ----------------------------------------------------- forbidden strings in the repo


def test_no_forbidden_strings_outside_firewall_test_and_docs():
    assert forbidden_string_hits(REPO) == []


def test_forbidden_string_scan_detects_each_pattern(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "a.py").write_text('x = ds(split="test")\n')
    (repo / "b.py").write_text("name = 'test-00000-of-00001.parquet'\n")
    (repo / "c.py").write_text("from datasets import load_dataset\nload_dataset('x')\n")
    (repo / "NOTES.md").write_text('docs may say load_dataset( and split="test"\n')
    (repo / "src" / "firewall.py").write_text("load_dataset(\n")
    hits = forbidden_string_hits(repo)
    assert sorted({h.split(":")[0] for h in hits}) == ["a.py", "b.py", "c.py"]


# ------------------------------------------------------ no cache or prior results


def test_no_cache_or_prior_results_in_repo():
    assert forbidden_artifacts(REPO, frozen()) == []


def test_artifact_scan_detects_cache_and_results(tmp_path):
    repo = make_repo(tmp_path)
    for rel in [".cache/jev/a.json", "sub/.cache/b.json", "results/r.json", "scored.lock"]:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("{}")
    (repo / "ok.py").write_text("")
    assert forbidden_artifacts(repo, is_frozen=False) == [".cache/jev/a.json", "results/r.json", "scored.lock", "sub/.cache/b.json"]
    assert forbidden_artifacts(repo, is_frozen=True) == [".cache/jev/a.json", "sub/.cache/b.json"]


# ----------------------------------------------------- predictors cannot read gold


def test_no_predict_file_reads_gold():
    offenders = [f for f in predict_files(REPO) if reads_gold((REPO / f).read_text())]
    assert offenders == []


@pytest.mark.parametrize(
    "source",
    [
        "import firewall\ntext, gold = firewall.load_test('PolyAI/banking77', t)\n",
        "from firewall import load_test\n",
        "from src.firewall import load_test as lt\n",
        "import firewall as fw\nrows = fw.load_test('x', t)\n",
        "from firewall import *\n",
        "import gold\n",
        "from gold import labels\n",
        "from data.gold import labels\n",
        "from data import gold\n",
        "import importlib\nimportlib.import_module('gold')\n",
        "import firewall\ngold = firewall.read_cached_gold('banking77')\n",
        "import firewall\nf = firewall._load_test\n",
        "from firewall import _hub_download\n",
        "import firewall\nf = getattr(firewall, 'PINNED_TEST_FILES')\n",
    ],
)
def test_gold_reader_check_flags_violations(tmp_path, source):
    repo = make_repo(tmp_path)
    (repo / "predict_fake.py").write_text(source)
    assert [f for f in predict_files(repo) if reads_gold((repo / f).read_text())] == ["predict_fake.py"]


@pytest.mark.parametrize(
    "source",
    ["import pandas as pd\nfrom firewall import FirewallError\n", "import numpy as np\ndef predict(texts):\n    return texts\n"],
)
def test_gold_reader_check_allows_clean_predictors(tmp_path, source):
    repo = make_repo(tmp_path)
    (repo / "predict_clean.py").write_text(source)
    (repo / "helper.py").write_text("import gold\n")  # not a predictor, so not this check's concern
    assert predict_files(repo) == ["predict_clean.py"]
    assert not reads_gold(source)


# ------------------------------------------------------------- load_test conditions


def test_freeze_token_is_sha256_of_tagged_manifest_bytes(tmp_path):
    repo = make_repo(tmp_path)
    tagged = git(repo, "show", f"{FREEZE_TAG}:manifest.json").encode()
    assert freeze_token(repo) == hashlib.sha256(tagged).hexdigest()
    assert freeze_token(repo / "manifest.json") == freeze_token(repo)  # the older call form


def test_freeze_token_reads_the_tag_not_the_working_file(tmp_path):
    repo = make_repo(tmp_path)
    before = freeze_token(repo)
    write_manifest(repo / "manifest.json", {**FROZEN_MANIFEST, "files": {}})
    assert freeze_token(repo) == before


def test_freeze_token_needs_the_tag(tmp_path):
    with pytest.raises(FirewallError, match="tag"):
        freeze_token(make_repo(tmp_path, tag=False))


@pytest.mark.parametrize("token", ["", "not-a-token", None])
def test_load_test_without_valid_token_raises_on_real_repo(token):
    with pytest.raises(FirewallError):
        load_test(PINNED[0], token)


def test_load_test_on_real_repo_refuses_before_freeze():
    # A well-formed but wrong token is refused whatever state the real repo is in.
    with pytest.raises(FirewallError):
        load_test(PINNED[0], "0" * 64)


def test_public_load_test_signature_is_dataset_and_token_only():
    assert list(inspect.signature(load_test).parameters) == ["dataset", "token"]
    for kwargs in ({"repo_dir": "x"}, {"freeze_tag": "x"}, {"downloader": print}, {"verifier": print}, {"lister": print}):
        with pytest.raises(TypeError):
            load_test(PINNED[0], "0" * 64, **kwargs)


def test_load_test_ignores_a_throwaway_frozen_repo(tmp_path):
    # A temp repo with its own manifest and tag cannot satisfy the public function.
    repo = make_repo(tmp_path)
    with pytest.raises(FirewallError):
        load_test(PINNED[0], token_for(repo))


@pytest.mark.parametrize(
    "args",
    [
        ("PolyAI/banking77", "data/train-00000-of-00001.parquet", "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"),
        ("PolyAI/banking77", "data/test-00000-of-00001.parquet", "main"),
        ("someone/else", "data/test-00000-of-00001.parquet", "1846984189f2bc0ac51c6fc3a600fa9e1714e0c6"),
    ],
)
def test_hub_download_refuses_anything_but_pinned_test_files(args):
    with pytest.raises(FirewallError, match="not a pinned test file"):
        _hub_download(*args)


@pytest.mark.parametrize("dataset", PINNED)
def test_load_test_success_splits_text_from_gold(tmp_path, dataset):
    repo = make_repo(tmp_path)
    lister = published(repo)
    (texts, gold), dl = load(repo, tmp_path, dataset=dataset, lister=lister)
    assert list(texts.columns) == ["row_id", "text"]
    assert list(gold.columns) == ["row_id", "label"]
    assert texts["row_id"].tolist() == gold["row_id"].tolist() == [0, 1, 2]
    assert texts["text"].tolist() == ["alpha", "beta", "gamma"]
    assert gold["label"].tolist() == [2, 0, 1]
    assert [(c[0], c[2]) for c in dl.calls] == [(dataset, firewall.PINNED_TEST_FILES[dataset]["revision"])]
    assert lister.calls == [(REGISTRATION_REPO_URL, [f"refs/tags/{FREEZE_TAG}", f"refs/tags/{FREEZE_TAG}^{{}}", "refs/heads/main"])]


def test_load_test_success_with_annotated_tag(tmp_path):
    repo = make_repo(tmp_path, annotated=True)
    tag_object = git(repo, "rev-parse", f"refs/tags/{FREEZE_TAG}").strip()
    assert tag_object != head_of(repo)
    load(repo, tmp_path, lister=public_lister(tag_object, head_of(repo), peeled=head_of(repo)))


def test_load_test_allows_bytecode_of_tracked_sources(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "src" / "__pycache__").mkdir()
    (repo / "src" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\0")
    load(repo, tmp_path)


@pytest.mark.parametrize("short,full", [("banking77", PINNED[0]), ("ag_news", PINNED[1])])
def test_load_test_accepts_short_dataset_names(tmp_path, short, full):
    repo = make_repo(tmp_path)
    _, dl = load(repo, tmp_path, dataset=short)
    assert dl.calls[0][0] == full


def test_load_test_rejects_unknown_dataset(tmp_path):
    repo = make_repo(tmp_path)
    dl = FakeDownloader(tmp_path)
    with pytest.raises(FirewallError, match="dataset"):
        _load_test("someone/else", token_for(repo), repo_dir=repo, downloader=dl, verifier=lambda r: [], lister=published(repo))
    assert dl.calls == []


def test_load_test_default_downloader_cannot_run_in_tests(tmp_path):
    # Every condition holds, no downloader injected: the conftest guard stops the real fetch.
    repo = make_repo(tmp_path)
    with pytest.raises(RuntimeError, match="blocked"):
        _load_test(PINNED[0], token_for(repo), repo_dir=repo, verifier=lambda r: [], lister=published(repo), frozen_dirs=TEST_DIRS)


# 1. token


@pytest.mark.parametrize("token", ["0" * 64, "", None])
def test_load_test_rejects_wrong_token(tmp_path, token):
    assert_refused(make_repo(tmp_path), tmp_path, "token", token=token)


# 2. HEAD at the tag


def test_load_test_rejects_missing_freeze_tag(tmp_path):
    repo = make_repo(tmp_path, tag=False)
    assert_refused(repo, tmp_path, "tag", token="0" * 64)


def test_load_test_rejects_head_not_at_freeze_tag(tmp_path):
    repo = make_repo(tmp_path)
    token = token_for(repo)
    (repo / "later.txt").write_text("a later commit\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "after freeze")
    assert_refused(repo, tmp_path, "HEAD is not at freeze tag", token=token)


def test_load_test_rejects_a_moved_tag(tmp_path):
    # Code changed after the freeze and the tag forced onto it: the public tag still names the old commit.
    repo = make_repo(tmp_path)
    frozen_commit = head_of(repo)
    (repo / "src" / "a.py").write_text("x = 2\n")
    git(repo, "commit", "-q", "-am", "post-freeze change")
    git(repo, "tag", "-f", FREEZE_TAG)
    assert_refused(repo, tmp_path, "public repo's freeze-v1", lister=public_lister(frozen_commit, head_of(repo)))


# 3. the tagged manifest


@pytest.mark.parametrize(
    "url",
    [
        None,
        "https://github.com/aarmens702-hub/jev-benchmark/",
        "https://github.com/aarmens702-hub/jev-benchmark.git",
        "http://github.com/aarmens702-hub/jev-benchmark",
        "https://github.com/someone/jev-benchmark",
        "https://github.com/AARMENS702-HUB/jev-benchmark",
        "",
    ],
)
def test_load_test_rejects_a_different_or_missing_repo_url(tmp_path, url):
    manifest = {k: v for k, v in FROZEN_MANIFEST.items() if k != "repo_url"}
    if url is not None:
        manifest["repo_url"] = url
    assert_refused(make_repo(tmp_path, manifest=manifest), tmp_path, "repo_url")


def test_load_test_rejects_osf_only_manifest(tmp_path):
    manifest = {k: v for k, v in FROZEN_MANIFEST.items() if k != "repo_url"}
    manifest["osf_id"] = "abc12"
    assert_refused(make_repo(tmp_path, manifest=manifest), tmp_path, "repo_url")


def test_load_test_rejects_any_pending_hash(tmp_path):
    manifest = {**FROZEN_MANIFEST, "train_parquet": {"banking77": {"sha256": "pending"}}}
    assert_refused(make_repo(tmp_path, manifest=manifest), tmp_path, "pending")


def test_load_test_rejects_a_manifest_that_does_not_verify(tmp_path):
    repo = make_repo(tmp_path)
    seen = []

    def verifier(r):
        seen.append(Path(r))
        return ["files.src/a.py: changed"]

    assert_refused(repo, tmp_path, "does not verify", verifier=verifier)
    assert seen == [repo]


@pytest.mark.parametrize(
    "raw",
    [
        b'{"repo_url": "https://github.com/aarmens702-hub/jev-benchmark"}',  # not indented
        b'{\n  "a": 1,\n  "a": 2,\n  "repo_url": "https://github.com/aarmens702-hub/jev-benchmark"\n}\n',  # duplicate keys
        b"[]\n",
        b"not json",
    ],
)
def test_load_test_rejects_non_canonical_tagged_manifest(tmp_path, raw):
    repo = make_repo(tmp_path, tag=False)
    (repo / "manifest.json").write_bytes(raw)
    git(repo, "commit", "-q", "-am", "bad manifest")
    git(repo, "tag", FREEZE_TAG)
    assert_refused(repo, tmp_path, "manifest")


def test_load_test_rejects_a_symlinked_manifest(tmp_path):
    repo = make_repo(tmp_path, tag=False)
    (repo / "runs").mkdir()
    (repo / "manifest.json").rename(repo / "runs" / "m.json")
    (repo / "manifest.json").symlink_to("runs/m.json")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "symlink")
    git(repo, "tag", FREEZE_TAG)
    assert_refused(repo, tmp_path, "manifest")


def test_load_test_rejects_disk_manifest_hidden_by_a_clean_filter(tmp_path):
    # A clean filter makes git see the tagged manifest while the file on disk says something else.
    repo = make_repo(tmp_path)
    original = tmp_path / "orig.json"
    original.write_bytes((repo / "manifest.json").read_bytes())
    git(repo, "config", "filter.hide.clean", f"cat {original}")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("manifest.json filter=hide\n")
    write_manifest(repo / "manifest.json", {**FROZEN_MANIFEST, "files": {}})
    git(repo, "add", "manifest.json")  # stores the filtered (tagged) bytes, so nothing is staged
    assert git(repo, "status", "--porcelain") == ""  # git is fooled
    assert_refused(repo, tmp_path, "differ from the committed bytes")


# 4. clean tree, nothing hidden from git


def test_load_test_rejects_dirty_tracked_file(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "src" / "a.py").write_text("x = 9\n")
    assert_refused(repo, tmp_path, "dirty")


def test_load_test_rejects_untracked_file(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "scratch.txt").write_text("untracked change")
    assert_refused(repo, tmp_path, "dirty")


@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"])
def test_load_test_rejects_hidden_index_edits(tmp_path, flag):
    repo = make_repo(tmp_path)
    git(repo, "update-index", flag, "src/a.py")
    (repo / "src" / "a.py").write_text("x = 666\n")
    assert_refused(repo, tmp_path, "assume-unchanged or skip-worktree")


def test_load_test_rejects_show_untracked_files_no(tmp_path):
    repo = make_repo(tmp_path)
    git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "evil.py").write_text("print(1)\n")
    assert_refused(repo, tmp_path, "dirty")


def test_load_test_rejects_file_hidden_by_info_exclude(tmp_path):
    repo = make_repo(tmp_path)
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "exclude").write_text("evil2.py\n")
    (repo / "evil2.py").write_text("print(2)\n")
    assert_refused(repo, tmp_path, "not ignored by .gitignore")


def test_load_test_rejects_ignored_file_inside_a_frozen_folder(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "src" / "debug.log").write_text("ignored by *.log\n")
    assert_refused(repo, tmp_path, "ignored files inside frozen folders")
    (repo / "src" / "debug.log").unlink()
    (repo / "src" / "__pycache__").mkdir()
    (repo / "src" / "__pycache__" / "evil.cpython-312.pyc").write_bytes(b"\0")  # no tracked evil.py
    assert_refused(repo, tmp_path, "ignored files inside frozen folders")


def test_load_test_ignores_git_dir_and_work_tree_variables(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    clean = tmp_path / "clean"
    subprocess.run(["git", "clone", "-q", str(repo), str(clean)], check=True, capture_output=True)
    (repo / "src" / "a.py").write_text("x = 999\n")  # the real tree is dirty
    monkeypatch.setenv("GIT_DIR", str(clean / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(clean))
    assert_refused(repo, tmp_path, "dirty")


def test_load_test_rejects_core_worktree_redirect(tmp_path):
    repo = make_repo(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    git(repo, "config", "core.worktree", str(other))
    assert_refused(repo, tmp_path, "top level")


def test_load_test_rejects_a_subdirectory_as_repo(tmp_path):
    repo = make_repo(tmp_path)
    assert_refused(repo / "src", tmp_path, "top level", token="0" * 64)


def test_load_test_rejects_git_replace(tmp_path):
    repo = make_repo(tmp_path)
    frozen_commit = head_of(repo)
    git(repo, "checkout", "-q", "-b", "x")
    (repo / "src" / "a.py").write_text("x = 3\n")
    git(repo, "commit", "-q", "-am", "evil")
    evil = head_of(repo)
    git(repo, "checkout", "-q", "--detach", frozen_commit)
    git(repo, "replace", frozen_commit, evil)
    git(repo, "reset", "-q", "--hard")
    assert_refused(repo, tmp_path, "replace", lister=public_lister(frozen_commit, frozen_commit))


# 5. the public repo


def test_load_test_rejects_when_the_repo_is_not_public(tmp_path):
    repo = make_repo(tmp_path)

    def private(url, refs):
        raise FirewallError("git ls-remote failed: is the repo public?")

    assert_refused(repo, tmp_path, "public", lister=private)


def test_load_test_rejects_public_main_not_at_head(tmp_path):
    repo = make_repo(tmp_path)
    assert_refused(repo, tmp_path, "main", lister=public_lister(head_of(repo), "1" * 40))


def test_load_test_rejects_annotated_public_tag_without_peeled_commit(tmp_path):
    repo = make_repo(tmp_path, annotated=True)
    tag_object = git(repo, "rev-parse", f"refs/tags/{FREEZE_TAG}").strip()
    assert_refused(repo, tmp_path, "public repo's freeze-v1", lister=public_lister(tag_object, head_of(repo)))


def test_check_public_reads_exact_ref_names_only():
    head = "a" * 40
    tricky = f"{head}\trefs/heads/x/refs/heads/main\n{head}\trefs/tags/x/refs/tags/{FREEZE_TAG}\n"
    with pytest.raises(FirewallError):
        firewall.check_public(head, lister=lambda url, refs: tricky)
    firewall.check_public(head, lister=public_lister(head, head))


def test_ls_remote_is_credential_free_and_outside_the_repo(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    for name, value in {"GIT_ASKPASS": "/bin/echo", "SSH_ASKPASS": "/bin/echo", "DISPLAY": ":0", "https_proxy": "http://p.invalid",
                        "HTTPS_PROXY": "http://p.invalid", "SSL_CERT_FILE": "/x.pem", "CURL_CA_BUNDLE": "/x.pem",
                        "XDG_CONFIG_HOME": "/tmp/xdg"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(firewall.subprocess, "run", fake_run)
    firewall.ls_remote(REGISTRATION_REPO_URL, ["refs/heads/main"])
    cmd = seen["cmd"]
    for flag in ("credential.helper=", "core.askPass=", "http.extraHeader=", "http.sslVerify=true"):
        assert flag in cmd and cmd[cmd.index(flag) - 1] == "-c", flag
    assert REGISTRATION_REPO_URL in cmd and "ls-remote" in cmd
    env = seen["env"]
    assert set(env) == {"PATH", "HOME", "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_GLOBAL", "GIT_TERMINAL_PROMPT", "GIT_CEILING_DIRECTORIES"}
    assert env["GIT_TERMINAL_PROMPT"] == "0" and env["GIT_CONFIG_GLOBAL"] == os.devnull and env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["HOME"] != os.path.expanduser("~") and env["HOME"] == seen["cwd"]
    assert env["GIT_CEILING_DIRECTORIES"] == str(Path(seen["cwd"]).parent)
    assert Path(seen["cwd"]).resolve() != REPO.resolve() and seen["timeout"] > 0


# A local git-over-HTTP stand-in, run as its own process (the test process has sockets blocked). It answers 401
# unless a request carries credentials or an extra header, and logs what each request carried.
SERVER = r"""
import http.server, json, sys
log = open(sys.argv[1], "a", buffering=1)
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        log.write(json.dumps({"path": self.path, "auth": self.headers.get("Authorization"),
                              "extra": self.headers.get("X-Extra")}) + "\n")
        if not self.headers.get("Authorization") and not self.headers.get("X-Extra"):
            self.send_response(401); self.send_header("WWW-Authenticate", 'Basic realm="x"'); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Type", "application/x-git-upload-pack-advertisement")
        self.end_headers(); self.wfile.write(b"001e# service=git-upload-pack\n0000")
    def log_message(self, *a): pass
srv = http.server.HTTPServer(("127.0.0.1", 0), H)
print(srv.server_port, flush=True)
srv.serve_forever()
"""


@pytest.fixture
def git_server(tmp_path):
    import sys

    script, log = tmp_path / "server.py", tmp_path / "requests.jsonl"
    script.write_text(SERVER)
    log.touch()
    proc = subprocess.Popen([sys.executable, "-I", str(script), str(log)], stdout=subprocess.PIPE, text=True)
    port = int(proc.stdout.readline())

    def requests():
        return [json.loads(line) for line in log.read_text().splitlines() if line]

    def clear():
        log.write_text("")

    try:
        yield f"http://127.0.0.1:{port}", requests, clear
    finally:
        proc.kill()
        proc.wait()


def _ls_remote_refused(url):
    with pytest.raises(FirewallError, match="is the repo public"):
        firewall.ls_remote(url, ["refs/heads/main"])


def test_ls_remote_sends_no_credentials_to_a_private_repo(tmp_path, monkeypatch, git_server):
    base, requests, clear = git_server
    url = f"{base}/aarmens702-hub/jev-benchmark"

    _ls_remote_refused(url)  # baseline: the server is reached and refuses, so the checks below are not vacuous
    assert requests() and not any(r["auth"] or r["extra"] for r in requests())

    home = tmp_path / "home"
    home.mkdir()
    (home / ".netrc").write_text("machine 127.0.0.1 login netrcuser password netrcpass\n")
    (home / ".netrc").chmod(0o600)
    (home / ".gitconfig").write_text("[http]\n\textraHeader = X-Extra: homegitconfig\n")
    xdg = tmp_path / "xdg"
    (xdg / "git").mkdir(parents=True)
    (xdg / "git" / "config").write_text("[http]\n\textraHeader = X-Extra: xdg\n")
    askpass = tmp_path / "askpass.sh"
    askpass.write_text("#!/bin/sh\necho askpassed\n")
    askpass.chmod(0o755)
    for env in ({"HOME": str(home)}, {"XDG_CONFIG_HOME": str(xdg)},
                {"SSH_ASKPASS": str(askpass), "GIT_ASKPASS": str(askpass), "DISPLAY": ":0", "SSH_ASKPASS_REQUIRE": "force"}):
        clear()
        with monkeypatch.context() as m:
            for k, v in env.items():
                m.setenv(k, v)
            _ls_remote_refused(url)
        assert requests() and not any(r["auth"] or r["extra"] for r in requests()), env


def test_ls_remote_ignores_a_repo_around_tmpdir_and_proxies(tmp_path, monkeypatch, git_server):
    import tempfile

    base, requests, clear = git_server
    evil = tmp_path / "evilrepo"
    subprocess.run(["git", "init", "-q", str(evil)], check=True)
    git(evil, "config", f"url.{base}/.insteadOf", "https://example.invalid/")
    git(evil, "config", "http.extraHeader", "X-Extra: repo-local")
    (evil / "tmp").mkdir()
    monkeypatch.setenv("TMPDIR", str(evil / "tmp"))
    monkeypatch.setattr(tempfile, "tempdir", None)
    _ls_remote_refused("https://example.invalid/aarmens702-hub/jev-benchmark")
    assert requests() == []  # the repo's insteadOf never applied

    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.delenv("TMPDIR")
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, base)
    _ls_remote_refused("http://example.invalid/aarmens702-hub/jev-benchmark")
    assert requests() == []  # no proxy was used


def test_pinned_revisions_match_design():
    design = (REPO / "DESIGN.md").read_text()
    for spec in firewall.PINNED_TEST_FILES.values():
        assert spec["revision"] in design


# --------------------------------------------------------------------------- content check and git binary


def test_content_check_handles_quoted_and_newline_paths(tmp_path):
    repo = make_repo(tmp_path, tag=False)
    odd = ['"quoted.py', "new\nline.py", "space name.py", "unié.py"]
    for name in odd:
        (repo / "src" / name).write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "odd names")
    assert firewall._content_problems(repo, (), FirewallError) == []
    for name in odd:
        (repo / "src" / name).write_text("x = 2\n")
        problems = firewall._content_problems(repo, (), FirewallError)
        assert problems and repr(f"src/{name}") in problems[0], name  # the message lists paths as a Python list
        (repo / "src" / name).write_text("x = 1\n")


def test_content_check_covers_symlinks_and_missing_files(tmp_path):
    repo = make_repo(tmp_path, tag=False)
    (repo / "src" / "link.py").symlink_to("a.py")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "link")
    assert firewall._content_problems(repo, (), FirewallError) == []
    (repo / "src" / "link.py").unlink()
    (repo / "src" / "link.py").symlink_to("../manifest.json")
    assert "src/link.py" in firewall._content_problems(repo, (), FirewallError)[0]
    (repo / "src" / "link.py").unlink()
    assert "src/link.py" in firewall._content_problems(repo, (), FirewallError)[0]


@pytest.fixture
def fresh_git_lookup():
    firewall.git_executable.cache_clear()
    yield
    firewall.git_executable.cache_clear()


def test_git_inside_the_repo_is_refused(monkeypatch, fresh_git_lookup):
    monkeypatch.setattr(firewall.shutil, "which", lambda name: str(firewall.REPO_ROOT / "tools" / "git"))
    with pytest.raises(FirewallError, match="inside the repo"):
        firewall.git_executable()
    firewall.git_executable.cache_clear()
    monkeypatch.setattr(firewall.shutil, "which", lambda name: None)
    with pytest.raises(FirewallError, match="not on PATH"):
        firewall.git_executable()


def test_git_identity_records_path_and_version(fresh_git_lookup):
    identity = firewall.git_identity()
    assert Path(identity["path"]).is_absolute() and firewall.REPO_ROOT.resolve() not in Path(identity["path"]).parents
    assert identity["version"].startswith("git version")
