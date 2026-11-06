"""Test firewall (VALIDATION.md, Guards). Nothing here reads, fetches or downloads test data.

This file is allowed to contain the forbidden strings because it defines them.
"""
import ast
import hashlib
import inspect
import json
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
    manifest_hash,
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


def reads_gold(source):
    """True if the source imports a module named gold, or imports firewall and uses load_test."""
    imports_firewall = uses_load_test = False
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
            uses_load_test |= "load_test" in names
        elif isinstance(node, ast.Attribute):
            uses_load_test |= node.attr == "load_test"
        elif isinstance(node, ast.Name):
            uses_load_test |= node.id == "load_test"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "gold":
                return True  # importlib.import_module("gold"), __import__("gold")
            uses_load_test |= node.value == "load_test"
    return imports_firewall and uses_load_test


def predict_files(repo):
    return [f for f in repo_files(repo) if f.endswith(".py") and PurePosixPath(f).name.startswith("predict")]


def write_manifest(path, manifest):
    Path(path).write_bytes(canonical_bytes(manifest))


def tag_commit(repo):
    return git(repo, "rev-parse", f"refs/tags/{FREEZE_TAG}^{{commit}}").strip()


def register(repo, *, url=REGISTRATION_REPO_URL, commit=None, commit_it=True):
    """Fill the registration after tagging, as freeze.py --registration does, then commit it on the tag."""
    manifest = json.loads((repo / "manifest.json").read_text())
    manifest["registration"] = {"repo_url": url, "commit": tag_commit(repo) if commit is None else commit}
    write_manifest(repo / "manifest.json", manifest)
    if commit_it:
        git(repo, "commit", "-q", "-am", "registration")
    return manifest


def make_repo(tmp_path, *, manifest=None, tag=True, registered=True):
    """A temp repo frozen like the real one: manifest with registration null, tag, then a registration commit."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    frozen_manifest = {"files": {"a.py": "0" * 64}, "registration": None} if manifest is None else manifest
    write_manifest(repo / "manifest.json", frozen_manifest)
    (repo / "a.py").write_text("x = 1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "freeze")
    if tag:
        git(repo, "tag", FREEZE_TAG)  # lightweight, so no identity is needed
        if registered and manifest is None:
            register(repo)
    return repo


def token_for(repo):
    return freeze_token(repo / "manifest.json")


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
    (repo / "src").mkdir()
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


def test_freeze_token_is_sha256_of_frozen_manifest_bytes(tmp_path):
    path = tmp_path / "manifest.json"
    raw = b'{\n  "files": {},\n  "registration": null\n}\n'
    path.write_bytes(raw)
    assert freeze_token(path) == hashlib.sha256(raw).hexdigest()


def test_freeze_token_ignores_only_the_registration_field(tmp_path):
    path = tmp_path / "manifest.json"
    frozen = {"files": {"a.py": "1" * 64}, "registration": None}
    write_manifest(path, frozen)
    before = freeze_token(path)
    write_manifest(path, {**frozen, "registration": {"repo_url": REGISTRATION_REPO_URL, "commit": "a" * 40}})
    assert freeze_token(path) == before
    write_manifest(path, {**frozen, "files": {"a.py": "2" * 64}})
    assert freeze_token(path) != before


@pytest.mark.parametrize("raw", [b'{"registration": null}', b'{\n  "registration": null\n}', b"[]\n", b"not json"])
def test_freeze_token_refuses_non_canonical_or_non_object_manifest(tmp_path, raw):
    path = tmp_path / "manifest.json"
    path.write_bytes(raw)
    with pytest.raises(FirewallError):
        freeze_token(path)


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
    for kwargs in ({"repo_dir": "x"}, {"manifest_path": "x"}, {"freeze_tag": "x"}, {"downloader": print}):
        with pytest.raises(TypeError):
            load_test(PINNED[0], "0" * 64, **kwargs)


def test_load_test_ignores_a_throwaway_frozen_repo(tmp_path):
    # The bypass from review: a temp repo with its own manifest and tag cannot satisfy the public function.
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


def test_load_test_rejects_wrong_token(tmp_path):
    repo = make_repo(tmp_path)
    dl = FakeDownloader(tmp_path)
    with pytest.raises(FirewallError, match="token"):
        _load_test(PINNED[0], "0" * 64, repo_dir=repo, downloader=dl)
    assert dl.calls == []


def assert_refused(repo, tmp_path, match, token=None):
    dl = FakeDownloader(tmp_path)
    with pytest.raises(FirewallError, match=match):
        _load_test(PINNED[0], token_for(repo) if token is None else token, repo_dir=repo, downloader=dl)
    assert dl.calls == []


@pytest.mark.parametrize(
    "registration",
    [None, {"osf_id": "abc12"}, {"repo_url": REGISTRATION_REPO_URL}, {"commit": "TAG"}, "https://github.com/aarmens702-hub/jev-benchmark"],
)
def test_load_test_rejects_manifest_without_registration(tmp_path, registration):
    repo = make_repo(tmp_path, registered=False)
    manifest = json.loads((repo / "manifest.json").read_text())
    if registration == {"commit": "TAG"}:
        registration = {"commit": tag_commit(repo)}
    manifest["registration"] = registration
    if registration is not None:  # None is the frozen state itself, HEAD at the tag
        write_manifest(repo / "manifest.json", manifest)
        git(repo, "commit", "-q", "-am", "registration")
    assert_refused(repo, tmp_path, "registration")


def test_load_test_rejects_osf_only_manifest(tmp_path):
    repo = make_repo(tmp_path, registered=False)
    manifest = json.loads((repo / "manifest.json").read_text())
    manifest["osf_id"] = "abc12"
    write_manifest(repo / "manifest.json", manifest)
    git(repo, "commit", "-q", "-am", "osf id only")
    assert_refused(repo, tmp_path, "registration")


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/aarmens702-hub/jev-benchmark/",
        "https://github.com/aarmens702-hub/jev-benchmark.git",
        "http://github.com/aarmens702-hub/jev-benchmark",
        "https://github.com/someone/jev-benchmark",
        "https://github.com/AARMENS702-HUB/jev-benchmark",
        "",
    ],
)
def test_load_test_rejects_a_different_repo_url(tmp_path, url):
    repo = make_repo(tmp_path, registered=False)
    register(repo, url=url)
    assert_refused(repo, tmp_path, "repo_url")


@pytest.mark.parametrize("which", ["earlier commit", "zeros", "short", "upper", "not a string"])
def test_load_test_rejects_a_wrong_commit(tmp_path, which):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "first.txt").write_text("before the freeze\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "earlier")
    earlier = git(repo, "rev-parse", "HEAD").strip()
    write_manifest(repo / "manifest.json", {"files": {}, "registration": None})
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "freeze")
    git(repo, "tag", FREEZE_TAG)
    sha = tag_commit(repo)
    commit = {"earlier commit": earlier, "zeros": "0" * 40, "short": sha[:12], "upper": sha.upper(), "not a string": 123}[which]
    register(repo, commit=commit)
    assert_refused(repo, tmp_path, "commit")


def test_load_test_rejects_registration_commit_that_changes_another_file(tmp_path):
    repo = make_repo(tmp_path, registered=False)
    register(repo, commit_it=False)
    (repo / "a.py").write_text("x = 2\n")
    git(repo, "commit", "-q", "-am", "registration plus a code change")
    assert_refused(repo, tmp_path, "HEAD is not at freeze tag")


def test_load_test_rejects_registration_commit_that_changes_a_frozen_hash(tmp_path):
    repo = make_repo(tmp_path, registered=False)
    manifest = register(repo, commit_it=False)
    manifest["files"]["a.py"] = "f" * 64
    write_manifest(repo / "manifest.json", manifest)
    git(repo, "commit", "-q", "-am", "registration plus a changed hash")
    assert_refused(repo, tmp_path, "more than the manifest's registration")


def test_load_test_rejects_registration_two_commits_after_the_tag(tmp_path):
    repo = make_repo(tmp_path, registered=False)
    (repo / "notes.txt").write_text("a later commit\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "later")
    register(repo)
    assert_refused(repo, tmp_path, "HEAD is not at freeze tag")


def test_load_test_rejects_registration_split_over_two_commits(tmp_path):
    # Both commits touch only the registration, but HEAD's parent is not the tag.
    repo = make_repo(tmp_path, registered=False)
    register(repo, commit="0" * 40)
    register(repo)
    assert_refused(repo, tmp_path, "HEAD is not at freeze tag")


def test_load_test_rejects_uncommitted_registration(tmp_path):
    repo = make_repo(tmp_path, registered=False)
    register(repo, commit_it=False)
    assert_refused(repo, tmp_path, "dirty|clean")


def test_load_test_rejects_frozen_manifest_with_registration_already_set(tmp_path):
    # A registration in the tagged manifest can never name its own commit; refuse any non-null value there.
    repo = make_repo(tmp_path, manifest={"files": {}, "registration": {"repo_url": REGISTRATION_REPO_URL, "commit": "0" * 40}})
    register(repo)
    assert_refused(repo, tmp_path, "registration null")


def test_registered_repo_token_equals_sha256_of_tagged_manifest(tmp_path):
    repo = make_repo(tmp_path)
    tagged = subprocess.run(["git", "show", f"{FREEZE_TAG}:manifest.json"], cwd=repo, capture_output=True, check=True).stdout
    assert token_for(repo) == hashlib.sha256(tagged).hexdigest() == manifest_hash(json.loads(tagged))


def test_load_test_rejects_dirty_tree(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "scratch.txt").write_text("untracked change")
    dl = FakeDownloader(tmp_path)
    with pytest.raises(FirewallError, match="dirty|clean"):
        _load_test(PINNED[0], token_for(repo), repo_dir=repo, downloader=dl)
    assert dl.calls == []


def test_load_test_rejects_head_not_at_freeze_tag(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "later.txt").write_text("a later commit")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "after freeze")
    dl = FakeDownloader(tmp_path)
    with pytest.raises(FirewallError, match="tag"):
        _load_test(PINNED[0], token_for(repo), repo_dir=repo, downloader=dl)
    assert dl.calls == []


def test_load_test_rejects_missing_freeze_tag(tmp_path):
    repo = make_repo(tmp_path, tag=False)
    dl = FakeDownloader(tmp_path)
    with pytest.raises(FirewallError, match="tag"):
        _load_test(PINNED[0], token_for(repo), repo_dir=repo, downloader=dl)
    assert dl.calls == []


def test_load_test_rejects_unknown_dataset(tmp_path):
    repo = make_repo(tmp_path)
    dl = FakeDownloader(tmp_path)
    with pytest.raises(FirewallError, match="dataset"):
        _load_test("someone/else", token_for(repo), repo_dir=repo, downloader=dl)
    assert dl.calls == []


def test_load_test_default_downloader_cannot_run_in_tests(tmp_path):
    # All four conditions hold, no downloader injected: the conftest guard stops the real fetch.
    repo = make_repo(tmp_path)
    with pytest.raises(RuntimeError, match="blocked"):
        _load_test(PINNED[0], token_for(repo), repo_dir=repo)


@pytest.mark.parametrize("dataset", PINNED)
def test_load_test_success_splits_text_from_gold(tmp_path, dataset):
    repo = make_repo(tmp_path)
    dl = FakeDownloader(tmp_path)
    texts, gold = _load_test(dataset, token_for(repo), repo_dir=repo, downloader=dl)
    assert list(texts.columns) == ["row_id", "text"]
    assert list(gold.columns) == ["row_id", "label"]
    assert texts["row_id"].tolist() == gold["row_id"].tolist() == [0, 1, 2]
    assert texts["text"].tolist() == ["alpha", "beta", "gamma"]
    assert gold["label"].tolist() == [2, 0, 1]
    assert [(c[0], c[2]) for c in dl.calls] == [(dataset, firewall.PINNED_TEST_FILES[dataset]["revision"])]


@pytest.mark.parametrize("short,full", [("banking77", PINNED[0]), ("ag_news", PINNED[1])])
def test_load_test_accepts_short_dataset_names(tmp_path, short, full):
    repo = make_repo(tmp_path)
    dl = FakeDownloader(tmp_path)
    _load_test(short, token_for(repo), repo_dir=repo, downloader=dl)
    assert dl.calls[0][0] == full


def test_pinned_revisions_match_design():
    design = (REPO / "DESIGN.md").read_text()
    for spec in firewall.PINNED_TEST_FILES.values():
        assert spec["revision"] in design
