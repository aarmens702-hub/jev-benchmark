"""What must never be committed: run outputs, private notes, data, caches, env files and real-looking API keys."""
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# An OpenRouter key is sk-or-v1- and 64 hex characters. Flag the prefix followed by 32 or more letters or digits.
KEY_LIKE = re.compile(r"sk-or-v1-([A-Za-z0-9]{32,})")
REPEATED = re.compile(r"(.{1,8})\1{3,}")


def git(*args, repo=REPO):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


def commit_candidates(repo=REPO):
    """Tracked files plus untracked files that are not ignored (what `git add -A` would commit)."""
    out = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", repo=repo).stdout
    return sorted(f for f in out.split("\0") if f and (Path(repo) / f).is_file())


def is_fake(match: re.Match) -> bool:
    """A clearly fake fixture: the matched key part itself says FAKE or TEST, or is one short unit repeated."""
    key = match.group(1)
    return "FAKE" in key or "TEST" in key or REPEATED.fullmatch(key) is not None


def key_like_hits(text: str) -> list[str]:
    return [m.group(0)[:14] + "..." for m in KEY_LIKE.finditer(text) if not is_fake(m)]


def real_looking_keys(repo=REPO) -> list[str]:
    hits = []
    for rel in commit_candidates(repo):
        text = (Path(repo) / rel).read_bytes().decode("utf-8", errors="ignore")
        hits += [f"{rel}: {h}" for h in key_like_hits(text)]
    return hits


# --------------------------------------------------------------------------- keys


def test_no_committable_file_contains_a_real_looking_openrouter_key():
    assert real_looking_keys() == []  # only a prefix is shown, never a value


def _fake_hex(n):
    """A random-looking hex run built at test time, so this file holds no key-shaped literal."""
    import hashlib

    return (hashlib.sha256(b"one").hexdigest() + hashlib.sha256(b"two").hexdigest())[:n]


PREFIX = "sk-or" + "-v1-"


@pytest.mark.parametrize(
    "text",
    [
        PREFIX + _fake_hex(64),
        PREFIX + _fake_hex(32),
        'OPENROUTER_API_KEY="' + PREFIX + _fake_hex(64) + '"',
        '{"key": "' + PREFIX + "Zq8" + _fake_hex(40) + '"}',
        "Authorization: Bearer " + PREFIX + _fake_hex(48).upper(),
        PREFIX + _fake_hex(64) + "-FAKE",  # FAKE outside the matched key does not excuse it
        "TEST " + PREFIX + _fake_hex(64),
    ],
)
def test_key_scan_flags_real_looking_keys(text):
    assert len(key_like_hits(text)) == 1


@pytest.mark.parametrize(
    "text",
    [
        PREFIX + _fake_hex(31),  # too short
        PREFIX + "FAKE-TEST-VALUE-0123456789",  # the existing client fixture
        PREFIX + "FAKE" + _fake_hex(60),
        PREFIX + _fake_hex(30) + "TEST" + _fake_hex(30),
        PREFIX + "ab" * 32,
        PREFIX + "0" * 64,
        PREFIX + "deadbeef" * 8,
        "sk-or-v1-",
        "sk-" + _fake_hex(64),  # not an OpenRouter key prefix; other guards cover bare hex in names
    ],
)
def test_key_scan_allows_clearly_fake_fixtures(text):
    assert key_like_hits(text) == []


def test_key_scan_reads_untracked_files_that_would_be_committed(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text(".env\n")
    (tmp_path / ".env").write_text(PREFIX + _fake_hex(64) + "\n")  # ignored, so never committed
    assert real_looking_keys(tmp_path) == []
    (tmp_path / "notes.txt").write_text(PREFIX + _fake_hex(64) + "\n")
    assert real_looking_keys(tmp_path) == ["notes.txt: sk-or-v1-" + _fake_hex(5) + "..."]


# --------------------------------------------------------------------------- .gitignore


@pytest.mark.parametrize(
    "path",
    [
        "runs/private/jev_cache.jsonl",
        "runs/stage1/x.parquet",
        ".superpowers/sdd/PLAN/progress.md",
        "data/banking77/train.parquet",
        ".cache/jev_decisions.jsonl",
        "src/.cache/x.json",
        "src/__pycache__/firewall.cpython-312.pyc",
        ".pytest_cache/v/cache/nodeids",
        ".ruff_cache/x",
        ".venv/bin/python",
        ".env",
        ".env.local",
    ],
)
def test_gitignore_ignores(path):
    assert git("check-ignore", "-q", "--no-index", path).returncode == 0, path


@pytest.mark.parametrize(
    "path",
    ["manifest.json", "registration.md", "deviations.md", "stage0/frozen.yaml", "stage0/stage0_report.md", "results/tables.md", "scored.lock"],
)
def test_gitignore_keeps_what_must_be_committed(path):
    assert git("check-ignore", "-q", "--no-index", path).returncode == 1, path
