"""Freeze manifest (VALIDATION.md, Manifest) and registration document. Offline; temp repos and a fake HF cache."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

import data
import firewall
import freeze
import routing
from firewall import FREEZE_TAG, REGISTRATION_REPO_URL, canonical_bytes
from freeze import FROZEN_DIRS, FROZEN_FILES, PENDING, FreezeError

REPO = Path(__file__).resolve().parents[1]


def git(repo, *args):
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args]
    return subprocess.run(cmd, cwd=repo, check=True, capture_output=True, text=True).stdout


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# One file per frozen category, plus one nested file, so every category is exercised.
DIR_FILES = {
    "src/code.py": "x = 1\n",
    "src/sub/more.py": "y = 2\n",
    "scripts/run.sh": "#!/bin/bash\necho run\n",
    "routing/rules.py": "RULE = 'a'\n",
    "prompts/p.v1.json": '{"instructions": "pick one"}\n',
    "third_party/NOTICE.md": "MIT\n",
    "tests/test_x.py": "def test_x():\n    pass\n",
}


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    for rel, text in DIR_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    for rel in FROZEN_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if rel.startswith("stage0/"):
            shutil.copy(REPO / rel, root / rel)  # the real Stage 0 outputs: models.small_llm is read from frozen.yaml
        else:
            (root / rel).write_text(f"content of {rel}\n")
    (root / ".gitignore").write_text("runs/\n__pycache__/\n")
    (root / "README.md").write_text("not frozen\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    return root


@pytest.fixture
def hub(tmp_path):
    """A fake HF cache holding both pinned train parquets at their pinned revisions."""
    root = tmp_path / "hub"
    for name, (repo_id, revision) in data.PINNED.items():
        path = root / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / revision / data.TRAIN_FILE
        path.parent.mkdir(parents=True)
        path.write_bytes(f"parquet bytes for {name}".encode())
    return root


def parquet_path(hub, name):
    repo_id, revision = data.PINNED[name]
    return hub / f"datasets--{repo_id.replace('/', '--')}" / "snapshots" / revision / data.TRAIN_FILE


def frozen_repo(repo, hub):
    """Build the manifest, commit it, and return it."""
    manifest = freeze.build_manifest(repo, hf_cache_dir=hub)
    git(repo, "add", "manifest.json")
    git(repo, "commit", "-q", "-m", "manifest")
    return manifest


def flip_last_byte(path):
    raw = bytearray(Path(path).read_bytes())
    raw[-1] ^= 0x01
    Path(path).write_bytes(bytes(raw))


# --------------------------------------------------------------------------- every category hashed


def test_manifest_hashes_code_lock_labels_prompts_rules_seeds_thresholds_and_train_parquet(repo, hub):
    manifest = freeze.build_manifest(repo, hf_cache_dir=hub)
    expected = sorted([*DIR_FILES, *FROZEN_FILES])
    assert sorted(manifest["files"]) == expected
    for rel, digest in manifest["files"].items():
        assert digest == sha(repo / rel), rel
    for category in FROZEN_DIRS:
        assert any(rel.startswith(category + "/") for rel in manifest["files"]), category
    for rel in ("uv.lock", "pyproject.toml", "DESIGN.md", "VALIDATION.md", "PLAN.md", "deviations.md",
                "stage0/frozen.yaml", "stage0/stage0_report.md", "registration.md"):
        assert rel in manifest["files"]
    assert "README.md" not in manifest["files"] and ".gitignore" in manifest["files"]
    assert manifest["repo_url"] == REGISTRATION_REPO_URL

    for name, (repo_id, revision) in data.PINNED.items():
        entry = manifest["train_parquet"][name]
        assert entry == {"repo_id": repo_id, "revision": revision, "filename": data.TRAIN_FILE,
                         "sha256": sha(parquet_path(hub, name))}

    assert manifest["routing_rule_fingerprint"] == routing.rule_fingerprint()
    assert manifest["seeds"]["split"] == 20261105
    assert manifest["seeds"]["option_order"] == 1105
    assert manifest["seeds"]["bootstrap"] == 2026
    assert manifest["seeds"]["routing_sample_prefix"] == "20261105:routing:"
    t = manifest["thresholds"]
    assert (t["coverage"], t["bootstrap_replicates"], t["margin_points"]) == (0.8, 10_000, 2)
    assert t["grid"] == [1, 2, 3, 5, 8, 16, 35, "all"]
    assert t["lr_sequence"] == ["all", 35, 16, 8, 5, 3, 2, 1] and t["jev_sequence"] == [1, 2, 3, 5, 8, 16, 35, "all"]
    assert t["failure_flag_at"] == {"banking77": 31, "ag_news": 76}
    assert (t["nll_floor"], t["ece_bins"], t["lr_c"], t["embed_temperature"]) == (0.005, 15, 100, 0.05)
    assert (t["routing_margin_points"], t["routing_n_sample"], t["hard_cap_usd"]) == (5, 800, 7.15)
    models = manifest["models"]
    assert models["jev"] == {"model": "typesafe/jev-1.14", "provider": "TypeSafe", "build": "jev-1.14-20261014"}
    assert models["small_llm"]["model"] == "mistralai/mistral-nemo"
    assert models["small_llm"]["provider"] == "DekaLLM"
    assert models["small_llm"]["provider_options"]["allow_fallbacks"] is False
    assert models["nli"]["large"]["revision"] == "b2730f16019076bb0009481121efbe4705e0e378"
    assert "registration" not in manifest

    raw = (repo / "manifest.json").read_bytes()
    assert raw == canonical_bytes(manifest)
    git(repo, "add", "manifest.json")
    git(repo, "commit", "-q", "-m", "manifest")
    git(repo, "tag", FREEZE_TAG)
    assert firewall.freeze_token(repo) == hashlib.sha256(raw).hexdigest()


def test_manifest_excluded_from_its_own_hashes(repo, hub):
    first = frozen_repo(repo, hub)  # manifest.json is now tracked at the repo root
    assert "manifest.json" not in first["files"]
    again = freeze.compute_manifest(repo, hf_cache_dir=hub)
    assert "manifest.json" not in again["files"]
    assert again == first
    assert freeze.verify_manifest(repo, hf_cache_dir=hub) == []


def test_unchanged_tree_verifies(repo, hub):
    frozen_repo(repo, hub)
    assert freeze.verify_manifest(repo, hf_cache_dir=hub) == []


def test_real_repo_frozen_paths_cover_every_category():
    paths = freeze.frozen_paths(REPO)
    for category in FROZEN_DIRS:
        assert any(p.startswith(category + "/") for p in paths), category
    assert set(FROZEN_FILES) <= set(paths)
    assert {"src/freeze.py", "src/firewall.py", "routing/rules.py", "scripts/freeze.sh"} <= set(paths)
    assert "manifest.json" not in paths
    tracked = git(REPO, "ls-files", "-z").split("\0")
    assert "frozen.yaml" not in tracked and "stage0_report.md" not in tracked  # moved into stage0/


def test_real_repo_manifest_computes_with_pending_train_hashes(tmp_path):
    manifest = freeze.compute_manifest(REPO, hf_cache_dir=tmp_path / "empty-hub")
    assert freeze.pending_hashes(manifest) == ["ag_news", "banking77"]
    assert manifest["models"]["small_llm"]["model"] == "mistralai/mistral-nemo"
    assert manifest["repo_url"] == REGISTRATION_REPO_URL


# --------------------------------------------------------------------------- any change is detected


@pytest.mark.parametrize("rel", sorted([*DIR_FILES, *FROZEN_FILES]))
def test_manifest_detects_any_change(repo, hub, rel):
    frozen_repo(repo, hub)
    flip_last_byte(repo / rel)
    problems = freeze.verify_manifest(repo, hf_cache_dir=hub)
    assert any(p.startswith(f"files.{rel}: changed") for p in problems), problems


@pytest.mark.parametrize("name", sorted(data.PINNED))
def test_one_byte_change_in_a_train_parquet_is_detected(repo, hub, name):
    frozen_repo(repo, hub)
    flip_last_byte(parquet_path(hub, name))
    problems = freeze.verify_manifest(repo, hf_cache_dir=hub)
    assert any(p.startswith(f"train_parquet.{name}.sha256: changed") for p in problems), problems


def test_added_removed_and_untracked_files_are_detected(repo, hub):
    frozen_repo(repo, hub)
    (repo / "src" / "new.py").write_text("z = 3\n")
    problems = freeze.verify_manifest(repo, hf_cache_dir=hub)
    assert any("dirty" in p and "src/new.py" in p for p in problems)
    git(repo, "add", "src/new.py")
    problems = freeze.verify_manifest(repo, hf_cache_dir=hub)
    assert any(p.startswith("files.src/new.py: present now but not in the manifest") for p in problems)
    git(repo, "rm", "-q", "src/code.py")
    problems = freeze.verify_manifest(repo, hf_cache_dir=hub)
    assert any(p.startswith("files.src/code.py: in the manifest but no longer present") for p in problems)


def test_ignored_files_in_frozen_folders_are_not_flagged(repo, hub):
    frozen_repo(repo, hub)
    (repo / "src" / "__pycache__").mkdir()
    (repo / "src" / "__pycache__" / "code.cpython-312.pyc").write_bytes(b"\0")
    assert freeze.verify_manifest(repo, hf_cache_dir=hub) == []


def test_settings_and_rule_fingerprint_changes_are_detected(repo, hub, monkeypatch):
    frozen_repo(repo, hub)
    monkeypatch.setattr(data, "SEED", 20261106)
    assert any(p.startswith("seeds.split: changed") for p in freeze.verify_manifest(repo, hf_cache_dir=hub))
    monkeypatch.undo()
    import verdict

    monkeypatch.setattr(verdict, "MARGIN", 1)
    assert any(p.startswith("thresholds.margin_points: changed") for p in freeze.verify_manifest(repo, hf_cache_dir=hub))
    monkeypatch.undo()
    monkeypatch.setattr(routing, "rule_fingerprint", lambda *a, **k: "0" * 64)
    assert any(p.startswith("routing_rule_fingerprint: changed") for p in freeze.verify_manifest(repo, hf_cache_dir=hub))


def test_edited_manifest_value_is_detected(repo, hub):
    manifest = frozen_repo(repo, hub)
    manifest["thresholds"]["coverage"] = 0.7
    (repo / "manifest.json").write_bytes(canonical_bytes(manifest))
    assert any(p.startswith("thresholds.coverage: changed") for p in freeze.verify_manifest(repo, hf_cache_dir=hub))


def test_non_canonical_manifest_is_refused(repo, hub):
    frozen_repo(repo, hub)
    raw = (repo / "manifest.json").read_bytes()
    (repo / "manifest.json").write_bytes(raw.rstrip(b"\n") + b" \n")
    assert any("canonical" in p for p in freeze.verify_manifest(repo, hf_cache_dir=hub))


# --------------------------------------------------------------------------- pending train hashes


def test_train_parquet_not_cached_is_recorded_as_pending(repo, tmp_path):
    manifest = freeze.build_manifest(repo, hf_cache_dir=tmp_path / "empty-hub")
    for name, (repo_id, revision) in data.PINNED.items():
        assert manifest["train_parquet"][name] == {"repo_id": repo_id, "revision": revision,
                                                   "filename": data.TRAIN_FILE, "sha256": PENDING}
    git(repo, "add", "manifest.json")
    git(repo, "commit", "-q", "-m", "manifest")
    problems = freeze.verify_manifest(repo, hf_cache_dir=tmp_path / "empty-hub")
    assert sorted(p for p in problems if "pending" in p) == [
        "train_parquet.ag_news: hash is pending; cache the file and rebuild before the tag",
        "train_parquet.banking77: hash is pending; cache the file and rebuild before the tag",
    ]


def test_recorded_parquet_missing_from_cache_cannot_verify(repo, hub, tmp_path):
    frozen_repo(repo, hub)
    problems = freeze.verify_manifest(repo, hf_cache_dir=tmp_path / "empty-hub")
    assert any("not in the Hugging Face cache" in p for p in problems)


def test_missing_frozen_file_refuses_build(repo, hub):
    git(repo, "rm", "-q", "deviations.md")
    git(repo, "commit", "-q", "-m", "drop deviations")
    with pytest.raises(FreezeError, match="deviations.md"):
        freeze.build_manifest(repo, hf_cache_dir=hub)


# --------------------------------------------------------------------------- build refusals and the firewall


def test_build_refuses_once_the_tag_exists(repo, hub):
    frozen_repo(repo, hub)
    git(repo, "tag", FREEZE_TAG)
    with pytest.raises(FreezeError, match="exists"):
        freeze.build_manifest(repo, hf_cache_dir=hub)


@pytest.mark.parametrize("how", ["modified", "untracked", "staged", "skip-worktree"])
def test_build_refuses_a_dirty_tree(repo, hub, how):
    if how == "modified":
        (repo / "src" / "code.py").write_text("x = 2\n")
    elif how == "untracked":
        (repo / "notes.txt").write_text("x\n")
    elif how == "staged":
        (repo / "src" / "code.py").write_text("x = 2\n")
        git(repo, "add", "src/code.py")
    else:
        git(repo, "update-index", "--skip-worktree", "src/code.py")
    with pytest.raises(FreezeError, match="not clean"):
        freeze.build_manifest(repo, hf_cache_dir=hub)
    assert not (repo / "manifest.json").exists()


def test_build_allows_rebuilding_its_own_manifest(repo, hub):
    frozen_repo(repo, hub)
    (repo / "manifest.json").write_text("stale\n")
    freeze.build_manifest(repo, hf_cache_dir=hub)
    assert freeze.verify_manifest(repo, hf_cache_dir=hub) == []


def test_type_changes_in_recorded_values_are_detected(repo, hub):
    manifest = frozen_repo(repo, hub)
    manifest["thresholds"]["draws"] = 10.0
    manifest["seeds"]["lr"] = False  # was 0
    (repo / "manifest.json").write_bytes(canonical_bytes(manifest))
    problems = freeze.verify_manifest(repo, hf_cache_dir=hub)
    assert any(p.startswith("thresholds.draws: changed") for p in problems)
    assert any(p.startswith("seeds.lr: changed") for p in problems)


def _firewall_load(repo, hub, tmp_path):
    head = git(repo, "rev-parse", "HEAD").strip()

    def lister(url, refs):
        return f"{head}\trefs/heads/main\n{head}\trefs/tags/{FREEZE_TAG}\n"

    def download(repo_id, filename, revision):
        path = tmp_path / "fake.parquet"
        pd.DataFrame({"text": ["a"], "label": [0]}).to_parquet(path)
        return str(path)

    return firewall._load_test(
        "banking77", firewall.freeze_token(repo), repo_dir=repo, downloader=download,
        verifier=lambda r: freeze.verify_manifest(r, hf_cache_dir=hub), lister=lister, frozen_dirs=FROZEN_DIRS,
    )


def test_frozen_repo_opens_the_firewall_end_to_end(repo, hub, tmp_path):
    frozen_repo(repo, hub)
    git(repo, "tag", "-a", FREEZE_TAG, "-m", "freeze")
    texts, gold = _firewall_load(repo, hub, tmp_path)
    assert texts["text"].tolist() == ["a"] and gold["label"].tolist() == [0]


def test_firewall_runs_verify_at_load_including_train_parquets(repo, hub, tmp_path):
    frozen_repo(repo, hub)
    git(repo, "tag", FREEZE_TAG)
    flip_last_byte(parquet_path(hub, "banking77"))
    with pytest.raises(firewall.FirewallError, match="does not verify"):
        _firewall_load(repo, hub, tmp_path)


def test_firewall_refuses_a_tag_on_a_manifest_with_pending_hashes(repo, tmp_path):
    freeze.build_manifest(repo, hf_cache_dir=tmp_path / "empty-hub")
    git(repo, "add", "manifest.json")
    git(repo, "commit", "-q", "-m", "manifest")
    git(repo, "tag", FREEZE_TAG)
    with pytest.raises(firewall.FirewallError, match="pending"):
        _firewall_load(repo, tmp_path / "empty-hub", tmp_path)


def test_cli_builds_and_verifies(repo, hub, monkeypatch, capsys):
    monkeypatch.setattr(freeze, "REPO", repo)
    real = freeze.train_parquet_hashes
    monkeypatch.setattr(freeze, "train_parquet_hashes", lambda hf_cache_dir=None: real(hub))
    assert freeze.main([]) == 0
    assert "files hashed" in capsys.readouterr().out
    assert freeze.main(["--verify"]) == 0
    assert "manifest verifies" in capsys.readouterr().out
    flip_last_byte(repo / "src" / "code.py")
    assert freeze.main(["--verify"]) == 1
    assert "files.src/code.py: changed" in capsys.readouterr().out
    flip_last_byte(repo / "src" / "code.py")
    git(repo, "tag", FREEZE_TAG)
    assert freeze.main([]) == 2  # the frozen manifest is never rebuilt


# --------------------------------------------------------------------------- documents


def test_deviations_table_rows_have_every_column():
    lines = [ln for ln in (REPO / "deviations.md").read_text().splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    assert header == ["Date", "What changed", "Reason", "Stage", "Approved by"]
    rows = lines[2:]
    assert len(rows) >= 4
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert len(cells) == len(header) and all(cells), row
        assert cells[header.index("Stage")].startswith("pre-freeze")


# Rule sentences that registration.md must quote exactly as DESIGN.md states them.
QUOTED_RULES = [
    "At equal coverage, how many labels per class does a light classifier need to beat Jev, when Jev gets no labels and when Jev gets the same labels?",
    "acc@80 = correct/0.8n.",
    "Test s = all, 35, 16, 8, 5, 3, 2, 1 in that order. s passes if L_s > 0 and D_s >= 2. Stop at the first failure.",
    "Test s = 1, 2, 3, 5, 8, 16, 35, all in that order. s passes if U_s < 0 and D_s <= -2. Stop at the first failure.",
    "| 1, 2, 3 | any | A light classifier with s* labels per class beats zero-label Jev at 80% coverage on this 77-way task. |",
    "| 5, 8, 16 | 1 or more | Jev leads through s_J labels per class and LR leads from s*. Jev's value on this task is limited to cold starts. |",
    "| 5, 8, 16 | none | LR leads clearly from s*. Below that, no clear difference. |",
    "| 35, all, none | 3 or more | Jev leads when labels are scarce, through s_J labels per class. |",
    '| 35, all, none | none, 1, 2 | No clear winner when labels are scarce. If [L_s, U_s] is inside [-2, +2] at every s, say "equivalent within 2 points". |',
    "**Prediction on record.** True if and only if s* is 8 or lower on Banking77.",
    '"Jev follows a changed rule from text with no labels" only if Jev minus Intent LR + code has lower limit above -5 under rule B',
    '"Not unique to Jev" if small LLM or NLI minus Jev has lower limit above -5.',
    "Registration prevents later changes; it does not make the design blind.",
    "Jev's training data is not disclosed, so memorisation of public datasets cannot be ruled out.",
    "Hard cap CA$10 (about US$7.15 at 1.3994), everything included, enforced by a limit on the key",
    'His claim, verbatim: "zeroshot NLI, embedding based classifiers and constrained output models already support dynamic classification without task specific retraining."',
]


def test_registration_quotes_design_rules_verbatim():
    design = (REPO / "DESIGN.md").read_text()
    registration = (REPO / "registration.md").read_text()
    for sentence in QUOTED_RULES:
        assert sentence in design, sentence
        assert sentence in registration, sentence
    disclosure = next(p for p in design.split("\n\n") if p.startswith("**Disclosure, in the registration and the post:**"))
    comment = next(p for p in design.split("\n\n") if p.startswith("Aarmen's comment, verbatim:"))
    assert comment in registration
    text = disclosure.split(":** ", 1)[1]
    assert text.startswith("the analyst") and "The" + text[3:] in registration  # capitalised as a paragraph start


def test_registration_names_the_selected_small_llm_and_public_repo():
    registration = (REPO / "registration.md").read_text()
    assert "mistralai/mistral-nemo" in registration and "DekaLLM" in registration
    assert REGISTRATION_REPO_URL in registration and FREEZE_TAG in registration
    assert "recorded as values, not hashed" in registration
    assert "registration commit" not in registration
    assert "—" not in registration and "–" not in registration  # no em or en dashes


def test_firewall_catches_code_hidden_by_a_clean_filter(repo, hub, tmp_path):
    # git is fooled by the filter; verify hashes the bytes on disk, so the firewall still refuses.
    frozen_repo(repo, hub)
    git(repo, "tag", FREEZE_TAG)
    original = tmp_path / "orig.py"
    original.write_bytes((repo / "src" / "code.py").read_bytes())
    git(repo, "config", "filter.hide.clean", f"cat {original}")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("src/code.py filter=hide\n")
    (repo / "src" / "code.py").write_text("x = 'changed after the freeze'\n")
    git(repo, "add", "src/code.py")
    assert git(repo, "status", "--porcelain") == ""
    with pytest.raises(firewall.FirewallError, match="differ from the committed bytes"):
        _firewall_load(repo, hub, tmp_path)
    assert any("src/code.py: changed" in p for p in freeze.verify_manifest(repo, hf_cache_dir=hub))  # verify catches it too


def _filter_hides(repo, tmp_path, rel):
    """A clean filter that makes git store the current bytes of `rel` whatever is on disk."""
    original = tmp_path / ("orig-" + rel.replace("/", "_"))
    original.write_bytes((repo / rel).read_bytes())
    git(repo, "config", "filter.hide.clean", f"cat {original}")
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text(f"{rel} filter=hide\n")


def _freeze_without_build_checks(repo, hub):
    """What a build that trusted git status would have produced: hashes of the bytes on disk, committed, tagged."""
    (repo / "manifest.json").write_bytes(canonical_bytes(freeze.compute_manifest(repo, hf_cache_dir=hub)))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "manifest")
    git(repo, "tag", FREEZE_TAG)


def test_clean_filter_set_before_the_build_is_refused(repo, hub, tmp_path):
    _filter_hides(repo, tmp_path, "src/code.py")
    (repo / "src" / "code.py").write_text("x = 'what runs, not what is published'\n")
    git(repo, "add", "src/code.py")
    assert git(repo, "status", "--porcelain") == ""
    with pytest.raises(FreezeError, match="differ from the committed bytes"):
        freeze.build_manifest(repo, hf_cache_dir=hub)
    _freeze_without_build_checks(repo, hub)
    assert git(repo, "show", "HEAD:src/code.py") != (repo / "src" / "code.py").read_text()  # public differs from disk
    with pytest.raises(firewall.FirewallError, match="differ from the committed bytes"):
        _firewall_load(repo, hub, tmp_path)


def test_eol_conversion_set_before_the_build_is_refused(repo, hub, tmp_path):
    (repo / ".git" / "info").mkdir(exist_ok=True)
    (repo / ".git" / "info" / "attributes").write_text("src/code.py text eol=crlf\n")
    (repo / "src" / "code.py").write_bytes(b"x = 1\r\n")
    git(repo, "add", "src/code.py")
    assert git(repo, "status", "--porcelain") == ""
    with pytest.raises(FreezeError, match="differ from the committed bytes"):
        freeze.build_manifest(repo, hf_cache_dir=hub)
    _freeze_without_build_checks(repo, hub)
    with pytest.raises(firewall.FirewallError, match="differ from the committed bytes"):
        _firewall_load(repo, hub, tmp_path)


def test_stage1_reads_the_tracked_frozen_yaml():
    import stage0
    import stage1

    assert stage1.FROZEN_YAML == stage0.FROZEN_YAML
