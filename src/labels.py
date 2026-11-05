"""Label text for the two datasets, from the committed third_party files.

`bare=True` gives the official class name as the text. `bare=False` gives the reused description.
Keys are the official class names, in the order of the committed file (match by name, not position).
"""
import json
import re
from pathlib import Path

THIRD_PARTY = Path(__file__).resolve().parents[1] / "third_party"
BANKING77_FILE = THIRD_PARTY / "banking77" / "banking77_intents.v1.json"
AG_NEWS_FILE = THIRD_PARTY / "ag_news" / "agnews.yaml"


def _banking77() -> tuple[dict[str, str], str]:
    blob = json.loads(BANKING77_FILE.read_text(encoding="utf-8"))
    return dict(blob["descriptions"]), blob["instructions"]


_AG_LABEL_LINE = re.compile(r"^\s*-\s*\{\s*name:\s*(?P<name>[^,]+?)\s*,\s*desc:\s*(?P<desc>.*?)\s*\}\s*$")
_AG_QUESTION_LINE = re.compile(r"^question:\s*(?P<q>.*?)\s*$")


def _ag_news() -> tuple[dict[str, str], str]:
    """Read agnews.yaml as literal text. Parsed as YAML, each flow-mapping line would be cut at the
    first comma inside its description ("world news" instead of the whole sentence)."""
    descriptions: dict[str, str] = {}
    question = None
    for line in AG_NEWS_FILE.read_text(encoding="utf-8").splitlines():
        if m := _AG_LABEL_LINE.match(line):
            descriptions[m["name"]] = m["desc"]
        elif m := _AG_QUESTION_LINE.match(line):
            question = m["q"]
        elif line.strip().startswith("- ") and "name:" in line:
            raise ValueError(f"unreadable label line in {AG_NEWS_FILE.name}: {line!r}")
    if question is None or not descriptions:
        raise ValueError(f"{AG_NEWS_FILE.name}: no question or labels found")
    return descriptions, question


def _load(dataset: str) -> tuple[dict[str, str], str]:
    if dataset == "banking77":
        return _banking77()
    if dataset == "ag_news":
        return _ag_news()
    raise ValueError(f"unknown dataset {dataset!r}; expected 'banking77' or 'ag_news'")


def label_texts(dataset: str, bare: bool) -> dict[str, str]:
    """{official class name: text}. bare=True maps each name to itself; else to its description."""
    descriptions, _ = _load(dataset)
    return {name: name for name in descriptions} if bare else descriptions


def question(dataset: str) -> str:
    """The task question that ships with the label file."""
    return _load(dataset)[1]
