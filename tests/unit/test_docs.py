"""Docs describe one install and Triton as the default (docs/history excepted)."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
CURRENT = [ROOT / "README.md"] + [p for p in sorted((ROOT / "docs").rglob("*.md"))
                                   if "history" not in p.relative_to(ROOT).parts
                                   and p.relative_to(ROOT).parts[:2] != ("docs", "chroma")
                                   and "superpowers" not in p.relative_to(ROOT).parts]

STALE = [
    "install Chroma (the chroma-lite distribution) first",
    "pip install -e trichroma",
    "git clone https://github.com/youngsm/trichroma.git",
    "when PyCUDA is installed",
    "Chroma itself (geometry, materials, detectors, events) still comes from chroma-lite",
    "from trichroma",
    "import trichroma",
    "python -m trichroma",
]


def test_docs_have_no_two_package_install():
    found = []
    for path in CURRENT:
        text = path.read_text()
        for phrase in STALE:
            if phrase.lower() in text.lower():
                found.append("%s: %r" % (path.relative_to(ROOT), phrase))
    assert found == []


def test_readme_links_resolve():
    text = (ROOT / "README.md").read_text()
    for target in re.findall(r"\]\(([^)#]+)\)", text):
        if not target.startswith("http"):
            assert (ROOT / target).exists(), target
