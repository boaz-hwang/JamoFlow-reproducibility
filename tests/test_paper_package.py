from __future__ import annotations

import json
import re
import shutil
import struct
from pathlib import Path

import pytest

from scripts import build_arr_paper
from scripts import generate_paper_figures


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/arr-submission.md"
REFERENCES = ROOT / "paper/references.bib"
TEMPLATE = ROOT / "paper/acl-template.tex"
METADATA = ROOT / "paper/arr-submission-metadata.json"
CHECKLIST = ROOT / "paper/arr-responsible-checklist-draft.md"


def test_arr_source_is_anonymous_and_has_required_sections() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    template = TEMPLATE.read_text(encoding="utf-8")
    assert "author:" not in source.split("---", 2)[1]
    assert "Anonymous ACL submission" in template
    assert "http://" not in source
    assert "https://" not in source
    conclusion = source.index("# Conclusion")
    limitations = source.index("# Limitations")
    ethics = source.index("# Ethical Considerations")
    assert conclusion < limitations < ethics


def test_abstract_and_claim_scope_are_submission_safe() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert build_arr_paper.extract_abstract_words() == 181
    assert build_arr_paper.extract_abstract_words() <= 200
    assert "does not amplify" in source
    assert "not a positive 10% inference technique" in source
    assert "not public preregistration or cryptographic one-shot execution" in source


def test_all_markdown_citations_exist_in_bibliography() -> None:
    citation_keys = set(re.findall(r"@([A-Za-z0-9_.:+-]+)", SOURCE.read_text(encoding="utf-8")))
    bibliography_keys = set(
        re.findall(
            r"@[A-Za-z]+\s*\{\s*([^,\s]+)",
            REFERENCES.read_text(encoding="utf-8"),
        )
    )
    assert citation_keys
    assert citation_keys <= bibliography_keys


def test_openreview_metadata_matches_paper_and_records_approved_release_choices() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    front_matter = source.split("---", 2)[1]
    title = re.search(r'^title: "(.+)"$', front_matter, re.MULTILINE)
    assert title is not None
    assert metadata["title"] == title.group(1)

    abstract_lines = []
    in_abstract = False
    for line in front_matter.splitlines():
        if line == "abstract: |":
            in_abstract = True
            continue
        if in_abstract:
            if not line.startswith("  "):
                break
            abstract_lines.append(line[2:])
    assert metadata["abstract"] == " ".join(abstract_lines)
    assert metadata["paper_type"] == "long"
    assert metadata["preferred_venue"] == "NAACL 2027"
    assert metadata["preprint_choice"] == "named_preprint"
    assert metadata["software_archive"] == {
        "attach": False,
        "status": "named_public_archive_released_separately_under_Apache-2.0",
        "maximum_bytes": 200_000_000,
    }
    assert metadata["data_archive"]["attach"] is False
    assert metadata["consent_to_share_anonymized_metadata"] is False
    assert metadata["consent_to_review"] is True
    assert metadata["external_author_inputs_required"] == []


def test_responsible_checklist_discloses_unresolved_items_and_ai_use() -> None:
    checklist = CHECKLIST.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    assert "B2 licenses or terms | **No, not fully in the anonymous paper**" in checklist
    assert "B4 PII/offensive-content checks | **No**" in checklist
    assert "C1 parameters, total compute, infrastructure | **No**" in checklist
    assert "C4 packages and implementations | **No in the anonymous package**" in checklist
    assert "E1 AI-assistant use | **Yes**" in checklist
    assert "AI assistants were used" in source
    assert "complete project-wide accelerator-hour total" in source
    assert "did not perform a content-level PII" in source


def test_named_preprint_has_public_reproducibility_link() -> None:
    template = (ROOT / "paper/acl-preprint-template.tex").read_text(encoding="utf-8")
    assert "\\usepackage[review]{acl}" not in template
    assert "Code and Reproducibility" in template
    assert "https://github.com/boaz-hwang/JamoFlow-reproducibility" in template


@pytest.mark.parametrize(
    "name",
    ["scale-headroom-versus-trained.png", "trained-scale-evidence.png"],
)
def test_tracked_pngs_are_opaque_eight_bit_rgb(name: str) -> None:
    value = (ROOT / "paper/figures" / name).read_bytes()
    assert value[:8] == b"\x89PNG\r\n\x1a\n"
    width, height, bit_depth, color_type, _, _, _ = struct.unpack(
        ">IIBBBBB", value[16:29]
    )
    assert width > 0 and height > 0
    assert bit_depth == 8
    assert color_type == 2  # truecolor RGB, with no alpha channel


def test_acl_style_and_source_date_are_pinned() -> None:
    assert build_arr_paper.ACL_STYLE_COMMIT == "d5adc823ff0f80f98c80405ca0ab66c68e684409"
    assert build_arr_paper.SOURCE_DATE_EPOCH == "1786924800"
    assert build_arr_paper.ACL_FILES == {
        "acl.sty": "19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de",
        "acl_natbib.bst": "6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e",
    }


@pytest.mark.skipif(shutil.which("magick") is None, reason="ImageMagick is unavailable")
def test_tracked_figures_match_aggregate_evidence() -> None:
    generate_paper_figures.verify()
