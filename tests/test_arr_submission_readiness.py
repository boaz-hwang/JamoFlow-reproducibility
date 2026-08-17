from __future__ import annotations

import io
import json
import stat
import zipfile

import pytest

from scripts import audit_arr_submission_readiness as readiness
from scripts import build_arxiv_preprint


def _decisions() -> dict[str, object]:
    return {
        "schema_version": 1,
        "authors": [
            {
                "name": "Ada Researcher",
                "email": "ada@research.org",
                "openreview_profile": "~Ada_Researcher1",
                "orcid": "0000-0002-1825-0097",
                "affiliation_ids": ["lab"],
                "contribution": "Designed the study and verified the evidence.",
                "authorship_confirmed": True,
                "reviewer_registration_ready": True,
            }
        ],
        "affiliations": [
            {"id": "lab", "name": "Research Lab", "address": "Seoul, Republic of Korea"}
        ],
        "acknowledgments": None,
        "funding": [],
        "conflicts_of_interest": [],
        "author_order_confirmed": True,
        "paper_approved_by_all_authors": True,
        "checklist_approved_by_all_authors": True,
        "approved_anonymous_pdf_sha256": "0" * 64,
        "approved_checklist_sha256": "1" * 64,
        "approved_public_metadata_sha256": "2" * 64,
        "preferred_venue": "NAACL 2027",
        "preprint_policy": "arr_anonymous_only_until_meta_review",
        "existing_preprints": [],
        "software_archive_choice": "none_for_anonymous_review",
        "public_code_release_choice": "after_review",
        "code_license": None,
        "consent_to_share_anonymized_metadata": False,
        "consent_to_review": True,
    }


def test_submission_decisions_validate_and_project_to_preprint_metadata() -> None:
    value = readiness.validate_decisions(_decisions(), tracked_license_present=False)
    metadata = value["author_metadata"]
    assert metadata["authors"][0]["name"] == "Ada Researcher"
    named = _decisions()
    named["preprint_policy"] = "named_preprint"
    projected = build_arxiv_preprint.author_metadata_from_submission_decisions(named)
    assert projected == metadata


def test_preprint_builder_requires_explicit_named_preprint_choice() -> None:
    with pytest.raises(ValueError, match="do not authorize"):
        build_arxiv_preprint.author_metadata_from_submission_decisions(_decisions())


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("author_order_confirmed", False, "not approved"),
        ("consent_to_review", False, "not approved"),
        ("preferred_venue", "", "preferred venue"),
        ("preprint_policy", "invalid", "preprint policy"),
        ("software_archive_choice", "deidentified_anonymous_archive", "no reviewed"),
    ],
)
def test_submission_decisions_fail_closed(field: str, value: object, match: str) -> None:
    decisions = _decisions()
    decisions[field] = value
    with pytest.raises(ValueError, match=match):
        readiness.validate_decisions(decisions, tracked_license_present=False)


def test_anonymous_preprint_policy_rejects_existing_named_preprint() -> None:
    decisions = _decisions()
    decisions["existing_preprints"] = ["https://arxiv.org/abs/1234.56789"]
    with pytest.raises(ValueError, match="conflicts"):
        readiness.validate_decisions(decisions, tracked_license_present=False)


def test_named_code_release_requires_tracked_license() -> None:
    decisions = _decisions()
    decisions["public_code_release_choice"] = "with_named_preprint"
    decisions["code_license"] = "Apache-2.0"
    with pytest.raises(ValueError, match="tracked license"):
        readiness.validate_decisions(decisions, tracked_license_present=False)


def test_author_approval_is_bound_to_exact_artifact_hashes() -> None:
    expected = {
        "approved_anonymous_pdf_sha256": "0" * 64,
        "approved_checklist_sha256": "1" * 64,
        "approved_public_metadata_sha256": "f" * 64,
    }
    with pytest.raises(ValueError, match="approved artifact has changed"):
        readiness.validate_decisions(
            _decisions(),
            tracked_license_present=False,
            expected_artifact_hashes=expected,
        )


def test_private_template_binds_artifacts_and_is_deliberately_invalid() -> None:
    expected = {
        "approved_anonymous_pdf_sha256": "0" * 64,
        "approved_checklist_sha256": "1" * 64,
        "approved_public_metadata_sha256": "2" * 64,
    }
    value = readiness.private_decision_template(artifact_hashes=expected)
    assert {field: value[field] for field in expected} == expected
    assert value["author_order_confirmed"] is False
    assert value["consent_to_share_anonymized_metadata"] is None
    with pytest.raises(ValueError, match="placeholder"):
        readiness.validate_decisions(value, tracked_license_present=False)


def test_private_template_rejects_bad_hashes() -> None:
    with pytest.raises(ValueError, match="keys differ"):
        readiness.private_decision_template(artifact_hashes={})
    with pytest.raises(ValueError, match="hash is invalid"):
        readiness.private_decision_template(
            artifact_hashes={
                "approved_anonymous_pdf_sha256": "0" * 63,
                "approved_checklist_sha256": "1" * 64,
                "approved_public_metadata_sha256": "2" * 64,
            }
        )


@pytest.mark.parametrize(
    "field,mutate",
    [
        ("contribution", lambda value: value["authors"][0].update(contribution="TODO")),
        ("preferred venue", lambda value: value.update(preferred_venue="TBD")),
        ("funding item", lambda value: value.update(funding=["placeholder"])),
        ("code license", lambda value: value.update(code_license="replace")),
    ],
)
def test_submission_decisions_reject_placeholder_decisions(field, mutate) -> None:
    decisions = _decisions()
    mutate(decisions)
    with pytest.raises(ValueError, match=field + ".*placeholder"):
        readiness.validate_decisions(decisions, tracked_license_present=False)


def test_private_template_is_mode_0600_and_no_clobber(monkeypatch, tmp_path) -> None:
    path = tmp_path / "private" / "decisions.json"
    monkeypatch.setattr(readiness, "_require_ignored_private_path", lambda _path: None)
    value = readiness.private_decision_template(
        artifact_hashes={
            "approved_anonymous_pdf_sha256": "0" * 64,
            "approved_checklist_sha256": "1" * 64,
            "approved_public_metadata_sha256": "2" * 64,
        }
    )
    first = readiness.write_private_decision_template(path, value)
    assert path.read_bytes() == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="refusing to overwrite"):
        readiness.write_private_decision_template(path, value)


def test_private_template_destination_must_stay_inside_repository(tmp_path) -> None:
    with pytest.raises(ValueError, match="inside the repository"):
        readiness._require_ignored_private_path(tmp_path / "decisions.json")


def test_private_handoff_is_deterministic_and_contains_expected_files(
    monkeypatch, tmp_path
) -> None:
    decisions = readiness.validate_decisions(_decisions(), tracked_license_present=False)
    public = {
        "title": "Paper title",
        "tldr": "One sentence.",
        "abstract": "Abstract.",
        "paper_type": "long",
        "primary_area_recommendation": "Efficient Methods for NLP",
        "contribution_type_recommendations": ["NLP engineering experiment"],
        "previous_url": None,
    }
    checklist = tmp_path / "checklist.md"
    checklist.write_bytes(b"checklist\n")
    monkeypatch.setattr(readiness, "CHECKLIST", checklist)
    first = readiness.build_handoff(
        public, decisions, b"%PDF-test\n", git_commit="a" * 40
    )
    second = readiness.build_handoff(
        public, decisions, b"%PDF-test\n", git_commit="a" * 40
    )
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        names = set(archive.namelist())
        root = readiness.HANDOFF_ROOT
        assert names == {
            f"{root}/HANDOFF_README.md",
            f"{root}/MANIFEST.json",
            f"{root}/openreview-form.json",
            f"{root}/paper.pdf",
            f"{root}/responsible-checklist.md",
        }
        form = json.loads(archive.read(f"{root}/openreview-form.json"))
        manifest = json.loads(archive.read(f"{root}/MANIFEST.json"))
    assert form["authors"][0]["openreview_profile"] == "~Ada_Researcher1"
    assert form["request_public_anonymous_preprint"] is True
    assert form["attach_software"] is False
    assert manifest["git_commit"] == "a" * 40
