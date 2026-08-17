#!/usr/bin/env python3
"""Audit ARR readiness and create a private handoff only after all gates pass."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

try:
    from scripts import build_arxiv_preprint
except ImportError:  # Direct execution places scripts/ rather than the repo on sys.path.
    import build_arxiv_preprint  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_METADATA = ROOT / "paper" / "arr-submission-metadata.json"
DECISIONS = ROOT / "paper" / "private" / "arr-submission-decisions.json"
PAPER_SOURCE = ROOT / "paper" / "arr-submission.md"
CHECKLIST = ROOT / "paper" / "arr-responsible-checklist-draft.md"
PDF = ROOT / "build" / "arr" / "main.pdf"
OUTPUT = ROOT / "dist" / "arr-submission-handoff-v1.zip"
HANDOFF_ROOT = "arr-submission-handoff-v1"
ZIP_TIMESTAMP = (2026, 8, 17, 0, 0, 0)
PREPRINT_POLICIES = {
    "arr_anonymous_only_until_meta_review",
    "named_preprint",
    "no_preprint",
}
SOFTWARE_CHOICES = {"none_for_anonymous_review", "deidentified_anonymous_archive"}
PUBLIC_RELEASE_CHOICES = {"after_review", "with_named_preprint", "no_public_release"}
OPENREVIEW_PROFILE = re.compile(r"~[^\s]+\d+$")
HTTPS_URL = re.compile(r"https://[^\s]+$")
DECISION_PLACEHOLDER = re.compile(
    r"\b(?:placeholder|replace|todo|tbd)\b|your name|example author",
    re.I,
)


def canonical_json(value: object, *, pretty: bool = False) -> bytes:
    if pretty:
        return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys differ: {sorted(set(value) ^ expected)}")


def _decision_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is missing")
    if DECISION_PLACEHOLDER.search(value):
        raise ValueError(f"{label} contains placeholder text")
    return value.strip()


def private_decision_template(*, artifact_hashes: dict[str, str]) -> dict[str, Any]:
    expected = {
        "approved_anonymous_pdf_sha256",
        "approved_checklist_sha256",
        "approved_public_metadata_sha256",
    }
    _exact_keys(artifact_hashes, expected, "template artifact hashes")
    for label, digest in artifact_hashes.items():
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"template artifact hash is invalid: {label}")
    return {
        "schema_version": 1,
        "authors": [
            {
                "name": "TODO: full publication name",
                "email": "TODO: publication email",
                "openreview_profile": "TODO: OpenReview profile such as ~Name_1",
                "orcid": None,
                "affiliation_ids": ["affiliation-1"],
                "contribution": "TODO: concrete authorship contribution",
                "authorship_confirmed": False,
                "reviewer_registration_ready": False,
            }
        ],
        "affiliations": [
            {
                "id": "affiliation-1",
                "name": "TODO: institution name",
                "address": "TODO: institution postal address",
            }
        ],
        "acknowledgments": None,
        "funding": [],
        "conflicts_of_interest": [],
        "author_order_confirmed": False,
        "paper_approved_by_all_authors": False,
        "checklist_approved_by_all_authors": False,
        **artifact_hashes,
        "preferred_venue": "TODO: preferred venue",
        "preprint_policy": "TODO: choose a policy from the schema",
        "existing_preprints": [],
        "software_archive_choice": "TODO: choose a software policy from the schema",
        "public_code_release_choice": "TODO: choose a release policy from the schema",
        "code_license": None,
        "consent_to_share_anonymized_metadata": None,
        "consent_to_review": False,
    }


def author_metadata_from_decisions(value: dict[str, Any]) -> dict[str, Any]:
    authors = []
    for author in value.get("authors", []):
        if not isinstance(author, dict):
            raise ValueError("each author must be an object")
        authors.append(
            {
                "name": author.get("name"),
                "email": author.get("email"),
                "orcid": author.get("orcid"),
                "affiliation_ids": author.get("affiliation_ids"),
            }
        )
    return build_arxiv_preprint.validate_author_metadata(
        {
            "schema_version": 1,
            "authors": authors,
            "affiliations": value.get("affiliations"),
            "acknowledgments": value.get("acknowledgments"),
        }
    )


def validate_decisions(
    value: object,
    *,
    tracked_license_present: bool,
    expected_artifact_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("submission decisions must be an object")
    expected = {
        "schema_version",
        "authors",
        "affiliations",
        "acknowledgments",
        "funding",
        "conflicts_of_interest",
        "author_order_confirmed",
        "paper_approved_by_all_authors",
        "checklist_approved_by_all_authors",
        "approved_anonymous_pdf_sha256",
        "approved_checklist_sha256",
        "approved_public_metadata_sha256",
        "preferred_venue",
        "preprint_policy",
        "existing_preprints",
        "software_archive_choice",
        "public_code_release_choice",
        "code_license",
        "consent_to_share_anonymized_metadata",
        "consent_to_review",
    }
    _exact_keys(value, expected, "submission decisions")
    if value["schema_version"] != 1:
        raise ValueError("unsupported submission decision schema")
    metadata = author_metadata_from_decisions(value)
    authors = value["authors"]
    for index, author in enumerate(authors):
        _exact_keys(
            author,
            {
                "name",
                "email",
                "openreview_profile",
                "orcid",
                "affiliation_ids",
                "contribution",
                "authorship_confirmed",
                "reviewer_registration_ready",
            },
            f"author {index}",
        )
        profile = author["openreview_profile"]
        if not isinstance(profile, str) or not OPENREVIEW_PROFILE.fullmatch(profile):
            raise ValueError(f"author {index} OpenReview profile is invalid")
        _decision_text(author["contribution"], f"author {index} contribution")
        if author["authorship_confirmed"] is not True:
            raise ValueError(f"author {index} has not confirmed authorship")
        if author["reviewer_registration_ready"] is not True:
            raise ValueError(f"author {index} is not reviewer-registration ready")

    for field in (
        "author_order_confirmed",
        "paper_approved_by_all_authors",
        "checklist_approved_by_all_authors",
        "consent_to_review",
    ):
        if value[field] is not True:
            raise ValueError(f"required author decision is not approved: {field}")
    hash_fields = {
        "approved_anonymous_pdf_sha256",
        "approved_checklist_sha256",
        "approved_public_metadata_sha256",
    }
    for field in hash_fields:
        digest = value[field]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"approved artifact hash is invalid: {field}")
        if expected_artifact_hashes is not None and digest != expected_artifact_hashes[field]:
            raise ValueError(f"approved artifact has changed: {field}")
    if not isinstance(value["consent_to_share_anonymized_metadata"], bool):
        raise ValueError("metadata-sharing consent must be true or false")
    _decision_text(value["preferred_venue"], "preferred venue")
    if value["preprint_policy"] not in PREPRINT_POLICIES:
        raise ValueError("preprint policy is invalid")
    urls = value["existing_preprints"]
    if not isinstance(urls, list) or len(urls) != len(set(urls)):
        raise ValueError("existing preprints must be a unique list")
    if any(not isinstance(url, str) or not HTTPS_URL.fullmatch(url) for url in urls):
        raise ValueError("existing preprint URLs must use https")
    if value["preprint_policy"] == "arr_anonymous_only_until_meta_review" and urls:
        raise ValueError("anonymous-only policy conflicts with an existing named preprint")
    if value["software_archive_choice"] not in SOFTWARE_CHOICES:
        raise ValueError("software archive choice is invalid")
    if value["software_archive_choice"] == "deidentified_anonymous_archive":
        raise ValueError("no reviewed de-identified anonymous software archive exists")
    if value["public_code_release_choice"] not in PUBLIC_RELEASE_CHOICES:
        raise ValueError("public release choice is invalid")
    code_license = value["code_license"]
    if code_license is not None:
        _decision_text(code_license, "code license")
    if value["public_code_release_choice"] == "with_named_preprint":
        if code_license is None or not tracked_license_present:
            raise ValueError("named code release requires a selected tracked license")
    for field in ("funding", "conflicts_of_interest"):
        items = value[field]
        if not isinstance(items, list):
            raise ValueError(f"{field} must be a list of nonempty strings")
        for index, item in enumerate(items):
            _decision_text(item, f"{field} item {index}")

    normalized = dict(value)
    normalized["authors"] = [dict(author) for author in authors]
    normalized["author_metadata"] = metadata
    return normalized


def _tracked_license_present() -> bool:
    for path in ("LICENSE", "LICENSE.md", "COPYING"):
        result = subprocess.run(
            ["git", "cat-file", "-e", f"HEAD:{path}"],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            return True
    return False


def _public_metadata() -> dict[str, Any]:
    value = json.loads(PUBLIC_METADATA.read_text(encoding="utf-8"))
    if value.get("source") != "paper/arr-submission.md":
        raise ValueError("public metadata points to the wrong paper source")
    source = PAPER_SOURCE.read_text(encoding="utf-8")
    front = source.split("---", 2)[1]
    title = re.search(r'^title: "(.+)"$', front, re.MULTILINE)
    if title is None or value.get("title") != title.group(1):
        raise ValueError("public metadata title differs from the paper")
    abstract_lines: list[str] = []
    active = False
    for line in front.splitlines():
        if line == "abstract: |":
            active = True
            continue
        if active:
            if not line.startswith("  "):
                break
            abstract_lines.append(line[2:])
    if value.get("abstract") != " ".join(abstract_lines):
        raise ValueError("public metadata abstract differs from the paper")
    return value


def _git_identity() -> dict[str, object]:
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("could not resolve the submission commit")
    if status:
        raise ValueError("submission handoff requires a clean worktree")
    return {"commit": commit, "clean": True}


def _pdf_identity() -> dict[str, object]:
    if not PDF.is_file():
        raise ValueError("anonymous ARR PDF is missing; run scripts/build_arr_paper.py")
    value = PDF.read_bytes()
    if not value.startswith(b"%PDF-"):
        raise ValueError("anonymous ARR output is not a PDF")
    text = subprocess.run(
        ["pdftotext", str(PDF), "-"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    if "Anonymous ACL submission" not in text:
        raise ValueError("ARR PDF is not anonymous")
    info = subprocess.run(
        ["pdfinfo", str(PDF)],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    pages = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    page_size = re.search(r"^Page size:\s+(.+)$", info, re.MULTILINE)
    if pages is None or page_size is None or "A4" not in page_size.group(1):
        raise ValueError("ARR PDF does not expose the expected A4 page contract")
    page_text = text.split("\f")
    conclusion_page = next(
        (index + 1 for index, page in enumerate(page_text) if "Conclusion" in page), None
    )
    if conclusion_page is None or conclusion_page > 8:
        raise ValueError("ARR main content exceeds the eight-page limit")
    return {
        "path": str(PDF.relative_to(ROOT)),
        "bytes": len(value),
        "sha256": sha256(value),
        "pages": int(pages.group(1)),
        "page_size": page_size.group(1),
        "conclusion_page": conclusion_page,
    }


def _require_ignored_private_path(path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("private decision template must stay inside the repository") from error
    current = ROOT.resolve()
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise ValueError(f"private decision parent must not be a symlink: {current}")
    ignored = subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ignored.returncode != 0:
        raise ValueError("private decision destination is not protected by .gitignore")


def write_private_decision_template(path: Path, value: dict[str, Any]) -> bytes:
    _require_ignored_private_path(path)
    data = canonical_json(value, pretty=True)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError(f"refusing to overwrite private decisions: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return data


def audit_readiness(decisions_path: Path = DECISIONS) -> dict[str, object]:
    errors: list[str] = []
    public: dict[str, Any] | None = None
    pdf: dict[str, object] | None = None
    decisions: dict[str, Any] | None = None
    git: dict[str, object] | None = None
    public_sha256 = sha256(PUBLIC_METADATA.read_bytes()) if PUBLIC_METADATA.is_file() else None
    checklist_sha256 = sha256(CHECKLIST.read_bytes()) if CHECKLIST.is_file() else None
    try:
        public = _public_metadata()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    if not CHECKLIST.is_file():
        errors.append("Responsible NLP checklist draft is missing")
    try:
        pdf = _pdf_identity()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        errors.append(str(error))
    try:
        git = _git_identity()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        errors.append(str(error))
    if not decisions_path.is_file():
        errors.append(
            "private submission decisions are missing; see "
            "paper/arr-private-decisions.schema.json or run "
            "scripts/audit_arr_submission_readiness.py --write-private-template"
        )
    else:
        try:
            decisions = validate_decisions(
                json.loads(decisions_path.read_text(encoding="utf-8")),
                tracked_license_present=_tracked_license_present(),
                expected_artifact_hashes=(
                    {
                        "approved_anonymous_pdf_sha256": str(pdf["sha256"]),
                        "approved_checklist_sha256": str(checklist_sha256),
                        "approved_public_metadata_sha256": str(public_sha256),
                    }
                    if pdf is not None
                    and checklist_sha256 is not None
                    and public_sha256 is not None
                    else None
                ),
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(str(error))
    return {
        "schema_version": 1,
        "kind": "arr_submission_readiness_audit_v1",
        "ready": not errors,
        "errors": errors,
        "public_metadata_valid": public is not None,
        "checklist_present": CHECKLIST.is_file(),
        "anonymous_pdf": pdf,
        "public_metadata_sha256": public_sha256,
        "checklist_sha256": checklist_sha256,
        "git": git,
        "private_decisions_present": decisions_path.is_file(),
        "private_decisions_valid": decisions is not None,
        "tracked_license_present": _tracked_license_present(),
    }


def _openreview_form(public: dict[str, Any], decisions: dict[str, Any]) -> dict[str, object]:
    return {
        "title": public["title"],
        "authors": [
            {
                "name": author["name"],
                "email": author["email"],
                "openreview_profile": author["openreview_profile"],
            }
            for author in decisions["authors"]
        ],
        "tldr": public["tldr"],
        "abstract": public["abstract"],
        "paper_type": public["paper_type"],
        "area": public["primary_area_recommendation"],
        "contribution_type_recommendations": public["contribution_type_recommendations"],
        "previous_url": public["previous_url"],
        "preferred_venue": decisions["preferred_venue"],
        "request_public_anonymous_preprint": (
            decisions["preprint_policy"] == "arr_anonymous_only_until_meta_review"
        ),
        "existing_preprints": decisions["existing_preprints"],
        "attach_software": decisions["software_archive_choice"]
        != "none_for_anonymous_review",
        "attach_data": False,
        "consent_to_share_anonymized_metadata": decisions[
            "consent_to_share_anonymized_metadata"
        ],
        "consent_to_review": decisions["consent_to_review"],
        "responsible_checklist_approved": decisions["checklist_approved_by_all_authors"],
    }


def _handoff_readme(pdf_sha256: str) -> bytes:
    return f"""# Private ARR submission handoff

This local archive is a transfer aid, not a single file to upload. Upload
`paper.pdf` to the ARR/OpenReview PDF field and copy the fields from
`openreview-form.json`. Enter the Responsible NLP answers from
`responsible-checklist.md` after a final author review.

Expected anonymous PDF SHA-256: `{pdf_sha256}`.

Do not upload this ZIP as public supplementary material: it contains author and
OpenReview identity data. No network submission has been performed by the
builder. Re-check the rendered OpenReview form and PDF preview before clicking
Submit.
""".encode("utf-8")


def build_handoff(
    public: dict[str, Any], decisions: dict[str, Any], pdf: bytes, *, git_commit: str
) -> bytes:
    files = {
        "paper.pdf": pdf,
        "openreview-form.json": canonical_json(_openreview_form(public, decisions), pretty=True),
        "responsible-checklist.md": CHECKLIST.read_bytes(),
        "HANDOFF_README.md": _handoff_readme(sha256(pdf)),
    }
    manifest = {
        "schema_version": 1,
        "kind": "private_arr_submission_handoff_v1",
        "git_commit": git_commit,
        "files": [
            {"path": path, "bytes": len(value), "sha256": sha256(value)}
            for path, value in sorted(files.items())
        ],
    }
    files["MANIFEST.json"] = canonical_json(manifest, pretty=True)
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, value in sorted(files.items()):
            info = zipfile.ZipInfo(f"{HANDOFF_ROOT}/{path}", date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100600 << 16
            info.create_system = 3
            archive.writestr(info, value, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=Path, default=DECISIONS)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--write-handoff", action="store_true")
    output.add_argument("--write-private-template", action="store_true")
    args = parser.parse_args()
    decisions_path = args.decisions.resolve()
    if args.write_private_template:
        _git_identity()
        _public_metadata()
        if not CHECKLIST.is_file():
            raise ValueError("Responsible NLP checklist draft is missing")
        pdf = _pdf_identity()
        value = private_decision_template(
            artifact_hashes={
                "approved_anonymous_pdf_sha256": str(pdf["sha256"]),
                "approved_checklist_sha256": sha256(CHECKLIST.read_bytes()),
                "approved_public_metadata_sha256": sha256(PUBLIC_METADATA.read_bytes()),
            }
        )
        data = write_private_decision_template(decisions_path, value)
        print(f"private_template={decisions_path}")
        print(f"private_template_bytes={len(data)}")
        print("private_template_status=invalid_until_all_TODO_and_false_fields_are_reviewed")
        return
    report = audit_readiness(decisions_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.write_handoff:
        return
    if not report["ready"]:
        raise ValueError("submission handoff is blocked by readiness errors")
    public = _public_metadata()
    decisions = validate_decisions(
        json.loads(decisions_path.read_text(encoding="utf-8")),
        tracked_license_present=_tracked_license_present(),
        expected_artifact_hashes={
            "approved_anonymous_pdf_sha256": sha256(PDF.read_bytes()),
            "approved_checklist_sha256": sha256(CHECKLIST.read_bytes()),
            "approved_public_metadata_sha256": sha256(PUBLIC_METADATA.read_bytes()),
        },
    )
    git = _git_identity()
    value = build_handoff(
        public, decisions, PDF.read_bytes(), git_commit=str(git["commit"])
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != value:
            raise ValueError(f"refusing to overwrite differing handoff: {OUTPUT}")
    else:
        with OUTPUT.open("xb") as handle:
            handle.write(value)
    print(f"handoff={OUTPUT}")
    print(f"handoff_bytes={len(value)}")
    print(f"handoff_sha256={sha256(value)}")


if __name__ == "__main__":
    main()
