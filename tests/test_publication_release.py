from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from scripts import build_arxiv_preprint
from scripts import build_reproducibility_archive


def _release_files(*, include_license: bool) -> dict[str, build_reproducibility_archive.ReleaseFile]:
    files = {
        "README.md": build_reproducibility_archive.ReleaseFile("README.md", b"research\n"),
        "src/jamoflow/example.py": build_reproducibility_archive.ReleaseFile(
            "src/jamoflow/example.py", b"print('ok')\n"
        ),
    }
    if include_license:
        files["LICENSE"] = build_reproducibility_archive.ReleaseFile(
            "LICENSE", b"A deliberately selected test license text.\n"
        )
    return files


def test_named_release_requires_license_and_is_deterministic() -> None:
    commit = "a" * 40
    without = _release_files(include_license=False)
    assert build_reproducibility_archive.audit(commit, without)["public_release_ready"] is False
    with pytest.raises(ValueError, match="tracked LICENSE"):
        build_reproducibility_archive.build_named_public_archive(commit, without)

    files = _release_files(include_license=True)
    first = build_reproducibility_archive.build_named_public_archive(commit, files)
    second = build_reproducibility_archive.build_named_public_archive(commit, files)
    assert first == second
    with tarfile.open(fileobj=io.BytesIO(first), mode="r:gz") as archive:
        names = set(archive.getnames())
        manifest_name = (
            f"{build_reproducibility_archive.ARCHIVE_ROOT}/ARTIFACT_MANIFEST.json"
        )
        assert manifest_name in names
        manifest_file = archive.extractfile(manifest_name)
        assert manifest_file is not None
        manifest = json.loads(manifest_file.read())
    assert manifest["git_commit"] == commit
    assert manifest["anonymous_review_artifact"] is False


def test_release_content_scan_rejects_identity_and_token_patterns() -> None:
    files = _release_files(include_license=True)
    files["docs/leak.md"] = build_reproducibility_archive.ReleaseFile(
        "docs/leak.md",
        b"/Users/" + b"private/repo and " + b"ghp_" + b"12345678901234567890\n",
    )
    findings = build_reproducibility_archive.scan_forbidden_content(files)
    assert {item["pattern"] for item in findings} == {
        "absolute_user_path",
        "github_token",
    }


def test_release_scan_allows_public_github_identity_but_rejects_local_ssh_alias() -> None:
    files = _release_files(include_license=True)
    files["docs/public.md"] = build_reproducibility_archive.ReleaseFile(
        "docs/public.md",
        b"https://github.com/boaz-hwang/JamoFlow-reproducibility\n",
    )
    assert build_reproducibility_archive.scan_forbidden_content(files) == []
    files["docs/local.md"] = build_reproducibility_archive.ReleaseFile(
        "docs/local.md",
        b"git@github" + b"-boaz:boaz-hwang/private.git\n",
    )
    findings = build_reproducibility_archive.scan_forbidden_content(files)
    assert findings == [
        {"path": "docs/local.md", "pattern": "private_github_ssh_alias"}
    ]


def _author_metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "authors": [
            {
                "name": "Ada Researcher",
                "email": "ada@research.org",
                "orcid": "0000-0002-1825-0097",
                "affiliation_ids": ["lab"],
            },
            {
                "name": "Kim Scholar",
                "email": "kim@research.org",
                "orcid": None,
                "affiliation_ids": ["lab"],
            },
        ],
        "affiliations": [
            {"id": "lab", "name": "Research Lab", "address": "Seoul, Republic of Korea"}
        ],
        "acknowledgments": "Supported by a declared research grant.",
    }


def test_preprint_metadata_generates_named_final_author_block() -> None:
    metadata = build_arxiv_preprint.validate_author_metadata(_author_metadata())
    author = build_arxiv_preprint.author_tex(metadata)
    acknowledgments = build_arxiv_preprint.acknowledgments_tex(metadata)
    assert "\\author{" in author
    assert "Ada Researcher" in author
    assert "Kim Scholar" in author
    assert "\\And" in author
    assert "Anonymous" not in author
    assert "\\section*{Acknowledgments}" in acknowledgments
    assert "[review]" not in build_arxiv_preprint.TEMPLATE.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "mutation,match",
    [
        (("authors", 0, "name", "Anonymous"), "placeholder"),
        (("authors", 0, "email", "invalid"), "email is invalid"),
        (("authors", 0, "affiliation_ids", ["missing"]), "unknown affiliations"),
    ],
)
def test_preprint_metadata_rejects_placeholders_and_invalid_identity(
    mutation: tuple[str, int, str, object], match: str
) -> None:
    value = _author_metadata()
    collection, index, key, replacement = mutation
    value[collection][index][key] = replacement  # type: ignore[index]
    with pytest.raises(ValueError, match=match):
        build_arxiv_preprint.validate_author_metadata(value)


def test_release_manifest_hashes_selected_files() -> None:
    files = _release_files(include_license=True)
    manifest = build_reproducibility_archive.release_manifest("b" * 40, files)
    rows = {item["path"]: item for item in manifest["files"]}
    assert rows["README.md"]["sha256"] == hashlib.sha256(b"research\n").hexdigest()
    assert rows["src/jamoflow/example.py"]["mode"] == "0644"
