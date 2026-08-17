#!/usr/bin/env python3
"""Build a named final-mode preprint and deterministic arXiv source archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

try:
    from scripts import build_arr_paper
except ImportError:  # Direct execution places scripts/ rather than the repo on sys.path.
    import build_arr_paper  # type: ignore[no-redef]


ACL_FILES = build_arr_paper.ACL_FILES
ACL_STYLE_COMMIT = build_arr_paper.ACL_STYLE_COMMIT
FIGURES = build_arr_paper.FIGURES
REFERENCES = build_arr_paper.REFERENCES
ROOT = build_arr_paper.ROOT
SOURCE = build_arr_paper.SOURCE
SOURCE_DATE_EPOCH = build_arr_paper.SOURCE_DATE_EPOCH
TABLE_FILTER = build_arr_paper.TABLE_FILTER
require_program = build_arr_paper.require_program
run = build_arr_paper.run


TEMPLATE = ROOT / "paper" / "acl-preprint-template.tex"
DEFAULT_METADATA = ROOT / "paper" / "private" / "arr-submission-decisions.json"
BUILD = ROOT / "build" / "arxiv"
OUTPUT = ROOT / "dist" / "jamoflow-arxiv-source.tar.gz"
PLACEHOLDER = re.compile(r"anonymous|placeholder|your name|example author|\btodo\b|\btbd\b", re.I)
ORCID = re.compile(r"\d{4}-\d{4}-\d{4}-\d{3}[\dX]")
EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys differ: {sorted(set(value) ^ expected)}")


def _real_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    result = value.strip()
    if PLACEHOLDER.search(result) or "..." in result:
        raise ValueError(f"{label} contains placeholder text")
    return result


def validate_author_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("author metadata must be a JSON object")
    _exact_keys(value, {"schema_version", "authors", "affiliations", "acknowledgments"}, "metadata")
    if value["schema_version"] != 1:
        raise ValueError("unsupported author metadata schema")
    authors = value["authors"]
    affiliations = value["affiliations"]
    if not isinstance(authors, list) or not authors:
        raise ValueError("at least one author is required")
    if not isinstance(affiliations, list) or not affiliations:
        raise ValueError("at least one affiliation is required")

    affiliation_map: dict[str, dict[str, str]] = {}
    for index, affiliation in enumerate(affiliations):
        if not isinstance(affiliation, dict):
            raise ValueError(f"affiliation {index} must be an object")
        _exact_keys(affiliation, {"id", "name", "address"}, f"affiliation {index}")
        affiliation_id = _real_text(affiliation["id"], f"affiliation {index} id")
        if affiliation_id in affiliation_map:
            raise ValueError(f"duplicate affiliation id: {affiliation_id}")
        affiliation_map[affiliation_id] = {
            "id": affiliation_id,
            "name": _real_text(affiliation["name"], f"affiliation {index} name"),
            "address": _real_text(affiliation["address"], f"affiliation {index} address"),
        }

    normalized_authors: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, author in enumerate(authors):
        if not isinstance(author, dict):
            raise ValueError(f"author {index} must be an object")
        _exact_keys(author, {"name", "email", "orcid", "affiliation_ids"}, f"author {index}")
        name = _real_text(author["name"], f"author {index} name")
        email = _real_text(author["email"], f"author {index} email")
        if not EMAIL.fullmatch(email):
            raise ValueError(f"author {index} email is invalid")
        if name.casefold() in seen_names:
            raise ValueError(f"duplicate author name: {name}")
        seen_names.add(name.casefold())
        orcid = author["orcid"]
        if orcid is not None and (not isinstance(orcid, str) or not ORCID.fullmatch(orcid)):
            raise ValueError(f"author {index} ORCID is invalid")
        ids = author["affiliation_ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
            raise ValueError(f"author {index} affiliation_ids are invalid")
        if len(ids) != len(set(ids)) or any(item not in affiliation_map for item in ids):
            raise ValueError(f"author {index} has duplicate or unknown affiliations")
        normalized_authors.append(
            {"name": name, "email": email, "orcid": orcid, "affiliation_ids": ids}
        )

    acknowledgments = value["acknowledgments"]
    if acknowledgments is not None:
        acknowledgments = _real_text(acknowledgments, "acknowledgments")
    return {
        "schema_version": 1,
        "authors": normalized_authors,
        "affiliations": list(affiliation_map.values()),
        "acknowledgments": acknowledgments,
    }


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def author_tex(metadata: dict[str, Any]) -> str:
    affiliation_map = {item["id"]: item for item in metadata["affiliations"]}
    rows = []
    for author in metadata["authors"]:
        affiliations = [affiliation_map[item] for item in author["affiliation_ids"]]
        lines = [tex_escape(author["name"])]
        lines.extend(tex_escape(item["name"]) for item in affiliations)
        lines.extend(tex_escape(item["address"]) for item in affiliations)
        lines.append(r"\texttt{" + tex_escape(author["email"]) + "}")
        if author["orcid"] is not None:
            lines.append("ORCID: " + tex_escape(author["orcid"]))
        rows.append((" " + r"\\" + "\n").join(lines))
    return "\\author{\n" + "\n\\And\n".join(rows) + "\n}\n"


def acknowledgments_tex(metadata: dict[str, Any]) -> str:
    value = metadata["acknowledgments"]
    if value is None:
        return "% No acknowledgments supplied.\n"
    return "\\section*{Acknowledgments}\n" + tex_escape(value) + "\n"


def author_metadata_from_submission_decisions(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("authors"), list):
        raise ValueError("submission decisions do not contain an author list")
    if value.get("preprint_policy") != "named_preprint":
        raise ValueError("submission decisions do not authorize a named preprint")
    for field in (
        "author_order_confirmed",
        "paper_approved_by_all_authors",
        "checklist_approved_by_all_authors",
    ):
        if value.get(field) is not True:
            raise ValueError(f"submission decisions have not approved: {field}")
    authors = []
    for author in value["authors"]:
        if not isinstance(author, dict):
            raise ValueError("submission decision author must be an object")
        if author.get("authorship_confirmed") is not True:
            raise ValueError("submission decision author has not confirmed authorship")
        authors.append(
            {
                "name": author.get("name"),
                "email": author.get("email"),
                "orcid": author.get("orcid"),
                "affiliation_ids": author.get("affiliation_ids"),
            }
        )
    return validate_author_metadata(
        {
            "schema_version": 1,
            "authors": authors,
            "affiliations": value.get("affiliations"),
            "acknowledgments": value.get("acknowledgments"),
        }
    )


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(
            f"private author metadata is missing: {path}; see "
            "paper/arr-private-decisions.schema.json"
        )
    value = json.loads(path.read_text(encoding="utf-8"))
    authors = value.get("authors") if isinstance(value, dict) else None
    if (
        isinstance(authors, list)
        and authors
        and isinstance(authors[0], dict)
        and "openreview_profile" in authors[0]
    ):
        return author_metadata_from_submission_decisions(value)
    return validate_author_metadata(value)


def _prepare(metadata: dict[str, Any]) -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    build_arr_paper.BUILD = BUILD
    build_arr_paper.fetch_acl_files()
    shutil.copy2(REFERENCES, BUILD / "references.bib")
    shutil.copytree(FIGURES, BUILD / "figures")
    (BUILD / "author.tex").write_text(author_tex(metadata), encoding="utf-8")
    (BUILD / "acknowledgments.tex").write_text(
        acknowledgments_tex(metadata), encoding="utf-8"
    )


def _build_tex() -> None:
    run(
        [
            require_program("pandoc"),
            str(SOURCE),
            "--from=markdown",
            "--to=latex",
            "--natbib",
            f"--bibliography={REFERENCES}",
            f"--resource-path={ROOT / 'paper'}",
            f"--template={TEMPLATE}",
            f"--lua-filter={TABLE_FILTER}",
            "--output",
            str(BUILD / "main.tex"),
        ]
    )


def _compile() -> None:
    result = subprocess.run(
        [require_program("tectonic"), "--keep-logs", "--keep-intermediates", "main.tex"],
        cwd=BUILD,
        env={**os.environ, "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH},
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stdout)


def _validate_pdf(metadata: dict[str, Any]) -> None:
    pdf = BUILD / "main.pdf"
    if not pdf.is_file():
        raise ValueError("preprint PDF was not built")
    text = run([require_program("pdftotext"), str(pdf), "-"]).stdout
    if "Anonymous ACL submission" in text:
        raise ValueError("named preprint remained anonymous")
    for author in metadata["authors"]:
        if author["name"] not in text:
            raise ValueError(f"author missing from preprint PDF: {author['name']}")
    if "\\usepackage[review]{acl}" in (BUILD / "main.tex").read_text(encoding="utf-8"):
        raise ValueError("preprint source uses review mode")


def _source_archive() -> bytes:
    names = [
        "main.tex",
        "main.bbl",
        "author.tex",
        "acknowledgments.tex",
        "references.bib",
        "acl.sty",
        "acl_natbib.bst",
    ]
    tex = (BUILD / "main.tex").read_text(encoding="utf-8")
    figure_names = sorted(set(re.findall(r"\{(figures/[^{}]+)\}", tex)))
    names.extend(figure_names)
    for name in names:
        if not (BUILD / name).is_file():
            raise ValueError(f"arXiv source dependency is missing: {name}")

    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=int(SOURCE_DATE_EPOCH)) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name in sorted(names):
                data = (BUILD / name).read_bytes()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                info.mtime = int(SOURCE_DATE_EPOCH)
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def _build_once(metadata_path: Path) -> tuple[bytes, bytes]:
    metadata = _load_metadata(metadata_path)
    _prepare(metadata)
    _build_tex()
    _compile()
    _validate_pdf(metadata)
    return (BUILD / "main.pdf").read_bytes(), _source_archive()


def build(metadata_path: Path, *, verify_reproducible: bool = False) -> tuple[str, str]:
    pdf, source = _build_once(metadata_path)
    if verify_reproducible:
        second_pdf, second_source = _build_once(metadata_path)
        if pdf != second_pdf:
            raise ValueError("named preprint PDF is not byte-reproducible")
        if source != second_source:
            raise ValueError("arXiv source archive is not byte-reproducible")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        if OUTPUT.read_bytes() != source:
            raise ValueError(f"refusing to overwrite differing preprint archive: {OUTPUT}")
    else:
        with OUTPUT.open("xb") as handle:
            handle.write(source)
    return (
        hashlib.sha256(pdf).hexdigest(),
        hashlib.sha256(source).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--validate-metadata", action="store_true")
    parser.add_argument("--verify-reproducible", action="store_true")
    args = parser.parse_args()
    metadata = _load_metadata(args.metadata.resolve())
    print(f"author_count={len(metadata['authors'])}")
    print(f"acl_style_commit={ACL_STYLE_COMMIT}")
    print(f"acl_files={json.dumps(ACL_FILES, sort_keys=True)}")
    if args.validate_metadata:
        return
    pdf_sha256, archive_sha256 = build(
        args.metadata.resolve(), verify_reproducible=args.verify_reproducible
    )
    print(f"pdf={BUILD / 'main.pdf'}")
    print(f"pdf_sha256={pdf_sha256}")
    print(f"source_archive={OUTPUT}")
    print(f"source_archive_sha256={archive_sha256}")


if __name__ == "__main__":
    main()
