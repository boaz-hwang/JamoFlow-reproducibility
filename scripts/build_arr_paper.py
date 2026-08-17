#!/usr/bin/env python3
"""Build and validate the anonymous ARR paper with a pinned official ACL style."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper" / "arr-submission.md"
TEMPLATE = ROOT / "paper" / "acl-template.tex"
TABLE_FILTER = ROOT / "paper" / "filters" / "acl-tables.lua"
REFERENCES = ROOT / "paper" / "references.bib"
FIGURES = ROOT / "paper" / "figures"
BUILD = ROOT / "build" / "arr"

ACL_STYLE_COMMIT = "d5adc823ff0f80f98c80405ca0ab66c68e684409"
SOURCE_DATE_EPOCH = "1786924800"  # 2026-08-17T00:00:00Z, the source date.
ACL_FILES = {
    "acl.sty": "19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de",
    "acl_natbib.bst": "6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run(
    command: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env is not None:
        process_env.update(env)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=process_env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}"
        )
    return result


def require_program(name: str) -> str:
    value = shutil.which(name)
    if value is None:
        raise RuntimeError(f"required program is unavailable: {name}")
    return value


def fetch_acl_files() -> None:
    for name, expected in ACL_FILES.items():
        url = (
            "https://raw.githubusercontent.com/acl-org/acl-style-files/"
            f"{ACL_STYLE_COMMIT}/{name}"
        )
        with urllib.request.urlopen(url, timeout=30) as response:
            value = response.read()
        actual = sha256_bytes(value)
        if actual != expected:
            raise ValueError(f"official ACL file hash mismatch: {name}: {actual}")
        (BUILD / name).write_bytes(value)


def extract_abstract_words() -> int:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    try:
        start = lines.index("abstract: |") + 1
    except ValueError as error:
        raise ValueError("submission source is missing YAML abstract") from error
    abstract_lines: list[str] = []
    for line in lines[start:]:
        if line == "---":
            break
        if not line.startswith("  "):
            raise ValueError("abstract YAML must use a two-space literal block")
        abstract_lines.append(line[2:])
    words = re.findall(r"\b[\w.+%'-]+\b", " ".join(abstract_lines))
    return len(words)


def prepare_build_tree() -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)
    fetch_acl_files()
    shutil.copy2(REFERENCES, BUILD / "references.bib")
    shutil.copytree(FIGURES, BUILD / "figures")


def build_tex() -> None:
    pandoc = require_program("pandoc")
    result = run(
        [
            pandoc,
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
    if result.stdout.strip():
        print(result.stdout, end="")


def compile_pdf() -> None:
    tectonic = require_program("tectonic")
    result = run(
        [tectonic, "--keep-logs", "--keep-intermediates", "main.tex"],
        cwd=BUILD,
        env={"SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH},
    )
    warnings = [line for line in result.stdout.splitlines() if line.startswith("warning:")]
    non_layout = [
        line
        for line in warnings
        if "Underfull" not in line and "lineno.sty:296: Invalid UTF-8" not in line
    ]
    print(f"tectonic_warning_count={len(warnings)}")
    for line in non_layout:
        print(line)


def validate_pdf() -> str:
    pdf = BUILD / "main.pdf"
    if not pdf.is_file():
        raise ValueError("paper PDF was not produced")
    info = run([require_program("pdfinfo"), str(pdf)]).stdout
    page_match = re.search(r"^Pages:\s+(\d+)\s*$", info, re.MULTILINE)
    size_match = re.search(r"^Page size:\s+(.+)$", info, re.MULTILINE)
    if page_match is None or size_match is None:
        raise ValueError("pdfinfo did not expose pages and page size")
    page_size = size_match.group(1)
    if "595.276 x 841.89 pts" not in page_size and "A4" not in page_size:
        raise ValueError(f"paper is not A4: {page_size}")

    fonts = run([require_program("pdffonts"), str(pdf)]).stdout.splitlines()
    font_rows = [line.split() for line in fonts[2:] if line.strip()]
    if not font_rows:
        raise ValueError("no embedded fonts were reported")
    if any(len(row) < 6 or row[5].lower() != "yes" for row in font_rows):
        raise ValueError("one or more PDF fonts are not embedded")

    image_lines = run([require_program("pdfimages"), "-list", str(pdf)]).stdout.splitlines()
    image_rows = [line.split() for line in image_lines if line.strip()[:1].isdigit()]
    raster_rows = [row for row in image_rows if len(row) >= 8 and row[2] == "image"]
    mask_rows = [row for row in image_rows if len(row) >= 3 and row[2] in {"mask", "smask"}]
    if not raster_rows:
        raise ValueError("the result figure is missing from the PDF")
    if mask_rows:
        raise ValueError("the PDF contains a raster transparency mask")
    if any(row[7] != "8" for row in raster_rows):
        raise ValueError("one or more PDF raster figures are not 8-bit")

    log = (BUILD / "main.log").read_text(encoding="utf-8", errors="replace")
    blg = (BUILD / "main.blg").read_text(encoding="utf-8", errors="replace")
    if "Overfull \\hbox" in log:
        raise ValueError("the ACL PDF contains an overfull horizontal box")
    if "undefined citations" in log.lower() or "undefined references" in log.lower():
        raise ValueError("the ACL PDF contains unresolved citations or references")
    if "error message" in blg.lower() or "illegal," in blg.lower():
        raise ValueError(f"BibTeX reported an error:\n{blg}")

    text = run([require_program("pdftotext"), "-layout", str(pdf), "-"]).stdout
    pages = text.split("\f")
    conclusion_page = next(
        (index + 1 for index, page in enumerate(pages) if "Conclusion" in page), None
    )
    limitations_page = next(
        (index + 1 for index, page in enumerate(pages) if "Limitations" in page), None
    )
    if conclusion_page is None or limitations_page is None:
        raise ValueError("could not locate Conclusion and Limitations in the PDF")
    if conclusion_page > 8:
        raise ValueError(f"main content exceeds eight pages: conclusion starts on {conclusion_page}")
    if limitations_page > 9:
        raise ValueError(f"limitations starts too late for an eight-page main paper: {limitations_page}")

    pdf_sha256 = sha256_file(pdf)
    print(f"pdf={pdf.relative_to(ROOT)}")
    print(f"pdf_sha256={pdf_sha256}")
    print(f"pages={page_match.group(1)}")
    print(f"page_size={page_size}")
    print(f"conclusion_page={conclusion_page}")
    print(f"limitations_page={limitations_page}")
    print(f"embedded_font_count={len(font_rows)}")
    print(f"embedded_raster_count={len(raster_rows)}")
    return pdf_sha256


def build_pdf() -> str:
    prepare_build_tree()
    build_tex()
    compile_pdf()
    return validate_pdf()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-reproducible",
        action="store_true",
        help="build twice and require byte-identical PDFs",
    )
    args = parser.parse_args()
    abstract_words = extract_abstract_words()
    if abstract_words > 200:
        raise ValueError(f"abstract exceeds 200 words: {abstract_words}")
    print(f"abstract_words={abstract_words}")
    print(f"source_date_epoch={SOURCE_DATE_EPOCH}")
    run([sys.executable, str(ROOT / "scripts" / "generate_paper_figures.py"), "--verify"])
    first_sha256 = build_pdf()
    if args.verify_reproducible:
        second_sha256 = build_pdf()
        if first_sha256 != second_sha256:
            raise ValueError(
                "paper build is not byte-reproducible: "
                f"{first_sha256} != {second_sha256}"
            )
        print(f"reproducible_pdf_sha256={first_sha256}")


if __name__ == "__main__":
    main()
