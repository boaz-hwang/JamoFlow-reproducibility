"""Label-blind benchmark contamination reference detector.

The functions here prioritize a transparent, exact specification.  A later
indexed corpus runner may accelerate candidate retrieval, but every reported
match must be rechecked by this implementation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import math
import re
import unicodedata
from typing import Mapping


CONTAMINATION_MINIMUM_EXACT_SCALARS = 20
CONTAMINATION_MINIMUM_INFORMATIVE_SCALARS = 8
CONTAMINATION_SHINGLE_SCALARS = 5
CONTAMINATION_MINIMUM_SHARED_SHINGLES = 10
CONTAMINATION_BENCHMARK_COVERAGE_THRESHOLD = 0.80
CONTAMINATION_MINIMUM_LOCAL_LENGTH_RATIO = 0.80
CONTAMINATION_MAXIMUM_LOCAL_LENGTH_RATIO = 1.25
CONTAMINATION_INDEX_VERSION = "publication-v1-reference-complete"

_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


@dataclass(frozen=True, slots=True)
class BenchmarkFingerprint:
    benchmark_id: str
    canonical_sha256: str
    canonical_scalars: int
    informative_scalars: int
    shingle_count: int
    eligible_for_exact: bool
    eligible_for_near: bool

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContaminationMatch:
    benchmark_id: str
    benchmark_canonical_sha256: str
    match_type: str
    document_start_scalar: int
    document_end_scalar: int
    benchmark_scalars: int
    local_scalars: int
    shared_shingles: int
    benchmark_shingle_coverage: float
    shingle_jaccard: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def canonicalize_contamination_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("contamination input must be text")
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    return _WHITESPACE.sub(" ", normalized).strip()


def _informative_scalar_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def scalar_shingles(text: str, width: int = CONTAMINATION_SHINGLE_SCALARS) -> frozenset[str]:
    if width <= 0:
        raise ValueError("shingle width must be positive")
    if len(text) < width:
        return frozenset()
    return frozenset(text[index : index + width] for index in range(len(text) - width + 1))


def benchmark_fingerprint(benchmark_id: str, text: str) -> BenchmarkFingerprint:
    if not benchmark_id:
        raise ValueError("benchmark id is required")
    canonical = canonicalize_contamination_text(text)
    informative = _informative_scalar_count(canonical)
    shingles = scalar_shingles(canonical)
    exact = bool(
        len(canonical) >= CONTAMINATION_MINIMUM_EXACT_SCALARS
        and informative >= CONTAMINATION_MINIMUM_INFORMATIVE_SCALARS
    )
    near = bool(
        exact and len(shingles) >= CONTAMINATION_MINIMUM_SHARED_SHINGLES
    )
    return BenchmarkFingerprint(
        benchmark_id=benchmark_id,
        canonical_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        canonical_scalars=len(canonical),
        informative_scalars=informative,
        shingle_count=len(shingles),
        eligible_for_exact=exact,
        eligible_for_near=near,
    )


def _candidate_local_starts(document: str, benchmark: str) -> tuple[int, ...]:
    width = CONTAMINATION_SHINGLE_SCALARS
    benchmark_positions: dict[str, list[int]] = defaultdict(list)
    for index in range(len(benchmark) - width + 1):
        benchmark_positions[benchmark[index : index + width]].append(index)
    starts: set[int] = set()
    for document_index in range(len(document) - width + 1):
        shingle = document[document_index : document_index + width]
        for benchmark_index in benchmark_positions.get(shingle, ()):
            starts.add(max(0, document_index - benchmark_index))
    return tuple(sorted(starts))


def _best_local_near_match(
    document: str,
    benchmark: str,
) -> tuple[int, int, int, float, float] | None:
    benchmark_shingles = scalar_shingles(benchmark)
    if len(benchmark_shingles) < CONTAMINATION_MINIMUM_SHARED_SHINGLES:
        return None
    minimum_length = max(
        CONTAMINATION_SHINGLE_SCALARS,
        math.floor(len(benchmark) * CONTAMINATION_MINIMUM_LOCAL_LENGTH_RATIO),
    )
    maximum_length = max(
        minimum_length,
        math.ceil(len(benchmark) * CONTAMINATION_MAXIMUM_LOCAL_LENGTH_RATIO),
    )
    best: tuple[int, int, int, float, float] | None = None
    for start in _candidate_local_starts(document, benchmark):
        for local_length in range(minimum_length, maximum_length + 1):
            end = start + local_length
            if end > len(document):
                break
            local_shingles = scalar_shingles(document[start:end])
            shared = len(benchmark_shingles & local_shingles)
            if shared < CONTAMINATION_MINIMUM_SHARED_SHINGLES:
                continue
            union = len(benchmark_shingles | local_shingles)
            jaccard = shared / union if union else 0.0
            coverage = shared / len(benchmark_shingles)
            if best is None or (coverage, jaccard, shared, -start, -end) > (
                best[4],
                best[3],
                best[2],
                -best[0],
                -best[1],
            ):
                best = (start, end, shared, jaccard, coverage)
    return best


def compare_document_to_benchmark(
    document_text: str,
    benchmark_text: str,
    *,
    benchmark_id: str,
) -> ContaminationMatch | None:
    """Return an exact or verified local near match without retaining text."""

    document = canonicalize_contamination_text(document_text)
    benchmark = canonicalize_contamination_text(benchmark_text)
    fingerprint = benchmark_fingerprint(benchmark_id, benchmark)
    if not document or not fingerprint.eligible_for_exact:
        return None
    exact_start = document.find(benchmark)
    if exact_start >= 0:
        benchmark_shingle_count = len(scalar_shingles(benchmark))
        return ContaminationMatch(
            benchmark_id=benchmark_id,
            benchmark_canonical_sha256=fingerprint.canonical_sha256,
            match_type="exact_local_containment",
            document_start_scalar=exact_start,
            document_end_scalar=exact_start + len(benchmark),
            benchmark_scalars=len(benchmark),
            local_scalars=len(benchmark),
            shared_shingles=benchmark_shingle_count,
            benchmark_shingle_coverage=1.0,
            shingle_jaccard=1.0,
        )
    if not fingerprint.eligible_for_near:
        return None
    best = _best_local_near_match(document, benchmark)
    if best is None:
        return None
    start, end, shared, jaccard, coverage = best
    if coverage < CONTAMINATION_BENCHMARK_COVERAGE_THRESHOLD:
        return None
    return ContaminationMatch(
        benchmark_id=benchmark_id,
        benchmark_canonical_sha256=fingerprint.canonical_sha256,
        match_type="near_local_shingle",
        document_start_scalar=start,
        document_end_scalar=end,
        benchmark_scalars=len(benchmark),
        local_scalars=end - start,
        shared_shingles=shared,
        benchmark_shingle_coverage=coverage,
        shingle_jaccard=jaccard,
    )


def scan_document_reference(
    document_text: str,
    benchmarks: Mapping[str, str],
) -> tuple[ContaminationMatch, ...]:
    """Correctness reference for an indexed full-corpus implementation."""

    if not benchmarks:
        raise ValueError("contamination scan requires benchmark inputs")
    matches = [
        match
        for benchmark_id, benchmark_text in sorted(benchmarks.items())
        if (
            match := compare_document_to_benchmark(
                document_text,
                benchmark_text,
                benchmark_id=benchmark_id,
            )
        )
        is not None
    ]
    return tuple(matches)


class IndexedContaminationDetector:
    """Reference-complete candidate retrieval without exposing benchmark text."""

    def __init__(self, benchmarks: Mapping[str, str]) -> None:
        if not benchmarks or any(not key for key in benchmarks):
            raise ValueError("contamination index requires benchmark identities")
        canonical_by_id: dict[str, str] = {}
        fingerprints: dict[str, BenchmarkFingerprint] = {}
        shingles_by_id: dict[str, frozenset[str]] = {}
        shingle_frequency: Counter[str] = Counter()
        for benchmark_id, text in sorted(benchmarks.items()):
            canonical = canonicalize_contamination_text(text)
            fingerprint = benchmark_fingerprint(benchmark_id, canonical)
            shingles = scalar_shingles(canonical)
            canonical_by_id[benchmark_id] = canonical
            fingerprints[benchmark_id] = fingerprint
            shingles_by_id[benchmark_id] = shingles
            shingle_frequency.update(shingles)

        near_index: dict[str, list[str]] = defaultdict(list)
        exact_anchor_index: dict[str, list[str]] = defaultdict(list)
        for benchmark_id in sorted(canonical_by_id):
            fingerprint = fingerprints[benchmark_id]
            shingles = shingles_by_id[benchmark_id]
            if fingerprint.eligible_for_near:
                for shingle in shingles:
                    near_index[shingle].append(benchmark_id)
            if fingerprint.eligible_for_exact:
                if not shingles:
                    raise AssertionError("eligible exact benchmark has no shingle")
                anchor = min(
                    shingles,
                    key=lambda value: (shingle_frequency[value], value),
                )
                exact_anchor_index[anchor].append(benchmark_id)

        manifest_lines = [
            "\x1f".join(
                (
                    benchmark_id,
                    fingerprints[benchmark_id].canonical_sha256,
                    str(fingerprints[benchmark_id].canonical_scalars),
                    str(fingerprints[benchmark_id].informative_scalars),
                )
            )
            for benchmark_id in sorted(fingerprints)
        ]
        self._canonical_by_id = canonical_by_id
        self._fingerprints = fingerprints
        self._near_index = {
            shingle: tuple(sorted(identities))
            for shingle, identities in near_index.items()
        }
        self._exact_anchor_index = {
            shingle: tuple(sorted(identities))
            for shingle, identities in exact_anchor_index.items()
        }
        self._manifest_sha256 = hashlib.sha256(
            "\n".join(manifest_lines).encode("utf-8")
        ).hexdigest()

    def public_metadata(self) -> dict[str, str | int]:
        """Return only content-free index provenance."""

        return {
            "index_version": CONTAMINATION_INDEX_VERSION,
            "benchmark_count": len(self._canonical_by_id),
            "exact_eligible_count": sum(
                value.eligible_for_exact for value in self._fingerprints.values()
            ),
            "near_eligible_count": sum(
                value.eligible_for_near for value in self._fingerprints.values()
            ),
            "exact_anchor_count": len(self._exact_anchor_index),
            "near_shingle_count": len(self._near_index),
            "benchmark_manifest_sha256": self._manifest_sha256,
        }

    def scan_document(
        self,
        document_text: str,
    ) -> tuple[ContaminationMatch, ...]:
        """Retrieve a superset, then verify every match with the reference."""

        document = canonicalize_contamination_text(document_text)
        if not document:
            return ()
        document_shingles = scalar_shingles(document)
        exact_candidates: set[str] = set()
        shared_counts: Counter[str] = Counter()
        for shingle in document_shingles:
            exact_candidates.update(self._exact_anchor_index.get(shingle, ()))
            shared_counts.update(self._near_index.get(shingle, ()))
        near_candidates = {
            benchmark_id
            for benchmark_id, count in shared_counts.items()
            if count >= CONTAMINATION_MINIMUM_SHARED_SHINGLES
        }
        candidates = exact_candidates | near_candidates
        matches = [
            match
            for benchmark_id in sorted(candidates)
            if (
                match := compare_document_to_benchmark(
                    document,
                    self._canonical_by_id[benchmark_id],
                    benchmark_id=benchmark_id,
                )
            )
            is not None
        ]
        return tuple(matches)
