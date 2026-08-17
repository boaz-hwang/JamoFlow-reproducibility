"""Pinned Korean downstream prompts, truncation, and KLUE internal split."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import operator
import unicodedata
from typing import Callable, Mapping, Sequence

from .publication_protocol import DATASET_PINS, PRIMARY_DOWNSTREAM_TASKS


DOWNSTREAM_PROMPT_VERSION = "publication-v1-20260811"
DOWNSTREAM_MAXIMUM_PROMPT_BYTES = 511
KLUE_INTERNAL_SELECTION_FRACTION = 0.10
DOWNSTREAM_LABEL_CONDITIONING = (
    "separate_prompt_and_label_no_cross_boundary_merge"
)


@dataclass(frozen=True, slots=True)
class PromptRender:
    task_key: str
    prompt: str
    label_digit: str
    prompt_version: str
    original_prompt_bytes: int
    rendered_prompt_bytes: int
    truncated: bool
    removed_scalars_by_field: tuple[tuple[str, int], ...]
    normalization_changed_fields: tuple[str, ...]

    def metadata(self) -> dict[str, object]:
        """Return content-free provenance safe for tracked aggregates."""

        return {
            "task_key": self.task_key,
            "prompt_version": self.prompt_version,
            "original_prompt_bytes": self.original_prompt_bytes,
            "rendered_prompt_bytes": self.rendered_prompt_bytes,
            "truncated": self.truncated,
            "removed_scalars_by_field": list(self.removed_scalars_by_field),
            "normalization_changed_fields": list(
                self.normalization_changed_fields
            ),
            "label_bytes": len(self.label_digit.encode("ascii")),
        }


@dataclass(frozen=True, slots=True)
class ConditionalUnitEncoding:
    """Ephemeral model units for a boundary-safe generative classifier."""

    task_key: str
    prompt_units: tuple[int, ...]
    allowed_label_units: tuple[int, ...]
    gold_label_unit: int
    joint_encoding_matches_separate: bool
    conditioning_contract: str

    def metadata(self) -> dict[str, object]:
        """Exclude reconstructable prompt/token IDs from tracked metadata."""

        return {
            "task_key": self.task_key,
            "prompt_unit_count": len(self.prompt_units),
            "allowed_label_count": len(self.allowed_label_units),
            "gold_label_unit_count": 1,
            "joint_boundary_merge_observed": (
                not self.joint_encoding_matches_separate
            ),
            "conditioning_contract": self.conditioning_contract,
        }


@dataclass(frozen=True, slots=True)
class StratifiedHashSplit:
    task_key: str
    fit_row_ids: tuple[str, ...]
    selection_row_ids: tuple[str, ...]
    counts_by_label: tuple[tuple[int, int, int], ...]
    selection_fraction: float
    assignment_sha256: str

    def metadata(self) -> dict[str, object]:
        return {
            "task_key": self.task_key,
            "fit_count": len(self.fit_row_ids),
            "selection_count": len(self.selection_row_ids),
            "counts_by_label": [
                {"label": label, "fit": fit, "selection": selection}
                for label, fit, selection in self.counts_by_label
            ],
            "selection_fraction": self.selection_fraction,
            "assignment_sha256": self.assignment_sha256,
        }


_REQUIRED_FIELDS = {
    "kobest_boolq": ("paragraph", "question"),
    "kobest_copa": ("premise", "question", "alternative_1", "alternative_2"),
    "kobest_wic": ("word", "context_1", "context_2"),
    "kobest_sentineg": ("sentence",),
    "klue_ynat": ("title",),
    "klue_nli": ("premise", "hypothesis"),
}

_TRUNCATABLE_FIELDS = {
    "kobest_boolq": (("paragraph", 16),),
    "kobest_copa": (("premise", 8),),
    "kobest_wic": (("context_1", 8), ("context_2", 8)),
    "kobest_sentineg": (("sentence", 4),),
    "klue_ynat": (("title", 4),),
    "klue_nli": (("premise", 8), ("hypothesis", 8)),
}

_LABEL_NAMES = {
    "kobest_boolq": ("아니오", "예"),
    "kobest_copa": ("첫 번째 선택지", "두 번째 선택지"),
    "kobest_wic": ("다른 의미", "같은 의미"),
    "kobest_sentineg": ("부정", "긍정"),
    "klue_ynat": ("IT과학", "경제", "사회", "생활문화", "세계", "스포츠", "정치"),
    "klue_nli": ("함의", "중립", "모순"),
}


def _render_fields(task_key: str, fields: Mapping[str, str]) -> str:
    if task_key == "kobest_boolq":
        return (
            f"[문단]\n{fields['paragraph']}\n"
            f"[질문]\n{fields['question']}\n"
            "[선택지]\n0: 아니오\n1: 예\n[정답]\n"
        )
    if task_key == "kobest_copa":
        return (
            f"[전제]\n{fields['premise']}\n"
            f"[질문 유형]\n{fields['question']}\n"
            f"[선택지]\n0: {fields['alternative_1']}\n"
            f"1: {fields['alternative_2']}\n[정답]\n"
        )
    if task_key == "kobest_wic":
        return (
            f"[대상어]\n{fields['word']}\n"
            f"[문장 1]\n{fields['context_1']}\n"
            f"[문장 2]\n{fields['context_2']}\n"
            "[질문]\n두 문장에서 대상어의 의미가 같은가?\n"
            "[선택지]\n0: 다른 의미\n1: 같은 의미\n[정답]\n"
        )
    if task_key == "kobest_sentineg":
        return (
            f"[문장]\n{fields['sentence']}\n"
            "[감성]\n0: 부정\n1: 긍정\n[정답]\n"
        )
    if task_key == "klue_ynat":
        return (
            f"[뉴스 제목]\n{fields['title']}\n"
            "[주제]\n0: IT과학\n1: 경제\n2: 사회\n3: 생활문화\n"
            "4: 세계\n5: 스포츠\n6: 정치\n[정답]\n"
        )
    if task_key == "klue_nli":
        return (
            f"[전제]\n{fields['premise']}\n"
            f"[가설]\n{fields['hypothesis']}\n"
            "[관계]\n0: 함의\n1: 중립\n2: 모순\n[정답]\n"
        )
    raise ValueError("unknown downstream prompt task")


def _normalized_fields(
    task_key: str,
    row: Mapping[str, object],
) -> tuple[dict[str, str], tuple[str, ...]]:
    if task_key not in _REQUIRED_FIELDS:
        raise ValueError("unknown downstream task")
    fields: dict[str, str] = {}
    changed: list[str] = []
    for field in _REQUIRED_FIELDS[task_key]:
        if field not in row or not isinstance(row[field], str) or not row[field]:
            raise ValueError(f"missing non-empty downstream field: {field}")
        source = row[field]
        assert isinstance(source, str)
        normalized = unicodedata.normalize("NFC", source)
        fields[field] = normalized
        if normalized != source:
            changed.append(field)
    return fields, tuple(changed)


def render_downstream_prompt(
    task_key: str,
    row: Mapping[str, object],
    *,
    maximum_prompt_bytes: int = DOWNSTREAM_MAXIMUM_PROMPT_BYTES,
) -> PromptRender:
    """Render one architecture-neutral prompt and truncate only context tails."""

    if maximum_prompt_bytes <= 0 or maximum_prompt_bytes > 511:
        raise ValueError("downstream prompt byte cap must lie in [1, 511]")
    spec = PRIMARY_DOWNSTREAM_TASKS.get(task_key)
    if spec is None:
        raise ValueError("prompt task is not in the primary suite")
    try:
        label = operator.index(row["label"])
    except (KeyError, TypeError) as error:
        raise ValueError("downstream row requires an integer label") from error
    if label not in range(spec.label_count):
        raise ValueError("downstream label is outside the task label set")
    fields, normalization_changed = _normalized_fields(task_key, row)
    prompt = _render_fields(task_key, fields)
    original_bytes = len(prompt.encode("utf-8"))
    removed = {field: 0 for field, _ in _TRUNCATABLE_FIELDS[task_key]}
    minimum_by_field = dict(_TRUNCATABLE_FIELDS[task_key])
    field_order = tuple(minimum_by_field)
    while len(prompt.encode("utf-8")) > maximum_prompt_bytes:
        candidates = [
            field
            for field in field_order
            if len(fields[field]) > minimum_by_field[field]
        ]
        if not candidates:
            raise ValueError("fixed prompt and preserved fields exceed the byte cap")
        chosen = max(
            candidates,
            key=lambda field: (
                len(fields[field].encode("utf-8")),
                -field_order.index(field),
            ),
        )
        fields[chosen] = fields[chosen][:-1]
        removed[chosen] += 1
        prompt = _render_fields(task_key, fields)
    rendered_bytes = len(prompt.encode("utf-8"))
    if len((prompt + str(label)).encode("utf-8")) > 512:
        raise RuntimeError("prompt plus one-byte answer exceeds model context")
    return PromptRender(
        task_key=task_key,
        prompt=prompt,
        label_digit=str(label),
        prompt_version=DOWNSTREAM_PROMPT_VERSION,
        original_prompt_bytes=original_bytes,
        rendered_prompt_bytes=rendered_bytes,
        truncated=any(removed.values()),
        removed_scalars_by_field=tuple(
            (field, removed[field]) for field in field_order
        ),
        normalization_changed_fields=normalization_changed,
    )


def benchmark_input_text(task_key: str, row: Mapping[str, object]) -> str:
    """Return label- and instruction-free fields for contamination detection."""

    fields, _ = _normalized_fields(task_key, row)
    return "\n".join(fields[field] for field in _REQUIRED_FIELDS[task_key])


def _unit_ids(
    encode: Callable[[str], Sequence[int]],
    text: str,
) -> tuple[int, ...]:
    values = encode(text)
    if isinstance(values, str):
        raise ValueError("downstream encoder must return integer model units")
    try:
        raw_units = tuple(values)
        units = tuple(operator.index(value) for value in raw_units)
    except (TypeError, ValueError) as error:
        raise ValueError("downstream encoder returned malformed units") from error
    if any(isinstance(value, bool) for value in raw_units) or any(
        value < 0 for value in units
    ):
        raise ValueError("downstream model units must be nonnegative integers")
    return units


def encode_downstream_conditioning(
    rendered: PromptRender,
    encode: Callable[[str], Sequence[int]],
) -> ConditionalUnitEncoding:
    """Encode prompt and every digit separately, never across the answer boundary."""

    spec = PRIMARY_DOWNSTREAM_TASKS.get(rendered.task_key)
    if (
        spec is None
        or rendered.prompt_version != DOWNSTREAM_PROMPT_VERSION
        or rendered.label_digit not in spec.labels
        or not rendered.prompt
    ):
        raise ValueError("downstream rendered example identity is invalid")
    prompt_units = _unit_ids(encode, rendered.prompt)
    labels = tuple(_unit_ids(encode, label) for label in spec.labels)
    if (
        not prompt_units
        or any(len(label) != 1 for label in labels)
        or len({label[0] for label in labels}) != spec.label_count
        or len(prompt_units) + 1 > 512
    ):
        raise ValueError(
            "downstream labels must be distinct single units inside context"
        )
    gold_index = int(rendered.label_digit)
    gold_unit = labels[gold_index][0]
    joint = _unit_ids(encode, rendered.prompt + rendered.label_digit)
    separate = prompt_units + (gold_unit,)
    return ConditionalUnitEncoding(
        task_key=rendered.task_key,
        prompt_units=prompt_units,
        allowed_label_units=tuple(label[0] for label in labels),
        gold_label_unit=gold_unit,
        joint_encoding_matches_separate=joint == separate,
        conditioning_contract=DOWNSTREAM_LABEL_CONDITIONING,
    )


def klue_internal_hash_split(
    task_key: str,
    rows: Sequence[Mapping[str, object]],
    *,
    selection_fraction: float = KLUE_INTERNAL_SELECTION_FRACTION,
) -> StratifiedHashSplit:
    """Make the preregistered label-stratified, row-order-invariant split."""

    if task_key not in {"klue_ynat", "klue_nli"}:
        raise ValueError("internal hash split is only defined for KLUE primary tasks")
    if not 0 < selection_fraction < 0.5 or not rows:
        raise ValueError("invalid internal selection fraction or empty rows")
    spec = PRIMARY_DOWNSTREAM_TASKS[task_key]
    revision = DATASET_PINS["klue"].revision
    grouped: dict[int, list[tuple[str, str]]] = {
        label: [] for label in range(spec.label_count)
    }
    seen_ids: set[str] = set()
    seen_digests: set[str] = set()
    for row in rows:
        row_id = row.get("guid")
        try:
            label = operator.index(row["label"])
        except (KeyError, TypeError) as error:
            raise ValueError("KLUE split row requires integer label") from error
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in seen_ids
            or label not in grouped
        ):
            raise ValueError("KLUE row identity or label is invalid")
        canonical_input = benchmark_input_text(task_key, row)
        payload = "\x1f".join((revision, task_key, row_id, canonical_input))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if digest in seen_digests:
            raise ValueError("KLUE split digest collision")
        seen_ids.add(row_id)
        seen_digests.add(digest)
        grouped[label].append((digest, row_id))
    if any(len(values) < 2 for values in grouped.values()):
        raise ValueError("every KLUE label needs at least two rows")

    fit: list[str] = []
    selection: list[str] = []
    assignments: list[tuple[str, str, str]] = []
    counts: list[tuple[int, int, int]] = []
    for label in range(spec.label_count):
        ordered = sorted(grouped[label])
        selection_count = math.ceil(len(ordered) * selection_fraction)
        if selection_count >= len(ordered):
            raise ValueError("internal selection consumes a full label stratum")
        selected = ordered[:selection_count]
        fitted = ordered[selection_count:]
        selection.extend(row_id for _, row_id in selected)
        fit.extend(row_id for _, row_id in fitted)
        assignments.extend((row_id, "selection", digest) for digest, row_id in selected)
        assignments.extend((row_id, "fit", digest) for digest, row_id in fitted)
        counts.append((label, len(fitted), len(selected)))
    assignment_payload = "\n".join(
        "\x1f".join(values) for values in sorted(assignments)
    )
    return StratifiedHashSplit(
        task_key=task_key,
        fit_row_ids=tuple(sorted(fit)),
        selection_row_ids=tuple(sorted(selection)),
        counts_by_label=tuple(counts),
        selection_fraction=selection_fraction,
        assignment_sha256=hashlib.sha256(
            assignment_payload.encode("utf-8")
        ).hexdigest(),
    )


def validate_downstream_data_protocol() -> None:
    expected = set(PRIMARY_DOWNSTREAM_TASKS)
    if (
        set(_REQUIRED_FIELDS) != expected
        or set(_TRUNCATABLE_FIELDS) != expected
        or set(_LABEL_NAMES) != expected
    ):
        raise ValueError("downstream renderer task set drifted")
    for task_key, spec in PRIMARY_DOWNSTREAM_TASKS.items():
        if (
            len(_LABEL_NAMES[task_key]) != spec.label_count
            or not _REQUIRED_FIELDS[task_key]
            or not _TRUNCATABLE_FIELDS[task_key]
            or any(minimum <= 0 for _, minimum in _TRUNCATABLE_FIELDS[task_key])
        ):
            raise ValueError("downstream renderer schema is invalid")


validate_downstream_data_protocol()
