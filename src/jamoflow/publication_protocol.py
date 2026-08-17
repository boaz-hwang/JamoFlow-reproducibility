"""Result-blind constants for publication-scale Korean evaluation.

This module intentionally contains no dataset loader or measured score.  It is a
machine-checkable preregistration surface shared by the later data, fine-tuning,
and actual-inference runners.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


PUBLICATION_PRETRAIN_SEEDS = (1729, 2718, 31415)
PUBLICATION_CONTEXT_BYTES = 512
PUBLICATION_BPE_VOCABULARY_SIZE = 32_000
PUBLICATION_BPE_STRESS_VOCABULARY_SIZE = 16_000
PUBLICATION_BPE_VOCABULARY_CANDIDATES = (
    PUBLICATION_BPE_STRESS_VOCABULARY_SIZE,
    PUBLICATION_BPE_VOCABULARY_SIZE,
)
PUBLICATION_CANDIDATE_MODEL_KEY = "candidate"
PUBLICATION_RAW_COMPARATOR_MODEL_KEY = "raw_byte_reference"
PUBLICATION_BPE_COMPARATOR_MODEL_KEYS: Mapping[int, str] = MappingProxyType(
    {
        PUBLICATION_BPE_STRESS_VOCABULARY_SIZE: (
            "byte_bpe_16000_body_matched"
        ),
        PUBLICATION_BPE_VOCABULARY_SIZE: "byte_bpe_32000",
    }
)
PUBLICATION_DOWNSTREAM_REFERENCE_KEYS = (
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
    *(
        PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[size]
        for size in PUBLICATION_BPE_VOCABULARY_CANDIDATES
    ),
)
PUBLICATION_BPE_INITIAL_ALPHABET_SIZE = 256
PUBLICATION_BPE_NORMALIZATION = "none_nfc_source_identity"
PUBLICATION_BPE_PRETOKENIZATION = "gpt2_byte_level_no_prefix_space"

DOWNSTREAM_FAMILY_NONINFERIORITY_MARGIN_PP = 2.0
DOWNSTREAM_TASK_GUARD_MARGIN_PP = 5.0
DOWNSTREAM_REFERENCE_FLOOR_ADVANTAGE_PP = 5.0
DOWNSTREAM_FAMILY_ONE_SIDED_CONFIDENCE = 0.975
DOWNSTREAM_MINIMUM_KOBEST_INFORMATIVE_TASKS = 3
DOWNSTREAM_REQUIRED_KLUE_INFORMATIVE_TASKS = 2
DOWNSTREAM_REFERENCE_TIE_PP = 0.5

PUBLICATION_BPB_NONINFERIORITY_MARGIN = 0.010
PUBLICATION_BPB_ONE_SIDED_CONFIDENCE = 0.975
PUBLICATION_BPB_CONTEXT_BYTES = PUBLICATION_CONTEXT_BYTES
PUBLICATION_BPB_TARGET_BLOCK_BYTES = 256
PUBLICATION_BPB_CONTEXT_CONTRACT = (
    "pairwise_utf8_complete_natural_unit_raw_capped_rolling_v2"
)
PUBLICATION_BPB_UNSCORED_PREFIX_POLICY = (
    "exclude_first_utf8_complete_comparator_group_from_both_models"
)
ACTUAL_INFERENCE_MINIMUM_REDUCTION = 0.10
ACTUAL_INFERENCE_BOOTSTRAP_REPETITIONS = 10_000
ACTUAL_INFERENCE_BOOTSTRAP_SEED = 20_260_811
ACTUAL_INFERENCE_MINIMUM_VALID_OUTPUT_BYTES = 128
ACTUAL_INFERENCE_VALID_OUTPUT_CONSTRAINT = (
    "shared_strict_rfc3629_transition_mask_no_horizon_closure"
)
ACTUAL_INFERENCE_BYTE_MAXIMUM_OVERSHOOT = 3
ACTUAL_INFERENCE_FREE_RUNNING_MAXIMUM_STEPS = 512
PUBLICATION_RUNTIME_PROTOCOL_VERSION = 3
PUBLICATION_RUNTIME_PROMPT_BYTES = 128
PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES = 128
PUBLICATION_RUNTIME_WARMUP_CASES = 8
PUBLICATION_RUNTIME_MEASURED_CASES = 64
PUBLICATION_RUNTIME_REPETITIONS = 5
PUBLICATION_RUNTIME_TIMING_ORDER_SEED = 20_260_811
PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED = 20_260_814
PUBLICATION_RUNTIME_EQUIVALENCE_MINIMUM_VECTORS_PER_PAIR = 16
PUBLICATION_RUNTIME_MODES = (
    "controlled_replay",
    "free_running_utf8_greedy",
)
PUBLICATION_RUNTIME_COMPONENTS = ("ttft_ms", "decode_ms", "end_to_end_ms")
PUBLICATION_RUNTIME_ROLES = ("candidate", "reference")
PUBLICATION_RUNTIME_EQUIVALENCE_PATHS = (
    "full_prefix_incremental",
    "parallel_prefill_continuation",
)
PUBLICATION_RUNTIME_TIMING_SCOPE_CONTRACT = (
    "batch1_fresh_runtime_parallel_prefill_selector_router_cache_utf8_mask_"
    "argmax_transition_stop_and_device_sync_inside_timing_router_counters_v2"
)


@dataclass(frozen=True, slots=True)
class DatasetPin:
    key: str
    repository: str
    revision: str
    license_spdx: str | None
    use: str
    release_policy: str


@dataclass(frozen=True, slots=True)
class DownstreamTaskSpec:
    key: str
    dataset_key: str
    config: str
    family: str
    labels: tuple[str, ...]
    fit_split: str
    selection_split: str
    sealed_evaluation_split: str
    primary_metric: str
    role: str

    @property
    def label_count(self) -> int:
        return len(self.labels)


DATASET_PINS: Mapping[str, DatasetPin] = MappingProxyType(
    {
        "klue": DatasetPin(
            key="klue",
            repository="klue/klue",
            revision="349481ec73fff722f88e0453ca05c77a447d967c",
            license_spdx="CC-BY-SA-4.0",
            use="primary_supervised_downstream",
            release_policy="manifest_and_code_only_no_repacked_rows",
        ),
        "kobest": DatasetPin(
            key="kobest",
            repository="skt/kobest_v1",
            revision="a5ea15e3ac77ed694b79f6204eb31889a2ba989f",
            license_spdx="CC-BY-SA-4.0",
            use="primary_supervised_downstream",
            release_policy="manifest_and_code_only_no_repacked_rows",
        ),
        "kmmlu": DatasetPin(
            key="kmmlu",
            repository="HAERAE-HUB/KMMLU",
            revision="d61b3f19e552c576bf5960dd24289763edc36a88",
            license_spdx="CC-BY-ND-4.0",
            use="secondary_floor_gated_knowledge_diagnostic",
            release_policy="code_and_aggregate_metrics_only_no_derived_data",
        ),
        "haerae_bench": DatasetPin(
            key="haerae_bench",
            repository="HAERAE-HUB/HAE_RAE_BENCH_1.0",
            revision="d5082e9b46bdd7012471d60ee1851e734606af72",
            license_spdx=None,
            use="excluded_until_license_is_clarified",
            release_policy="no_downloaded_or_derived_artifact_release",
        ),
    }
)


PRIMARY_DOWNSTREAM_TASKS: Mapping[str, DownstreamTaskSpec] = MappingProxyType(
    {
        "kobest_boolq": DownstreamTaskSpec(
            key="kobest_boolq",
            dataset_key="kobest",
            config="boolq",
            family="kobest",
            labels=("0", "1"),
            fit_split="train",
            selection_split="validation",
            sealed_evaluation_split="test",
            primary_metric="macro_f1",
            role="primary",
        ),
        "kobest_copa": DownstreamTaskSpec(
            key="kobest_copa",
            dataset_key="kobest",
            config="copa",
            family="kobest",
            labels=("0", "1"),
            fit_split="train",
            selection_split="validation",
            sealed_evaluation_split="test",
            primary_metric="macro_f1",
            role="primary",
        ),
        "kobest_wic": DownstreamTaskSpec(
            key="kobest_wic",
            dataset_key="kobest",
            config="wic",
            family="kobest",
            labels=("0", "1"),
            fit_split="train",
            selection_split="validation",
            sealed_evaluation_split="test",
            primary_metric="macro_f1",
            role="primary",
        ),
        "kobest_sentineg": DownstreamTaskSpec(
            key="kobest_sentineg",
            dataset_key="kobest",
            config="sentineg",
            family="kobest",
            labels=("0", "1"),
            fit_split="train",
            selection_split="validation",
            sealed_evaluation_split="test",
            primary_metric="macro_f1",
            role="primary_robustness",
        ),
        "klue_ynat": DownstreamTaskSpec(
            key="klue_ynat",
            dataset_key="klue",
            config="ynat",
            family="klue",
            labels=("0", "1", "2", "3", "4", "5", "6"),
            fit_split="official_train_minus_internal_dev",
            selection_split="label_stratified_hash_10pct_of_official_train",
            sealed_evaluation_split="validation",
            primary_metric="macro_f1",
            role="primary",
        ),
        "klue_nli": DownstreamTaskSpec(
            key="klue_nli",
            dataset_key="klue",
            config="nli",
            family="klue",
            labels=("0", "1", "2"),
            fit_split="official_train_minus_internal_dev",
            selection_split="label_stratified_hash_10pct_of_official_train",
            sealed_evaluation_split="validation",
            primary_metric="accuracy",
            role="primary",
        ),
    }
)


SECONDARY_DOWNSTREAM_TASKS: Mapping[str, DownstreamTaskSpec] = MappingProxyType(
    {
        "kobest_hellaswag": DownstreamTaskSpec(
            key="kobest_hellaswag",
            dataset_key="kobest",
            config="hellaswag",
            family="kobest",
            labels=("0", "1", "2", "3"),
            fit_split="train",
            selection_split="validation",
            sealed_evaluation_split="test",
            primary_metric="macro_f1",
            role="secondary_after_split_and_truncation_audit",
        ),
    }
)


def choose_validation_reference(
    validation_scores: Mapping[str, tuple[float, ...]],
    *,
    tie_pp: float = DOWNSTREAM_REFERENCE_TIE_PP,
) -> str:
    """Choose a task reference before opening its sealed evaluation split.

    Scores are fractions in [0, 1].  A deployment-default BPE reference wins a
    near tie so a slower raw-byte reference cannot be selected opportunistically.
    """

    bpe_key = PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[
        PUBLICATION_BPE_VOCABULARY_SIZE
    ]
    if (
        set(validation_scores) != set(PUBLICATION_DOWNSTREAM_REFERENCE_KEYS)
        or not 0 <= tie_pp <= 100
    ):
        raise ValueError(
            "reference selection requires exact raw, 16K, and 32K scores"
        )
    if any(not scores for scores in validation_scores.values()):
        raise ValueError("every reference requires validation scores")
    lengths = {len(scores) for scores in validation_scores.values()}
    if lengths != {len(PUBLICATION_PRETRAIN_SEEDS)}:
        raise ValueError("reference scores must follow the three paired seeds")
    means: dict[str, float] = {}
    for key, scores in validation_scores.items():
        if not key or any(not 0.0 <= score <= 1.0 for score in scores):
            raise ValueError("reference scores must be finite fractions")
        means[key] = sum(scores) / len(scores)
    best_key = max(sorted(means), key=lambda key: means[key])
    if means[best_key] - means[bpe_key] <= tie_pp / 100.0:
        return bpe_key
    return best_key


def validate_publication_protocol() -> None:
    """Reject accidental drift in the preregistered manifest."""

    if len(set(PUBLICATION_PRETRAIN_SEEDS)) != 3:
        raise ValueError("publication design requires three unique paired seeds")
    if PUBLICATION_CONTEXT_BYTES != 512:
        raise ValueError("publication context-byte budget drifted")
    if PUBLICATION_BPE_INITIAL_ALPHABET_SIZE != 256:
        raise ValueError("byte BPE must retain the complete byte alphabet")
    if (
        PUBLICATION_BPE_VOCABULARY_CANDIDATES
        != (16_000, 32_000)
        or len(set(PUBLICATION_BPE_VOCABULARY_CANDIDATES)) != 2
        or PUBLICATION_BPE_VOCABULARY_SIZE
        != max(PUBLICATION_BPE_VOCABULARY_CANDIDATES)
    ):
        raise ValueError("publication BPE vocabulary controls drifted")
    if (
        set(PUBLICATION_BPE_COMPARATOR_MODEL_KEYS)
        != set(PUBLICATION_BPE_VOCABULARY_CANDIDATES)
        or PUBLICATION_DOWNSTREAM_REFERENCE_KEYS
        != (
            PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
        )
        or len(set(PUBLICATION_DOWNSTREAM_REFERENCE_KEYS)) != 3
        or PUBLICATION_CANDIDATE_MODEL_KEY
        in PUBLICATION_DOWNSTREAM_REFERENCE_KEYS
    ):
        raise ValueError("publication model-family identities drifted")
    if not 0 < PUBLICATION_BPB_NONINFERIORITY_MARGIN < 1:
        raise ValueError("publication BPB margin is invalid")
    if not 0.5 < PUBLICATION_BPB_ONE_SIDED_CONFIDENCE < 1:
        raise ValueError("publication BPB confidence is invalid")
    if (
        PUBLICATION_BPB_CONTEXT_BYTES != PUBLICATION_CONTEXT_BYTES
        or PUBLICATION_BPB_TARGET_BLOCK_BYTES != 256
        or PUBLICATION_BPB_TARGET_BLOCK_BYTES * 2
        > PUBLICATION_BPB_CONTEXT_BYTES
        or PUBLICATION_BPB_CONTEXT_CONTRACT
        != "pairwise_utf8_complete_natural_unit_raw_capped_rolling_v2"
        or PUBLICATION_BPB_UNSCORED_PREFIX_POLICY
        != "exclude_first_utf8_complete_comparator_group_from_both_models"
    ):
        raise ValueError("publication BPB raw-context contract drifted")
    if not 0 < ACTUAL_INFERENCE_MINIMUM_REDUCTION < 1:
        raise ValueError("actual-inference reduction must be a fraction")
    if (
        ACTUAL_INFERENCE_MINIMUM_VALID_OUTPUT_BYTES != 128
        or ACTUAL_INFERENCE_BYTE_MAXIMUM_OVERSHOOT != 3
        or ACTUAL_INFERENCE_VALID_OUTPUT_CONSTRAINT
        != "shared_strict_rfc3629_transition_mask_no_horizon_closure"
        or ACTUAL_INFERENCE_FREE_RUNNING_MAXIMUM_STEPS
        < ACTUAL_INFERENCE_MINIMUM_VALID_OUTPUT_BYTES
    ):
        raise ValueError("publication valid-output contract drifted")
    if (
        PUBLICATION_RUNTIME_PROTOCOL_VERSION != 3
        or PUBLICATION_RUNTIME_PROMPT_BYTES != 128
        or PUBLICATION_RUNTIME_MINIMUM_OUTPUT_BYTES != 128
        or PUBLICATION_RUNTIME_WARMUP_CASES != 8
        or PUBLICATION_RUNTIME_MEASURED_CASES != 64
        or PUBLICATION_RUNTIME_REPETITIONS != 5
        or PUBLICATION_RUNTIME_SEED_EXECUTION_ORDER_SEED != 20_260_814
        or PUBLICATION_RUNTIME_EQUIVALENCE_MINIMUM_VECTORS_PER_PAIR != 16
        or PUBLICATION_RUNTIME_MODES
        != ("controlled_replay", "free_running_utf8_greedy")
        or PUBLICATION_RUNTIME_COMPONENTS
        != ("ttft_ms", "decode_ms", "end_to_end_ms")
        or PUBLICATION_RUNTIME_ROLES != ("candidate", "reference")
        or PUBLICATION_RUNTIME_EQUIVALENCE_PATHS
        != ("full_prefix_incremental", "parallel_prefill_continuation")
        or "device_sync_inside_timing"
        not in PUBLICATION_RUNTIME_TIMING_SCOPE_CONTRACT
    ):
        raise ValueError("publication runtime evidence contract drifted")
    for key, pin in DATASET_PINS.items():
        if (
            pin.key != key
            or len(pin.revision) != 40
            or any(character not in "0123456789abcdef" for character in pin.revision)
        ):
            raise ValueError("dataset manifest identity or revision is invalid")
    if DATASET_PINS["haerae_bench"].license_spdx is not None:
        raise ValueError("HAE-RAE Bench remains excluded pending license clarity")
    expected_primary = {
        "kobest_boolq",
        "kobest_copa",
        "kobest_wic",
        "kobest_sentineg",
        "klue_ynat",
        "klue_nli",
    }
    if set(PRIMARY_DOWNSTREAM_TASKS) != expected_primary:
        raise ValueError("primary downstream suite drifted")
    for key, task in PRIMARY_DOWNSTREAM_TASKS.items():
        if (
            task.key != key
            or task.dataset_key not in DATASET_PINS
            or not task.labels
            or task.labels != tuple(str(index) for index in range(task.label_count))
            or any(len(label.encode("ascii")) != 1 for label in task.labels)
        ):
            raise ValueError("downstream task or ASCII label mapping is invalid")


validate_publication_protocol()
