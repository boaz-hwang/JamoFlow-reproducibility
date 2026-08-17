"""Blind Mac feasibility candidates for publication-scale replication."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping

from .neural_model import Phase1ModelSpec
from .publication_reference import (
    PUBLICATION_AUXILIARY_ENTROPY_ROUTER,
    PUBLICATION_AUXILIARY_NONE,
    PublicationRawReferenceDescriptor,
    is_sha256,
    publication_auxiliary_kind_for_policy,
    validate_publication_raw_reference_descriptor,
)
from .publication_protocol import PUBLICATION_PRETRAIN_SEEDS


PUBLICATION_SEQUENCE_LENGTH = 512
PUBLICATION_TRAIN_BYTES = 256_000_000
PUBLICATION_BATCH_SIZE = 32
PUBLICATION_EVALUATION_BATCH_SIZE = 64
PUBLICATION_SCALE_ORDER = (50, 75, 100)
PUBLICATION_SCALE_SPECS = {
    50: Phase1ModelSpec(
        sequence_length=PUBLICATION_SEQUENCE_LENGTH,
        patch_count=86,
        patch_stride=6,
        local_width=256,
        global_width=512,
        local_heads=8,
        global_heads=8,
        encoder_layers=2,
        global_layers=12,
        decoder_layers=2,
        local_ffn=768,
        global_ffn=1536,
        cross_attention_k=2,
        hash_group_size=3,
        hash_vocabulary=16384,
        router_width=256,
        router_heads=8,
        router_layers=4,
        router_ffn=768,
    ),
    75: Phase1ModelSpec(
        sequence_length=PUBLICATION_SEQUENCE_LENGTH,
        patch_count=86,
        patch_stride=6,
        local_width=320,
        global_width=640,
        local_heads=8,
        global_heads=10,
        encoder_layers=2,
        global_layers=12,
        decoder_layers=2,
        local_ffn=960,
        global_ffn=1920,
        cross_attention_k=2,
        hash_group_size=3,
        hash_vocabulary=16384,
        router_width=320,
        router_heads=8,
        router_layers=4,
        router_ffn=960,
    ),
    100: Phase1ModelSpec(
        sequence_length=PUBLICATION_SEQUENCE_LENGTH,
        patch_count=86,
        patch_stride=6,
        local_width=352,
        global_width=704,
        local_heads=11,
        global_heads=11,
        encoder_layers=2,
        global_layers=13,
        decoder_layers=2,
        local_ffn=1056,
        global_ffn=2112,
        cross_attention_k=2,
        hash_group_size=3,
        hash_vocabulary=16384,
        router_width=352,
        router_heads=11,
        router_layers=4,
        router_ffn=1056,
    ),
}
PUBLICATION_EXPECTED_PARAMETERS = {
    50: 49_823_488,
    75: 76_492_480,
    100: 98_403_360,
}
PUBLICATION_BPE_16000_EXPECTED_PARAMETERS = {
    50: 42_617_792,
    75: 66_710_368,
    100: 86_975_680,
}
PUBLICATION_BPE_32000_EXPECTED_PARAMETERS = {
    50: 49_785_792,
    75: 76_438_368,
    100: 98_239_680,
}
PUBLICATION_ROUTER_EXPECTED_PARAMETERS = {
    50: 3_541_248,
    75: 5_491_520,
    100: 6_626_400,
}
MAXIMUM_RECOMMENDED_MEMORY_FRACTION = 0.75
MAXIMUM_SAFETY_ADJUSTED_HOURS_PER_MODEL = 12.0
MAXIMUM_CORE_CAMPAIGN_HOURS = 120.0
WALL_TIME_SAFETY_FACTOR = 1.20
PUBLICATION_FEASIBILITY_MEASUREMENT_STEPS = 3
CORE_MODEL_FAMILIES = (
    "candidate",
    "raw_byte_reference",
    "byte_bpe_16000_body_matched",
    "byte_bpe_32000",
)
CORE_MODEL_RUNS = len(CORE_MODEL_FAMILIES) * len(PUBLICATION_PRETRAIN_SEEDS)
PUBLICATION_FAMILY_EXPECTED_PARAMETERS = {
    target: {
        "candidate": PUBLICATION_EXPECTED_PARAMETERS[target],
        "raw_byte_reference": PUBLICATION_EXPECTED_PARAMETERS[target],
        "byte_bpe_16000_body_matched": (
            PUBLICATION_BPE_16000_EXPECTED_PARAMETERS[target]
        ),
        "byte_bpe_32000": PUBLICATION_BPE_32000_EXPECTED_PARAMETERS[target],
    }
    for target in PUBLICATION_SCALE_ORDER
}
PUBLICATION_PROJECTED_TRAIN_STEPS = math.ceil(
    PUBLICATION_TRAIN_BYTES
    / (PUBLICATION_SEQUENCE_LENGTH * PUBLICATION_BATCH_SIZE)
)


def publication_model_spec(target_millions: int, patch_count: int) -> Phase1ModelSpec:
    if target_millions not in PUBLICATION_SCALE_SPECS:
        raise ValueError("unknown publication-scale target")
    if not 1 < patch_count <= PUBLICATION_SEQUENCE_LENGTH:
        raise ValueError("publication patch count is invalid")
    return replace(
        PUBLICATION_SCALE_SPECS[target_millions],
        patch_count=patch_count,
    )


@dataclass(frozen=True, slots=True)
class ScaleFeasibility:
    """Candidate-graph-only preflight; not a final campaign lock."""

    target_millions: int
    completed: bool
    finite_steps: bool
    parameter_count: int
    maximum_driver_allocated_bytes: int
    recommended_max_memory_bytes: int
    projected_hours_per_model: float

    @property
    def memory_fraction(self) -> float:
        if self.recommended_max_memory_bytes <= 0:
            return math.inf
        return (
            self.maximum_driver_allocated_bytes
            / self.recommended_max_memory_bytes
        )

    @property
    def safety_adjusted_hours_per_model(self) -> float:
        return self.projected_hours_per_model * WALL_TIME_SAFETY_FACTOR

    @property
    def projected_core_campaign_hours(self) -> float:
        return self.safety_adjusted_hours_per_model * CORE_MODEL_RUNS

    @property
    def passes(self) -> bool:
        return bool(
            self.completed
            and self.finite_steps
            and math.isfinite(self.projected_hours_per_model)
            and self.parameter_count
            == PUBLICATION_EXPECTED_PARAMETERS[self.target_millions]
            and self.recommended_max_memory_bytes > 0
            and self.memory_fraction <= MAXIMUM_RECOMMENDED_MEMORY_FRACTION
            and self.safety_adjusted_hours_per_model
            <= MAXIMUM_SAFETY_ADJUSTED_HOURS_PER_MODEL
            and self.projected_core_campaign_hours <= MAXIMUM_CORE_CAMPAIGN_HOURS
        )

    def to_dict(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return value if math.isfinite(value) else None

        return {
            "target_millions": self.target_millions,
            "completed": self.completed,
            "finite_steps": self.finite_steps,
            "parameter_count": self.parameter_count,
            "maximum_driver_allocated_bytes": self.maximum_driver_allocated_bytes,
            "recommended_max_memory_bytes": self.recommended_max_memory_bytes,
            "memory_fraction": finite_or_none(self.memory_fraction),
            "maximum_memory_fraction": MAXIMUM_RECOMMENDED_MEMORY_FRACTION,
            "projected_hours_per_model": finite_or_none(
                self.projected_hours_per_model
            ),
            "wall_time_safety_factor": WALL_TIME_SAFETY_FACTOR,
            "safety_adjusted_hours_per_model": finite_or_none(
                self.safety_adjusted_hours_per_model
            ),
            "maximum_safety_adjusted_hours_per_model": (
                MAXIMUM_SAFETY_ADJUSTED_HOURS_PER_MODEL
            ),
            "core_model_runs": CORE_MODEL_RUNS,
            "core_model_families": CORE_MODEL_FAMILIES,
            "family_aware_campaign_lock": False,
            "projected_core_campaign_hours": finite_or_none(
                self.projected_core_campaign_hours
            ),
            "maximum_core_campaign_hours": MAXIMUM_CORE_CAMPAIGN_HOURS,
            "pass": self.passes,
        }


def select_largest_feasible_scale(
    results: Mapping[int, ScaleFeasibility],
) -> int | None:
    """Select a provisional scale from candidate-only preflight results."""

    if set(results) != set(PUBLICATION_SCALE_ORDER):
        raise ValueError("scale selection requires all three blind candidates")
    for target in reversed(PUBLICATION_SCALE_ORDER):
        result = results[target]
        if result.target_millions != target:
            raise ValueError("scale feasibility identity mismatch")
        if result.passes:
            return target
    return None


@dataclass(frozen=True, slots=True)
class FamilyScaleFeasibility:
    target_millions: int
    family: str
    completed: bool
    finite_steps: bool
    parameter_count: int
    expected_parameter_count: int
    maximum_driver_allocated_bytes: int
    recommended_max_memory_bytes: int
    median_train_step_seconds: float
    raw_source_bytes_per_step: int
    raw_reference_descriptor_identity_sha256: str = ""
    raw_reference_policy: str = ""
    auxiliary_kind: str = PUBLICATION_AUXILIARY_NONE
    auxiliary_parameter_count: int = 0
    expected_auxiliary_parameter_count: int = 0
    auxiliary_config_sha256: str = ""
    auxiliary_train_completed: bool = False
    auxiliary_train_finite_steps: bool = False
    auxiliary_train_measurement_count: int = 0
    auxiliary_train_workload_sha256: str = ""
    median_auxiliary_train_step_seconds: float = 0.0
    auxiliary_train_raw_source_bytes_per_step: int = 0
    auxiliary_train_total_raw_source_bytes: int = 0
    auxiliary_train_maximum_driver_allocated_bytes: int = 0
    auxiliary_score_completed: bool = False
    auxiliary_score_finite_steps: bool = False
    auxiliary_score_measurement_count: int = 0
    auxiliary_score_workload_sha256: str = ""
    median_auxiliary_score_step_seconds: float = 0.0
    auxiliary_score_raw_source_bytes_per_step: int = 0
    auxiliary_score_total_raw_source_bytes: int = 0
    auxiliary_score_maximum_driver_allocated_bytes: int = 0
    auxiliary_runtime_completed: bool = False
    auxiliary_runtime_finite: bool = False
    auxiliary_runtime_measurement_count: int = 0
    auxiliary_runtime_observed_bytes: int = 0
    auxiliary_runtime_expected_bytes: int = 0
    auxiliary_runtime_router_forward_calls: int = 0
    auxiliary_runtime_workload_sha256: str = ""
    auxiliary_runtime_maximum_driver_allocated_bytes: int = 0

    @property
    def memory_fraction(self) -> float:
        if self.recommended_max_memory_bytes <= 0:
            return math.inf
        return self.effective_maximum_driver_allocated_bytes / (
            self.recommended_max_memory_bytes
        )

    @property
    def effective_maximum_driver_allocated_bytes(self) -> int:
        return max(
            self.maximum_driver_allocated_bytes,
            self.auxiliary_train_maximum_driver_allocated_bytes,
            self.auxiliary_score_maximum_driver_allocated_bytes,
            self.auxiliary_runtime_maximum_driver_allocated_bytes,
        )

    @property
    def safety_adjusted_hours_per_model(self) -> float:
        return self.projected_hours_per_model * WALL_TIME_SAFETY_FACTOR

    @staticmethod
    def _projected_component_hours(
        median_step_seconds: float,
        raw_source_bytes_per_step: int,
        total_raw_source_bytes: int,
    ) -> float:
        if (
            median_step_seconds == 0.0
            and raw_source_bytes_per_step == 0
            and total_raw_source_bytes == 0
        ):
            return 0.0
        if (
            not math.isfinite(median_step_seconds)
            or median_step_seconds <= 0
            or raw_source_bytes_per_step <= 0
            or total_raw_source_bytes <= 0
        ):
            return math.inf
        steps = math.ceil(total_raw_source_bytes / raw_source_bytes_per_step)
        return median_step_seconds * steps / 3_600

    @property
    def projected_train_steps(self) -> int:
        if self.raw_source_bytes_per_step <= 0:
            return 0
        return math.ceil(
            PUBLICATION_TRAIN_BYTES / self.raw_source_bytes_per_step
        )

    @property
    def projected_hours_per_model(self) -> float:
        return sum(
            (
                self._projected_component_hours(
                    self.median_train_step_seconds,
                    self.raw_source_bytes_per_step,
                    PUBLICATION_TRAIN_BYTES,
                ),
                self._projected_component_hours(
                    self.median_auxiliary_train_step_seconds,
                    self.auxiliary_train_raw_source_bytes_per_step,
                    self.auxiliary_train_total_raw_source_bytes,
                ),
                self._projected_component_hours(
                    self.median_auxiliary_score_step_seconds,
                    self.auxiliary_score_raw_source_bytes_per_step,
                    self.auxiliary_score_total_raw_source_bytes,
                ),
            )
        )

    @property
    def total_runtime_parameter_count(self) -> int:
        return self.parameter_count + self.auxiliary_parameter_count

    @property
    def projected_main_train_hours(self) -> float:
        return self._projected_component_hours(
            self.median_train_step_seconds,
            self.raw_source_bytes_per_step,
            PUBLICATION_TRAIN_BYTES,
        )

    @property
    def projected_auxiliary_train_hours(self) -> float:
        return self._projected_component_hours(
            self.median_auxiliary_train_step_seconds,
            self.auxiliary_train_raw_source_bytes_per_step,
            self.auxiliary_train_total_raw_source_bytes,
        )

    @property
    def projected_auxiliary_score_hours(self) -> float:
        return self._projected_component_hours(
            self.median_auxiliary_score_step_seconds,
            self.auxiliary_score_raw_source_bytes_per_step,
            self.auxiliary_score_total_raw_source_bytes,
        )

    @property
    def auxiliary_contract_pass(self) -> bool:
        if self.family == "raw_byte_reference":
            if (
                not is_sha256(self.raw_reference_descriptor_identity_sha256)
                or not self.raw_reference_policy
            ):
                return False
            try:
                expected_kind = publication_auxiliary_kind_for_policy(
                    self.raw_reference_policy
                )
            except (TypeError, ValueError):
                return False
        else:
            if (
                self.raw_reference_descriptor_identity_sha256
                or self.raw_reference_policy
            ):
                return False
            expected_kind = PUBLICATION_AUXILIARY_NONE
        if self.auxiliary_kind != expected_kind:
            return False
        if self.auxiliary_kind == PUBLICATION_AUXILIARY_NONE:
            return bool(
                self.auxiliary_parameter_count == 0
                and self.expected_auxiliary_parameter_count == 0
                and self.auxiliary_config_sha256 == ""
                and not self.auxiliary_train_completed
                and not self.auxiliary_train_finite_steps
                and self.auxiliary_train_measurement_count == 0
                and self.auxiliary_train_workload_sha256 == ""
                and self.median_auxiliary_train_step_seconds == 0.0
                and self.auxiliary_train_raw_source_bytes_per_step == 0
                and self.auxiliary_train_total_raw_source_bytes == 0
                and self.auxiliary_train_maximum_driver_allocated_bytes == 0
                and not self.auxiliary_score_completed
                and not self.auxiliary_score_finite_steps
                and self.auxiliary_score_measurement_count == 0
                and self.auxiliary_score_workload_sha256 == ""
                and self.median_auxiliary_score_step_seconds == 0.0
                and self.auxiliary_score_raw_source_bytes_per_step == 0
                and self.auxiliary_score_total_raw_source_bytes == 0
                and self.auxiliary_score_maximum_driver_allocated_bytes == 0
                and not self.auxiliary_runtime_completed
                and not self.auxiliary_runtime_finite
                and self.auxiliary_runtime_measurement_count == 0
                and self.auxiliary_runtime_observed_bytes == 0
                and self.auxiliary_runtime_expected_bytes == 0
                and self.auxiliary_runtime_router_forward_calls == 0
                and self.auxiliary_runtime_workload_sha256 == ""
                and self.auxiliary_runtime_maximum_driver_allocated_bytes == 0
            )
        if self.auxiliary_kind != PUBLICATION_AUXILIARY_ENTROPY_ROUTER:
            return False
        return bool(
            self.family == "raw_byte_reference"
            and self.expected_auxiliary_parameter_count
            == PUBLICATION_ROUTER_EXPECTED_PARAMETERS.get(self.target_millions)
            and self.auxiliary_parameter_count
            == self.expected_auxiliary_parameter_count
            and self.auxiliary_parameter_count > 0
            and is_sha256(self.auxiliary_config_sha256)
            and self.auxiliary_train_completed
            and self.auxiliary_train_finite_steps
            and self.auxiliary_train_measurement_count
            == PUBLICATION_FEASIBILITY_MEASUREMENT_STEPS
            and is_sha256(self.auxiliary_train_workload_sha256)
            and math.isfinite(self.median_auxiliary_train_step_seconds)
            and self.median_auxiliary_train_step_seconds > 0
            and self.auxiliary_train_raw_source_bytes_per_step > 0
            and self.auxiliary_train_total_raw_source_bytes
            == PUBLICATION_TRAIN_BYTES
            and self.auxiliary_train_maximum_driver_allocated_bytes > 0
            and self.auxiliary_score_completed
            and self.auxiliary_score_finite_steps
            and self.auxiliary_score_measurement_count
            == PUBLICATION_FEASIBILITY_MEASUREMENT_STEPS
            and is_sha256(self.auxiliary_score_workload_sha256)
            and math.isfinite(self.median_auxiliary_score_step_seconds)
            and self.median_auxiliary_score_step_seconds > 0
            and self.auxiliary_score_raw_source_bytes_per_step > 0
            and self.auxiliary_score_total_raw_source_bytes
            >= PUBLICATION_TRAIN_BYTES
            and self.auxiliary_score_maximum_driver_allocated_bytes > 0
            and self.auxiliary_runtime_completed
            and self.auxiliary_runtime_finite
            and self.auxiliary_runtime_measurement_count
            == PUBLICATION_FEASIBILITY_MEASUREMENT_STEPS
            and self.auxiliary_runtime_observed_bytes > 0
            and self.auxiliary_runtime_observed_bytes
            == self.auxiliary_runtime_expected_bytes
            and self.auxiliary_runtime_router_forward_calls > 0
            and is_sha256(self.auxiliary_runtime_workload_sha256)
            and self.auxiliary_runtime_maximum_driver_allocated_bytes > 0
            and len(
                {
                    self.auxiliary_config_sha256,
                    self.auxiliary_train_workload_sha256,
                    self.auxiliary_score_workload_sha256,
                    self.auxiliary_runtime_workload_sha256,
                }
            )
            == 4
        )

    @property
    def passes(self) -> bool:
        expected_by_family = PUBLICATION_FAMILY_EXPECTED_PARAMETERS.get(
            self.target_millions,
            {},
        )
        return bool(
            self.family in CORE_MODEL_FAMILIES
            and self.expected_parameter_count
            == expected_by_family.get(self.family)
            and self.completed
            and self.finite_steps
            and self.parameter_count == self.expected_parameter_count
            and self.expected_parameter_count > 0
            and self.auxiliary_contract_pass
            and self.maximum_driver_allocated_bytes > 0
            and self.raw_source_bytes_per_step > 0
            and math.isfinite(self.median_train_step_seconds)
            and self.median_train_step_seconds > 0
            and math.isfinite(self.projected_hours_per_model)
            and self.projected_hours_per_model > 0
            and self.recommended_max_memory_bytes > 0
            and self.memory_fraction <= MAXIMUM_RECOMMENDED_MEMORY_FRACTION
            and self.safety_adjusted_hours_per_model
            <= MAXIMUM_SAFETY_ADJUSTED_HOURS_PER_MODEL
        )

    def to_dict(self) -> dict[str, object]:
        def finite_or_none(value: float) -> float | None:
            return value if math.isfinite(value) else None

        return {
            "target_millions": self.target_millions,
            "family": self.family,
            "completed": self.completed,
            "finite_steps": self.finite_steps,
            "parameter_count": self.parameter_count,
            "expected_parameter_count": self.expected_parameter_count,
            "raw_reference_descriptor_identity_sha256": (
                self.raw_reference_descriptor_identity_sha256
            ),
            "raw_reference_policy": self.raw_reference_policy,
            "auxiliary_kind": self.auxiliary_kind,
            "auxiliary_parameter_count": self.auxiliary_parameter_count,
            "expected_auxiliary_parameter_count": (
                self.expected_auxiliary_parameter_count
            ),
            "auxiliary_config_sha256": self.auxiliary_config_sha256,
            "total_runtime_parameter_count": self.total_runtime_parameter_count,
            "auxiliary_contract_pass": self.auxiliary_contract_pass,
            "maximum_driver_allocated_bytes": self.maximum_driver_allocated_bytes,
            "effective_maximum_driver_allocated_bytes": (
                self.effective_maximum_driver_allocated_bytes
            ),
            "recommended_max_memory_bytes": self.recommended_max_memory_bytes,
            "memory_fraction": finite_or_none(self.memory_fraction),
            "median_train_step_seconds": finite_or_none(
                self.median_train_step_seconds
            ),
            "raw_source_bytes_per_step": self.raw_source_bytes_per_step,
            "auxiliary_train_completed": self.auxiliary_train_completed,
            "auxiliary_train_finite_steps": self.auxiliary_train_finite_steps,
            "auxiliary_train_measurement_count": (
                self.auxiliary_train_measurement_count
            ),
            "auxiliary_train_workload_sha256": (
                self.auxiliary_train_workload_sha256
            ),
            "median_auxiliary_train_step_seconds": finite_or_none(
                self.median_auxiliary_train_step_seconds
            ),
            "auxiliary_train_raw_source_bytes_per_step": (
                self.auxiliary_train_raw_source_bytes_per_step
            ),
            "auxiliary_train_total_raw_source_bytes": (
                self.auxiliary_train_total_raw_source_bytes
            ),
            "auxiliary_train_maximum_driver_allocated_bytes": (
                self.auxiliary_train_maximum_driver_allocated_bytes
            ),
            "auxiliary_score_completed": self.auxiliary_score_completed,
            "auxiliary_score_finite_steps": self.auxiliary_score_finite_steps,
            "auxiliary_score_measurement_count": (
                self.auxiliary_score_measurement_count
            ),
            "auxiliary_score_workload_sha256": (
                self.auxiliary_score_workload_sha256
            ),
            "median_auxiliary_score_step_seconds": finite_or_none(
                self.median_auxiliary_score_step_seconds
            ),
            "auxiliary_score_raw_source_bytes_per_step": (
                self.auxiliary_score_raw_source_bytes_per_step
            ),
            "auxiliary_score_total_raw_source_bytes": (
                self.auxiliary_score_total_raw_source_bytes
            ),
            "auxiliary_score_maximum_driver_allocated_bytes": (
                self.auxiliary_score_maximum_driver_allocated_bytes
            ),
            "auxiliary_runtime_completed": self.auxiliary_runtime_completed,
            "auxiliary_runtime_finite": self.auxiliary_runtime_finite,
            "auxiliary_runtime_measurement_count": (
                self.auxiliary_runtime_measurement_count
            ),
            "auxiliary_runtime_observed_bytes": (
                self.auxiliary_runtime_observed_bytes
            ),
            "auxiliary_runtime_expected_bytes": (
                self.auxiliary_runtime_expected_bytes
            ),
            "auxiliary_runtime_router_forward_calls": (
                self.auxiliary_runtime_router_forward_calls
            ),
            "auxiliary_runtime_workload_sha256": (
                self.auxiliary_runtime_workload_sha256
            ),
            "auxiliary_runtime_maximum_driver_allocated_bytes": (
                self.auxiliary_runtime_maximum_driver_allocated_bytes
            ),
            "projected_train_bytes": PUBLICATION_TRAIN_BYTES,
            "projected_train_steps": self.projected_train_steps,
            "projected_main_train_hours": finite_or_none(
                self.projected_main_train_hours
            ),
            "projected_auxiliary_train_hours": finite_or_none(
                self.projected_auxiliary_train_hours
            ),
            "projected_auxiliary_score_hours": finite_or_none(
                self.projected_auxiliary_score_hours
            ),
            "projected_hours_per_model": finite_or_none(
                self.projected_hours_per_model
            ),
            "safety_adjusted_hours_per_model": finite_or_none(
                self.safety_adjusted_hours_per_model
            ),
            "pass": self.passes,
        }


@dataclass(frozen=True, slots=True)
class CampaignScaleFeasibility:
    target_millions: int
    raw_reference_descriptor: PublicationRawReferenceDescriptor
    family_results: tuple[FamilyScaleFeasibility, ...]

    @property
    def projected_campaign_hours(self) -> float:
        return len(PUBLICATION_PRETRAIN_SEEDS) * sum(
            result.safety_adjusted_hours_per_model
            for result in self.family_results
        )

    @property
    def passes(self) -> bool:
        try:
            validate_publication_raw_reference_descriptor(
                self.raw_reference_descriptor
            )
            descriptor_pass = True
        except (TypeError, ValueError):
            descriptor_pass = False
        raw_result = (
            self.family_results[1]
            if len(self.family_results) == len(CORE_MODEL_FAMILIES)
            else None
        )
        return bool(
            self.target_millions in PUBLICATION_SCALE_ORDER
            and descriptor_pass
            and tuple(result.family for result in self.family_results)
            == CORE_MODEL_FAMILIES
            and raw_result is not None
            and raw_result.raw_reference_descriptor_identity_sha256
            == self.raw_reference_descriptor.identity_sha256
            and raw_result.raw_reference_policy
            == self.raw_reference_descriptor.policy
            and raw_result.auxiliary_kind
            == self.raw_reference_descriptor.auxiliary_kind
            and all(
                result.target_millions == self.target_millions
                for result in self.family_results
            )
            and all(result.passes for result in self.family_results)
            and math.isfinite(self.projected_campaign_hours)
            and self.projected_campaign_hours <= MAXIMUM_CORE_CAMPAIGN_HOURS
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "target_millions": self.target_millions,
            "raw_reference_descriptor": self.raw_reference_descriptor.to_dict(),
            "family_results": tuple(
                result.to_dict() for result in self.family_results
            ),
            "pretrain_seeds": PUBLICATION_PRETRAIN_SEEDS,
            "projected_campaign_hours": (
                self.projected_campaign_hours
                if math.isfinite(self.projected_campaign_hours)
                else None
            ),
            "maximum_campaign_hours": MAXIMUM_CORE_CAMPAIGN_HOURS,
            "family_aware_campaign_lock": True,
            "pass": self.passes,
        }


def select_largest_campaign_feasible_scale(
    results: Mapping[int, CampaignScaleFeasibility],
) -> int | None:
    """Lock the largest scale only after all four runtime families run."""

    if set(results) != set(PUBLICATION_SCALE_ORDER):
        raise ValueError("campaign scale selection requires all blind candidates")
    for target in reversed(PUBLICATION_SCALE_ORDER):
        result = results[target]
        if result.target_millions != target:
            raise ValueError("campaign feasibility identity mismatch")
        if result.passes:
            return target
    return None
