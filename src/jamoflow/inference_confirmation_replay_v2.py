"""Independent causal-forward replay for post-selection confirmation models."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .compute_conversion import (
    CONVERSION_RATES,
    conversion_model_spec,
    conversion_patch_matrices,
)
from .hplt3 import hash_file
from .inference_calibration_replay_v2 import (
    array_sha256,
    release_model,
    state_sha256,
)
from .inference_confirmation_evidence_v2 import CALIBRATION_SEQUENCE_COUNT
from .inference_final_authorization_v2 import (
    FINAL_MAIN_PARAMETER_COUNT,
    FINAL_ROUTER_PARAMETER_COUNT,
    canonical_sha256,
    expected_model_paths,
    validate_final_auxiliary_bundle,
)
from .neural_data import build_neural_stream
from .neural_model import build_main_model, build_router, parameter_count
from .neural_training import evaluate_main_model, router_entropy_scores
from .phase1 import stream_arrays
from .phase2_patching import (
    calibrate_threshold,
    compact_whitespace_mask,
    threshold_patch_matrix,
    validate_padded_patch_matrix,
    variable_patch_diagnostics,
)
from .phase3 import (
    PHASE3_MODEL_SPEC,
    PHASE3_OPTIMIZATION_SPEC,
    THRESHOLD_POLICIES,
    spacebyte_causal_prefix_mask,
    structural_patch_matrices,
)
from .publication_reference import entropy_policy_definition_sha256


SOURCE_PATH = Path("data/processed/hplt3-korean-phase3/ko.jsonl")
INTEGRITY_PATH = Path("data/processed/hplt3-korean-phase3/integrity.json")
PHASE3_MANIFEST_PATH = Path("runs/phase3/manifest.json")
CALIBRATION_BYTES = 8_000_000
GLOBAL_POSITION_LIMIT = 1_032


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def confirmation_model_spec(descriptor: Mapping[str, Any]):
    if descriptor["model_family"] == "phase3":
        return PHASE3_MODEL_SPEC
    return conversion_model_spec(int(descriptor["patch_count"]))


def load_confirmation_calibration_context(
    plan: Mapping[str, Any],
) -> tuple[bytes, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if (
        hash_file(SOURCE_PATH) != plan["initial_design"]["source_artifact_sha256"]
        or hash_file(INTEGRITY_PATH)
        != plan["initial_design"]["source_integrity_artifact_sha256"]
    ):
        raise ValueError("confirmation calibration source differs from the plan")
    stream = build_neural_stream(
        SOURCE_PATH,
        language="ko",
        split="calibration",
        byte_limit=CALIBRATION_BYTES,
        sequence_length=PHASE3_MODEL_SPEC.sequence_length,
    )
    inputs, boundaries = stream_arrays(
        stream.data,
        stream.codepoint_boundaries,
        stream.sequence_length,
    )
    if (
        hashlib.sha256(stream.data).hexdigest()
        != plan["calibration_evaluator"]["input_stream_sha256"]
        or len(inputs) != CALIBRATION_SEQUENCE_COUNT
        or len(inputs) != plan["calibration_evaluator"]["sequence_count"]
    ):
        raise ValueError("confirmation calibration stream differs from the plan")
    whitespace = compact_whitespace_mask(stream.data).reshape(inputs.shape)
    spacelike = spacebyte_causal_prefix_mask(stream.data).reshape(inputs.shape)
    matrices = structural_patch_matrices(boundaries, whitespace, spacelike)
    for rate in CONVERSION_RATES:
        matrices.update(
            conversion_patch_matrices(boundaries, whitespace, rate=rate)
        )
    return stream.data, inputs, boundaries, matrices


def confirmation_entropy_matrices_and_auxiliary(
    *,
    seed: int,
    required_policies: tuple[str, ...],
    inputs: np.ndarray,
    boundaries: np.ndarray,
    plan: Mapping[str, Any],
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    entropy_policies = tuple(
        policy for policy in required_policies if policy in THRESHOLD_POLICIES
    )
    if not entropy_policies:
        return {}, {}
    artifact_root = Path("artifacts/phase3") / f"seed-{seed}"
    run_root = Path("runs/phase3") / f"seed-{seed}"
    router_checkpoint = artifact_root / "router.pt"
    router_report_path = run_root / "router.json"
    cache_path = artifact_root / "threshold-patches.npz"
    diagnostics_path = run_root / "threshold-patch-diagnostics.json"
    for artifact_path in (
        router_checkpoint,
        router_report_path,
        cache_path,
        diagnostics_path,
        PHASE3_MANIFEST_PATH,
    ):
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
    router_report = _read_json(router_report_path)
    phase3_manifest = _read_json(PHASE3_MANIFEST_PATH)
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    router.load_state_dict(
        torch.load(router_checkpoint, map_location="cpu", weights_only=True)
    )
    router_state_sha256 = state_sha256(router)
    if (
        router_report.get("seed") != seed
        or router_report.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or router_report.get("optimization_spec")
        != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or router_report.get("parameters") != FINAL_ROUTER_PARAMETER_COUNT
        or parameter_count(router) != FINAL_ROUTER_PARAMETER_COUNT
        or router_report.get("trained_state_sha256") != router_state_sha256
    ):
        raise ValueError(f"confirmation router identity differs for seed {seed}")
    scores = router_entropy_scores(router, inputs, device)
    matrices: dict[str, np.ndarray] = {}
    calibrations: dict[str, Any] = {}
    for policy in entropy_policies:
        candidate_mask = None if policy == "entropy_threshold_full" else boundaries
        calibration = calibrate_threshold(
            scores,
            PHASE3_MODEL_SPEC.patch_count,
            candidate_masks=candidate_mask,
            maximum_patch_length=24,
        )
        calibrations[policy] = calibration
        matrices[policy] = threshold_patch_matrix(
            scores,
            calibration.threshold_nats,
            candidate_masks=candidate_mask,
            maximum_patch_length=24,
        )
    del scores
    release_model(router, device)

    diagnostics = _read_json(diagnostics_path)
    provenance = diagnostics.get("_provenance", {})
    if (
        provenance.get("kind") != "phase3_threshold_patch_cache"
        or provenance.get("seed") != seed
        or provenance.get("router_state_sha256") != router_state_sha256
        or provenance.get("model_spec") != PHASE3_MODEL_SPEC.to_dict()
        or provenance.get("maximum_patch_length") != 24
        or provenance.get("splits", {}).get("calibration", {}).get(
            "inputs_sha256"
        )
        != array_sha256(inputs)
        or provenance.get("splits", {}).get("calibration", {}).get(
            "boundaries_sha256"
        )
        != array_sha256(boundaries)
    ):
        raise ValueError(f"confirmation router provenance differs for seed {seed}")
    auxiliary: dict[str, dict[str, Any]] = {}
    with np.load(cache_path, allow_pickle=False) as cache:
        for policy in entropy_policies:
            matrix = matrices[policy]
            key = f"calibration__{policy}"
            calibration = calibrations[policy]
            expected_diagnostics = {
                **variable_patch_diagnostics(matrix, boundaries).to_dict(),
                "matrix_sha256": array_sha256(matrix),
            }
            if (
                key not in cache.files
                or not np.array_equal(cache[key], matrix)
                or diagnostics.get("calibration", {}).get(policy)
                != calibration.to_dict()
                or diagnostics.get("splits", {}).get("calibration", {}).get(
                    policy
                )
                != expected_diagnostics
            ):
                raise ValueError(
                    f"confirmation router matrix differs for {seed}/{policy}"
                )
            auxiliary[policy] = {
                "calibration_stream_sha256": plan["calibration_evaluator"][
                    "input_stream_sha256"
                ],
                "candidate_mask": (
                    "none" if policy == "entropy_threshold_full" else "codepoint"
                ),
                "kind": "entropy_router",
                "maximum_patch_length": 24,
                "policy": policy,
                "policy_definition_sha256": entropy_policy_definition_sha256(policy),
                "router_checkpoint_artifact_sha256": hash_file(router_checkpoint),
                "router_checkpoint_path": str(router_checkpoint),
                "router_checkpoint_state_sha256": router_state_sha256,
                "router_config_sha256": canonical_sha256(
                    PHASE3_MODEL_SPEC.to_dict()
                ),
                "router_parameter_count": FINAL_ROUTER_PARAMETER_COUNT,
                "router_report_artifact_sha256": hash_file(router_report_path),
                "router_report_path": str(router_report_path),
                "router_training_stream_sha256": phase3_manifest["streams"][
                    "train"
                ]["selected_stream_sha256"],
                "seed": seed,
                "threshold_cache_artifact_sha256": hash_file(cache_path),
                "threshold_cache_path": str(cache_path),
                "threshold_diagnostics_artifact_sha256": hash_file(
                    diagnostics_path
                ),
                "threshold_diagnostics_path": str(diagnostics_path),
                "threshold_nats": calibration.threshold_nats,
            }
    return matrices, auxiliary


def replay_confirmation_unit(
    *,
    descriptor: Mapping[str, Any],
    seed: int,
    inputs: np.ndarray,
    boundaries: np.ndarray,
    matrix: np.ndarray,
    auxiliary: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    paths = expected_model_paths(descriptor, seed)
    report_path = Path(paths["training_report"])
    checkpoint_path = Path(paths["checkpoint"])
    if (
        report_path.is_symlink()
        or checkpoint_path.is_symlink()
        or not report_path.is_file()
        or not checkpoint_path.is_file()
    ):
        raise FileNotFoundError(f"confirmation model is missing: {seed}/{descriptor['policy']}")
    report = _read_json(report_path)
    spec = confirmation_model_spec(descriptor)
    model = build_main_model(
        spec,
        seed=seed,
        global_max_position_embeddings=GLOBAL_POSITION_LIMIT,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    )
    state_sha256_value = state_sha256(model)
    report_bpb = report.get("evaluation", {}).get("calibration", {}).get("bpb")
    canonical_auxiliary = validate_final_auxiliary_bundle(
        auxiliary,
        descriptor,
        seed,
    )
    family_specific_identity_matches = True
    if descriptor["model_family"] == "compute_conversion":
        family_specific_identity_matches = (
            report.get("global_max_position_embeddings")
            == GLOBAL_POSITION_LIMIT
            and report.get("checkpoint_artifact_sha256")
            == hash_file(checkpoint_path)
        )
    else:
        phase3_manifest = _read_json(PHASE3_MANIFEST_PATH)
        family_specific_identity_matches = (
            phase3_manifest.get("global_max_position_embeddings")
            == GLOBAL_POSITION_LIMIT
        )
    if (
        report.get("seed") != seed
        or report.get("policy") != descriptor["policy"]
        or report.get("parameters") != FINAL_MAIN_PARAMETER_COUNT
        or parameter_count(model) != FINAL_MAIN_PARAMETER_COUNT
        or report.get("model_spec") != spec.to_dict()
        or report.get("optimization_spec") != PHASE3_OPTIMIZATION_SPEC.to_dict()
        or report.get("trained_state_sha256") != state_sha256_value
        or not family_specific_identity_matches
        or report.get("patch_matrix_sha256", {}).get("calibration")
        != array_sha256(matrix)
        or report.get("patch_diagnostics", {}).get("calibration")
        != variable_patch_diagnostics(matrix, boundaries).to_dict()
        or not isinstance(report_bpb, (int, float))
        or isinstance(report_bpb, bool)
        or not math.isfinite(float(report_bpb))
    ):
        release_model(model, device)
        raise ValueError(f"confirmation training evidence differs: {seed}")
    validate_padded_patch_matrix(matrix, PHASE3_MODEL_SPEC.sequence_length)
    summary, losses = evaluate_main_model(
        model,
        inputs,
        matrix,
        device,
        batch_size=PHASE3_OPTIMIZATION_SPEC.evaluation_batch_size,
        return_sequence_nll=True,
    )
    if losses is None or losses.dtype != np.float32:
        release_model(model, device)
        raise ValueError("confirmation evaluator did not return exact float32 NLL")
    values = np.ascontiguousarray(losses)
    release_model(model, device)
    bpb = math.fsum(float(value) for value in values) / (
        len(values) * (PHASE3_MODEL_SPEC.sequence_length - 1) * math.log(2)
    )
    if (
        values.shape != (CALIBRATION_SEQUENCE_COUNT,)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or summary.examples != CALIBRATION_SEQUENCE_COUNT
        or summary.predicted_bytes
        != CALIBRATION_SEQUENCE_COUNT * (PHASE3_MODEL_SPEC.sequence_length - 1)
        or not math.isclose(summary.bpb, bpb, rel_tol=0, abs_tol=1e-7)
        or not math.isclose(float(report_bpb), bpb, rel_tol=0, abs_tol=1e-7)
    ):
        raise ValueError(f"confirmation calibration replay differs: {seed}")
    return {
        "auxiliary": canonical_auxiliary,
        "bpb": bpb,
        "checkpoint_artifact_sha256": hash_file(checkpoint_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_state_sha256": state_sha256_value,
        "losses": values,
        "nll_array_sha256": array_sha256(values),
        "parameter_count": FINAL_MAIN_PARAMETER_COUNT,
        "report_artifact_sha256": hash_file(report_path),
        "report_path": str(report_path),
        "spec_sha256": canonical_sha256(spec.to_dict()),
    }


def validate_confirmation_replay_against_receipt(
    *,
    receipt: Mapping[str, Any],
    replay: Mapping[str, Any],
    stream: bytes,
    inputs: np.ndarray,
    boundaries: np.ndarray,
    matrix: np.ndarray,
) -> None:
    calibration = receipt.get("calibration")
    checkpoint = receipt.get("checkpoint")
    report = receipt.get("training_report")
    model = receipt.get("model")
    if not all(
        isinstance(value, Mapping)
        for value in (calibration, checkpoint, report, model)
    ):
        raise ValueError("confirmation receipt replay sections are malformed")
    if (
        replay.get("nll_array_sha256") != calibration.get("nll_array_sha256")
        or replay.get("bpb") != calibration.get("bpb")
        or calibration.get("inputs_sha256") != array_sha256(inputs)
        or calibration.get("boundaries_sha256") != array_sha256(boundaries)
        or calibration.get("matrix_sha256") != array_sha256(matrix)
        or calibration.get("stream_sha256") != hashlib.sha256(stream).hexdigest()
        or calibration.get("count") != len(inputs)
        or calibration.get("predicted_bytes") != len(inputs) * 511
        or calibration.get("dtype") != "float32"
        or replay.get("checkpoint_artifact_sha256")
        != checkpoint.get("artifact_sha256")
        or replay.get("checkpoint_path") != checkpoint.get("path")
        or replay.get("checkpoint_state_sha256") != checkpoint.get("state_sha256")
        or replay.get("report_artifact_sha256") != report.get("artifact_sha256")
        or replay.get("report_path") != report.get("path")
        or replay.get("parameter_count") != model.get("parameter_count")
        or replay.get("spec_sha256") != model.get("spec_sha256")
        or replay.get("auxiliary") != receipt.get("auxiliary")
    ):
        raise ValueError("confirmation receipt fails independent causal replay")
