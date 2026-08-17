"""Exact physical-model loading and runtime construction for timing v5."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .compute_conversion import conversion_model_spec
from .hplt3 import hash_file
from .incremental_blt import (
    INCREMENTAL_ENTROPY_POLICIES,
    IncrementalBltDecoder,
    IncrementalEntropyBltDecoder,
)
from .inference_final_authorization_v2 import (
    FINAL_MAIN_PARAMETER_COUNT,
    FINAL_ROUTER_PARAMETER_COUNT,
    FINAL_SEEDS,
    canonical_sha256,
    validate_final_model_identity,
)
from .inference_actual_v5 import (
    ACTUAL_INFERENCE_V5_CPU_EQUIVALENCE_ATOL,
    ACTUAL_INFERENCE_V5_CPU_EQUIVALENCE_RTOL,
    ACTUAL_INFERENCE_V5_GLOBAL_POSITION_LIMIT,
    ACTUAL_INFERENCE_V5_MAXIMUM_PROBABILITY_TOTAL_VARIATION,
    ACTUAL_INFERENCE_V5_MPS_EQUIVALENCE_ATOL,
    ACTUAL_INFERENCE_V5_MPS_EQUIVALENCE_RTOL,
    ACTUAL_INFERENCE_V5_PATCHING_HORIZON,
)
from .neural_model import build_main_model, build_router, parameter_count
from .phase2_patching import (
    entropy_threshold_boundaries,
    padded_hf_patch_matrix,
)
from .phase3 import PHASE3_MODEL_SPEC
from .publication_reference import entropy_policy_definition_sha256
from .utf8 import prefix_boundary_mask


ACTUAL_INFERENCE_GLOBAL_POSITION_LIMIT = ACTUAL_INFERENCE_V5_GLOBAL_POSITION_LIMIT
ACTUAL_INFERENCE_PATCH_HORIZON = ACTUAL_INFERENCE_V5_PATCHING_HORIZON
# The CPU semantic oracle retains the original v5/v5r1 contract.  MPS uses a
# separate backend envelope because parallel and sequential tensor shapes can
# select different reduction kernels even when the cache semantics are equal.
ACTUAL_INFERENCE_EQUIVALENCE_RTOL = ACTUAL_INFERENCE_V5_CPU_EQUIVALENCE_RTOL
ACTUAL_INFERENCE_EQUIVALENCE_ATOL = ACTUAL_INFERENCE_V5_CPU_EQUIVALENCE_ATOL
ACTUAL_INFERENCE_MPS_EQUIVALENCE_RTOL = ACTUAL_INFERENCE_V5_MPS_EQUIVALENCE_RTOL
ACTUAL_INFERENCE_MPS_EQUIVALENCE_ATOL = ACTUAL_INFERENCE_V5_MPS_EQUIVALENCE_ATOL
ACTUAL_INFERENCE_MAXIMUM_PROBABILITY_TOTAL_VARIATION = (
    ACTUAL_INFERENCE_V5_MAXIMUM_PROBABILITY_TOTAL_VARIATION
)


def state_dict_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def model_spec_for_descriptor(descriptor: Mapping[str, Any]) -> Any:
    if descriptor["model_family"] == "phase3":
        return PHASE3_MODEL_SPEC
    if descriptor["model_family"] == "compute_conversion":
        return conversion_model_spec(int(descriptor["patch_count"]))
    raise ValueError("actual-inference model family differs")


@dataclass(slots=True)
class LoadedActualModel:
    role: str
    identity: Mapping[str, Any]
    seed: int
    model: Any
    router: Any | None
    threshold_nats: float | None
    maximum_patch_length: int | None

    @property
    def descriptor(self) -> Mapping[str, Any]:
        return self.identity["descriptor"]

    @property
    def runtime_policy(self) -> str:
        return str(self.descriptor["runtime_policy"])

    @property
    def requires_entropy_router(self) -> bool:
        return bool(self.descriptor["requires_entropy_router"])

    @property
    def patch_count(self) -> int:
        return int(self.descriptor["patch_count"])

    @property
    def evidence(self) -> Mapping[str, Any]:
        return self.identity["seeds"][str(self.seed)]

    @property
    def device(self) -> str:
        return str(next(self.model.parameters()).device)

    def runtime(self) -> Any:
        if self.runtime_policy in INCREMENTAL_ENTROPY_POLICIES:
            if (
                self.router is None
                or self.threshold_nats is None
                or self.maximum_patch_length is None
            ):
                raise RuntimeError("entropy runtime lacks its sealed router")
            return IncrementalEntropyBltDecoder(
                self.model,
                self.router,
                self.runtime_policy,
                threshold_nats=self.threshold_nats,
                maximum_patch_length=self.maximum_patch_length,
                horizon=ACTUAL_INFERENCE_PATCH_HORIZON,
                patch_count=self.patch_count,
                fixed_stride=model_spec_for_descriptor(self.descriptor).patch_stride,
            )
        if self.router is not None:
            raise RuntimeError("structural runtime unexpectedly has a router")
        return IncrementalBltDecoder(
            self.model,
            self.runtime_policy,
            horizon=ACTUAL_INFERENCE_PATCH_HORIZON,
            patch_count=self.patch_count,
            fixed_stride=model_spec_for_descriptor(self.descriptor).patch_stride,
        )


def _load_router(
    identity: Mapping[str, Any],
    seed: int,
    *,
    device: str,
) -> tuple[Any, float, int]:
    auxiliary = identity["seeds"][str(seed)]["auxiliary"]
    descriptor = identity["descriptor"]
    policy = str(descriptor["policy"])
    runtime_policy = str(descriptor["runtime_policy"])
    expected_candidate_mask = (
        "none"
        if runtime_policy == "entropy_threshold_full"
        else "codepoint"
    )
    if (
        set(auxiliary)
        != {
            "calibration_stream_sha256",
            "candidate_mask",
            "kind",
            "maximum_patch_length",
            "policy",
            "policy_definition_sha256",
            "router_checkpoint_artifact_sha256",
            "router_checkpoint_path",
            "router_checkpoint_state_sha256",
            "router_config_sha256",
            "router_parameter_count",
            "router_report_artifact_sha256",
            "router_report_path",
            "router_training_stream_sha256",
            "seed",
            "threshold_cache_artifact_sha256",
            "threshold_cache_path",
            "threshold_diagnostics_artifact_sha256",
            "threshold_diagnostics_path",
            "threshold_nats",
        }
        or auxiliary.get("kind") != "entropy_router"
        or auxiliary.get("seed") != seed
        or auxiliary.get("policy") != policy
        or auxiliary.get("candidate_mask") != expected_candidate_mask
        or auxiliary.get("maximum_patch_length") != 24
        or auxiliary.get("router_parameter_count")
        != FINAL_ROUTER_PARAMETER_COUNT
        or auxiliary.get("router_config_sha256")
        != canonical_sha256(PHASE3_MODEL_SPEC.to_dict())
        or auxiliary.get("policy_definition_sha256")
        != entropy_policy_definition_sha256(policy)
    ):
        raise ValueError("actual-inference entropy bundle differs")
    for path_key, hash_key in (
        ("router_checkpoint_path", "router_checkpoint_artifact_sha256"),
        ("router_report_path", "router_report_artifact_sha256"),
        ("threshold_cache_path", "threshold_cache_artifact_sha256"),
        (
            "threshold_diagnostics_path",
            "threshold_diagnostics_artifact_sha256",
        ),
    ):
        if hash_file(Path(auxiliary[path_key])) != auxiliary[hash_key]:
            raise ValueError(f"actual-inference entropy artifact differs: {path_key}")
    threshold = auxiliary["threshold_nats"]
    if (
        not isinstance(threshold, (int, float))
        or not np.isfinite(float(threshold))
    ):
        raise ValueError("actual-inference entropy threshold differs")
    router = build_router(PHASE3_MODEL_SPEC, seed=seed)
    import torch

    router.load_state_dict(
        torch.load(
            Path(auxiliary["router_checkpoint_path"]),
            map_location="cpu",
            weights_only=True,
        )
    )
    if (
        parameter_count(router) != FINAL_ROUTER_PARAMETER_COUNT
        or state_dict_sha256(router)
        != auxiliary["router_checkpoint_state_sha256"]
    ):
        raise ValueError("actual-inference entropy state differs")
    router.to(device).eval()
    return router, float(threshold), int(auxiliary["maximum_patch_length"])


def load_actual_model(
    *,
    role: str,
    identity: Mapping[str, Any],
    seed: int,
    device: str,
) -> LoadedActualModel:
    if role not in {"candidate", "reference"} or seed not in FINAL_SEEDS:
        raise ValueError("actual-inference model role/seed differs")
    validate_final_model_identity(identity)
    evidence = identity["seeds"][str(seed)]
    checkpoint = evidence["checkpoint"]
    report = evidence["training_report"]
    if (
        hash_file(Path(checkpoint["path"])) != checkpoint["artifact_sha256"]
        or hash_file(Path(report["path"])) != report["artifact_sha256"]
    ):
        raise ValueError("actual-inference main artifact differs")
    spec = model_spec_for_descriptor(identity["descriptor"])
    model = build_main_model(
        spec,
        seed=seed,
        global_max_position_embeddings=ACTUAL_INFERENCE_GLOBAL_POSITION_LIMIT,
    )
    import torch

    model.load_state_dict(
        torch.load(
            Path(checkpoint["path"]),
            map_location="cpu",
            weights_only=True,
        )
    )
    if (
        parameter_count(model) != FINAL_MAIN_PARAMETER_COUNT
        or state_dict_sha256(model) != checkpoint["state_sha256"]
    ):
        raise ValueError("actual-inference main checkpoint state differs")
    model.to(device).eval()
    router = None
    threshold = None
    maximum = None
    requires_router = bool(identity["descriptor"]["requires_entropy_router"])
    if requires_router:
        router, threshold, maximum = _load_router(
            identity,
            seed,
            device=device,
        )
    elif evidence["auxiliary"] != {"kind": "none"}:
        raise ValueError("structural actual-inference bundle has auxiliary state")
    return LoadedActualModel(
        role=role,
        identity=identity,
        seed=seed,
        model=model,
        router=router,
        threshold_nats=threshold,
        maximum_patch_length=maximum,
    )


def release_actual_model(bundle: LoadedActualModel) -> None:
    import gc
    import torch

    bundle.model.to("cpu")
    if bundle.router is not None:
        bundle.router.to("cpu")
    del bundle.model
    del bundle.router
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def full_router_trace(bundle: LoadedActualModel, data: bytes) -> tuple[Any, np.ndarray]:
    if bundle.router is None:
        raise ValueError("structural model has no full router trace")
    import torch

    with torch.inference_mode():
        entropies, _, logits = bundle.router(
            torch.tensor(
                [list(data)],
                dtype=torch.long,
                device=next(bundle.router.parameters()).device,
            ),
            patch_size=None,
            use_cache=False,
        )
    return logits[0].float(), entropies[0].float().cpu().numpy()


def full_entropy_boundaries(
    bundle: LoadedActualModel,
    data: bytes,
    entropies: np.ndarray,
) -> tuple[int, ...]:
    if (
        bundle.threshold_nats is None
        or bundle.maximum_patch_length is None
        or entropies.shape != (len(data),)
    ):
        raise ValueError("full entropy boundary input differs")
    aligned = np.zeros(len(data), dtype=np.float32)
    aligned[1:] = entropies[:-1]
    candidate_mask = None
    if bundle.runtime_policy == "entropy_threshold_codepoint":
        candidate_mask = np.frombuffer(
            prefix_boundary_mask(data)[:-1],
            dtype=np.uint8,
        )
    return entropy_threshold_boundaries(
        aligned,
        bundle.threshold_nats,
        candidate_mask=candidate_mask,
        maximum_patch_length=bundle.maximum_patch_length,
    )


def full_main_logits(
    bundle: LoadedActualModel,
    data: bytes,
    boundaries: tuple[int, ...],
) -> Any:
    import torch

    patches = padded_hf_patch_matrix([boundaries], len(data))
    with torch.inference_mode():
        return bundle.model(
            input_ids=torch.tensor(
                [list(data)],
                dtype=torch.long,
                device=next(bundle.model.parameters()).device,
            ),
            patch_lengths=torch.from_numpy(
                patches.astype(np.int64, copy=False)
            ).to(next(bundle.model.parameters()).device),
            use_cache=False,
        ).logits[0].float()
