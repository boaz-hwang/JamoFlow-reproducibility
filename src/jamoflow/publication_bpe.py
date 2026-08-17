"""Reversible byte-BPE and parameter-matched token Transformer controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import unicodedata
from typing import Iterable

from .publication_protocol import (
    PUBLICATION_BPE_INITIAL_ALPHABET_SIZE,
    PUBLICATION_BPE_STRESS_VOCABULARY_SIZE,
    PUBLICATION_BPE_VOCABULARY_SIZE,
    PUBLICATION_BPE_VOCABULARY_CANDIDATES,
    PUBLICATION_CONTEXT_BYTES,
)
from .publication_scale import (
    PUBLICATION_EXPECTED_PARAMETERS,
    PUBLICATION_FAMILY_EXPECTED_PARAMETERS,
)
from .utf8 import (
    StrictUtf8TokenTransitions,
    compile_strict_utf8_token_transitions,
)


PINNED_TOKENIZERS_VERSION = "0.22.2"
PUBLICATION_BPE_MINIMUM_FREQUENCY = 2
PUBLICATION_PARAMETER_TOLERANCE = 0.01
PUBLICATION_BPE_PARAMETER_GRID_HIDDEN_SIZES = tuple(range(384, 769, 32))
PUBLICATION_BPE_PARAMETER_GRID_INTERMEDIATE_MULTIPLE = 64
PUBLICATION_BPE_PARAMETER_GRID_LAYER_COUNT = 12
PUBLICATION_BPE_PARAMETER_GRID_HEAD_COUNTS = tuple(range(4, 17))
PUBLICATION_BPE_PARAMETER_GRID_HEAD_DIMENSION_RANGE = (64, 128)
PUBLICATION_BPE_PARAMETER_GRID_FFN_RATIO_RANGE = (2.5, 4.0)
PUBLICATION_BPE_PARAMETER_GRID_PREFERRED_HEAD_DIMENSION = 64


@dataclass(frozen=True, slots=True)
class TokenTransformerSpec:
    target_millions: int
    vocabulary_size: int
    maximum_positions: int
    hidden_size: int
    intermediate_size: int
    layers: int
    attention_heads: int
    key_value_heads: int
    expected_parameters: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def analytical_token_transformer_parameters(
    *,
    vocabulary_size: int,
    hidden_size: int,
    intermediate_size: int,
    layers: int,
) -> int:
    """Count the tied-embedding, bias-free Llama graph analytically."""

    if min(vocabulary_size, hidden_size, intermediate_size, layers) <= 0:
        raise ValueError("token Transformer dimensions must be positive")
    embedding = vocabulary_size * hidden_size
    attention = 4 * hidden_size * hidden_size
    feed_forward = 3 * hidden_size * intermediate_size
    layer_norms = 2 * hidden_size
    final_norm = hidden_size
    return embedding + layers * (attention + feed_forward + layer_norms) + final_norm


def derive_parameter_matched_bpe_spec(
    target_millions: int,
    vocabulary_size: int,
) -> TokenTransformerSpec:
    """Select a graph using parameter count only, never quality or timing."""

    if target_millions not in PUBLICATION_EXPECTED_PARAMETERS:
        raise ValueError("unknown publication BPE target")
    if vocabulary_size != PUBLICATION_BPE_VOCABULARY_SIZE:
        raise ValueError("only the primary 32K BPE is total-parameter matched")
    target_parameters = PUBLICATION_EXPECTED_PARAMETERS[target_millions]
    candidates: list[
        tuple[tuple[int, int, int, int, int], TokenTransformerSpec]
    ] = []
    minimum_head_dimension, maximum_head_dimension = (
        PUBLICATION_BPE_PARAMETER_GRID_HEAD_DIMENSION_RANGE
    )
    minimum_ffn_ratio, maximum_ffn_ratio = (
        PUBLICATION_BPE_PARAMETER_GRID_FFN_RATIO_RANGE
    )
    for hidden_size in PUBLICATION_BPE_PARAMETER_GRID_HIDDEN_SIZES:
        for head_count in PUBLICATION_BPE_PARAMETER_GRID_HEAD_COUNTS:
            if hidden_size % head_count:
                continue
            head_dimension = hidden_size // head_count
            if not (
                minimum_head_dimension
                <= head_dimension
                <= maximum_head_dimension
            ):
                continue
            for intermediate_size in range(
                PUBLICATION_BPE_PARAMETER_GRID_INTERMEDIATE_MULTIPLE,
                4_097,
                PUBLICATION_BPE_PARAMETER_GRID_INTERMEDIATE_MULTIPLE,
            ):
                ratio = intermediate_size / hidden_size
                if not minimum_ffn_ratio <= ratio <= maximum_ffn_ratio:
                    continue
                parameters = analytical_token_transformer_parameters(
                    vocabulary_size=vocabulary_size,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    layers=PUBLICATION_BPE_PARAMETER_GRID_LAYER_COUNT,
                )
                spec = TokenTransformerSpec(
                    target_millions=target_millions,
                    vocabulary_size=vocabulary_size,
                    maximum_positions=PUBLICATION_CONTEXT_BYTES,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    layers=PUBLICATION_BPE_PARAMETER_GRID_LAYER_COUNT,
                    attention_heads=head_count,
                    key_value_heads=head_count,
                    expected_parameters=parameters,
                )
                rank = (
                    abs(parameters - target_parameters),
                    abs(
                        head_dimension
                        - PUBLICATION_BPE_PARAMETER_GRID_PREFERRED_HEAD_DIMENSION
                    ),
                    hidden_size,
                    intermediate_size,
                    -head_count,
                )
                candidates.append((rank, spec))
    if not candidates:
        raise RuntimeError("publication BPE parameter grid is empty")
    return min(candidates, key=lambda item: item[0])[1]


def body_matched_bpe_spec(
    primary_spec: TokenTransformerSpec,
    vocabulary_size: int,
) -> TokenTransformerSpec:
    """Change only vocabulary/output rows while holding the body fixed."""

    if vocabulary_size not in PUBLICATION_BPE_VOCABULARY_CANDIDATES:
        raise ValueError("unknown publication BPE vocabulary control")
    parameters = analytical_token_transformer_parameters(
        vocabulary_size=vocabulary_size,
        hidden_size=primary_spec.hidden_size,
        intermediate_size=primary_spec.intermediate_size,
        layers=primary_spec.layers,
    )
    return replace(
        primary_spec,
        vocabulary_size=vocabulary_size,
        expected_parameters=parameters,
    )


# The ordinary 32K graph is total-parameter matched to the byte-latent model.
PUBLICATION_BPE_MODEL_SPECS = {
    target: derive_parameter_matched_bpe_spec(
        target,
        PUBLICATION_BPE_VOCABULARY_SIZE,
    )
    for target in PUBLICATION_EXPECTED_PARAMETERS
}
# The 16K stress control holds the 32K Transformer body exactly fixed so that
# vocabulary/output rows are the only graph change.
PUBLICATION_BPE_STRESS_MODEL_SPECS = {
    target: body_matched_bpe_spec(spec, PUBLICATION_BPE_STRESS_VOCABULARY_SIZE)
    for target, spec in PUBLICATION_BPE_MODEL_SPECS.items()
}
PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY = {
    PUBLICATION_BPE_STRESS_VOCABULARY_SIZE: PUBLICATION_BPE_STRESS_MODEL_SPECS,
    PUBLICATION_BPE_VOCABULARY_SIZE: PUBLICATION_BPE_MODEL_SPECS,
}


@dataclass(frozen=True, slots=True)
class ByteBPEAudit:
    tokenizers_version: str
    vocabulary_size: int
    expected_vocabulary_size: int
    full_byte_alphabet: bool
    normalizer_absent: bool
    pretokenizer_exact: bool
    decoder_exact: bool
    no_added_tokens: bool
    digit_labels_single_token: bool
    roundtrip_identity: bool
    null_byte_padding_token_id: int
    tokenizer_json_sha256: str
    overall_pass: bool

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ByteBPEPromptBoundaryAudit:
    prompt_bytes: int
    continuation_bytes: int
    prompt_tokens: int
    continuation_tokens_separate: int
    joint_tokens: int
    prompt_roundtrip_identity: bool
    continuation_roundtrip_identity: bool
    separate_concatenation_roundtrip_identity: bool
    joint_roundtrip_identity: bool
    joint_has_exact_prompt_token_boundary: bool
    separate_and_joint_token_ids_identical: bool
    primary_replay_semantics: str
    overall_pass: bool

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ByteBPEReplayPlan:
    """Prepared private token ids plus content-free boundary diagnostics."""

    prompt_token_ids: tuple[int, ...]
    continuation_token_ids: tuple[int, ...]
    audit: ByteBPEPromptBoundaryAudit


def _tokenizer_ids(tokenizer, text: str) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids
    )


def _tokenizer_decode(tokenizer, token_ids: tuple[int, ...]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
    )


def _gpt2_byte_to_unicode() -> dict[int, str]:
    direct = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    byte_values = list(direct)
    unicode_values = list(direct)
    extra = 0
    for value in range(256):
        if value in direct:
            continue
        byte_values.append(value)
        unicode_values.append(256 + extra)
        extra += 1
    return {
        value: chr(codepoint)
        for value, codepoint in zip(byte_values, unicode_values, strict=True)
    }


def byte_bpe_token_bytes(tokenizer) -> tuple[bytes, ...]:
    """Recover raw bytes from GPT-2 ByteLevel token strings without decoding."""

    byte_to_unicode = _gpt2_byte_to_unicode()
    unicode_to_byte = {character: value for value, character in byte_to_unicode.items()}
    vocabulary_size = int(tokenizer.get_vocab_size(with_added_tokens=True))
    if vocabulary_size <= 0 or len(unicode_to_byte) != 256:
        raise ValueError("byte-BPE vocabulary or alphabet is invalid")
    output: list[bytes] = []
    for token_id in range(vocabulary_size):
        token = tokenizer.id_to_token(token_id)
        if not isinstance(token, str) or not token:
            raise ValueError("byte-BPE token IDs must be contiguous byte strings")
        try:
            values = bytes(unicode_to_byte[character] for character in token)
        except KeyError as error:
            raise ValueError("byte-BPE token is outside the ByteLevel alphabet") from error
        output.append(values)
    return tuple(output)


def compile_byte_bpe_utf8_transitions(tokenizer) -> StrictUtf8TokenTransitions:
    """Compile the shared strict UTF-8 mask for every ByteLevel BPE token."""

    return compile_strict_utf8_token_transitions(byte_bpe_token_bytes(tokenizer))


def prepare_byte_bpe_replay(
    tokenizer,
    prompt: str,
    continuation: str,
) -> ByteBPEReplayPlan:
    """Freeze API-realistic replay tokens and audit the raw prompt boundary.

    A deployed tokenizer cannot revise prompt tokens after continuation bytes
    arrive.  Controlled replay therefore encodes the prompt and held-out
    continuation separately.  Joint encoding is retained only as a sensitivity
    audit because a BPE merge may straddle the raw prompt boundary.
    """

    if not isinstance(prompt, str) or not prompt:
        raise ValueError("BPE replay prompt must be non-empty text")
    if not isinstance(continuation, str) or not continuation:
        raise ValueError("BPE replay continuation must be non-empty text")
    prompt_ids = _tokenizer_ids(tokenizer, prompt)
    continuation_ids = _tokenizer_ids(tokenizer, continuation)
    joint_ids = _tokenizer_ids(tokenizer, prompt + continuation)
    if not prompt_ids or not continuation_ids or not joint_ids:
        raise ValueError("BPE replay produced an empty token sequence")

    prompt_roundtrip = _tokenizer_decode(tokenizer, prompt_ids) == prompt
    continuation_roundtrip = (
        _tokenizer_decode(tokenizer, continuation_ids) == continuation
    )
    separate_roundtrip = (
        _tokenizer_decode(tokenizer, prompt_ids + continuation_ids)
        == prompt + continuation
    )
    joint_roundtrip = (
        _tokenizer_decode(tokenizer, joint_ids) == prompt + continuation
    )
    joint_boundary = any(
        _tokenizer_decode(tokenizer, joint_ids[:index]) == prompt
        for index in range(1, len(joint_ids) + 1)
    )
    stable = joint_ids == prompt_ids + continuation_ids
    overall = bool(
        prompt_roundtrip
        and continuation_roundtrip
        and separate_roundtrip
        and joint_roundtrip
    )
    return ByteBPEReplayPlan(
        prompt_token_ids=prompt_ids,
        continuation_token_ids=continuation_ids,
        audit=ByteBPEPromptBoundaryAudit(
            prompt_bytes=len(prompt.encode("utf-8")),
            continuation_bytes=len(continuation.encode("utf-8")),
            prompt_tokens=len(prompt_ids),
            continuation_tokens_separate=len(continuation_ids),
            joint_tokens=len(joint_ids),
            prompt_roundtrip_identity=prompt_roundtrip,
            continuation_roundtrip_identity=continuation_roundtrip,
            separate_concatenation_roundtrip_identity=separate_roundtrip,
            joint_roundtrip_identity=joint_roundtrip,
            joint_has_exact_prompt_token_boundary=joint_boundary,
            separate_and_joint_token_ids_identical=stable,
            primary_replay_semantics=(
                "frozen prompt tokens plus separately encoded continuation; "
                "joint encoding is a prompt-boundary sensitivity only"
            ),
            overall_pass=overall,
        ),
    )


def publication_bpe_spec(
    target_millions: int,
    *,
    vocabulary_size: int = PUBLICATION_BPE_VOCABULARY_SIZE,
) -> TokenTransformerSpec:
    try:
        return PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY[vocabulary_size][
            target_millions
        ]
    except KeyError as error:
        raise ValueError("unknown publication BPE target or vocabulary") from error


def parameter_match_fraction(
    target_millions: int,
    *,
    vocabulary_size: int = PUBLICATION_BPE_VOCABULARY_SIZE,
) -> float:
    spec = publication_bpe_spec(
        target_millions,
        vocabulary_size=vocabulary_size,
    )
    target = PUBLICATION_EXPECTED_PARAMETERS[target_millions]
    return abs(spec.expected_parameters - target) / target


def train_byte_bpe_tokenizer(
    texts: Iterable[str],
    *,
    vocabulary_size: int = PUBLICATION_BPE_VOCABULARY_SIZE,
    minimum_frequency: int = PUBLICATION_BPE_MINIMUM_FREQUENCY,
):
    """Train the fixed reversible tokenizer without added or special tokens."""

    if vocabulary_size < PUBLICATION_BPE_INITIAL_ALPHABET_SIZE:
        raise ValueError("BPE vocabulary cannot be smaller than the byte alphabet")
    if minimum_frequency <= 0:
        raise ValueError("BPE minimum frequency must be positive")
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    def validated_texts() -> Iterable[str]:
        count = 0
        for text in texts:
            if not isinstance(text, str) or not text:
                raise ValueError("BPE training rows must be non-empty text")
            if unicodedata.normalize("NFC", text) != text:
                raise ValueError("BPE training source must preserve pinned NFC bytes")
            count += 1
            yield text
        if count == 0:
            raise ValueError("BPE training requires at least one row")

    tokenizer = Tokenizer(
        models.BPE(
            unk_token=None,
            dropout=None,
            fuse_unk=False,
            byte_fallback=False,
        )
    )
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=True,
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocabulary_size,
        min_frequency=minimum_frequency,
        show_progress=False,
        special_tokens=[],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
    )
    tokenizer.train_from_iterator(validated_texts(), trainer=trainer)
    return tokenizer


def audit_byte_bpe_tokenizer(
    tokenizer,
    audit_texts: Iterable[str],
    *,
    expected_vocabulary_size: int = PUBLICATION_BPE_VOCABULARY_SIZE,
) -> ByteBPEAudit:
    """Validate the artifact before model construction or any evaluation."""

    import tokenizers
    from tokenizers import pre_tokenizers

    configuration = json.loads(tokenizer.to_str())
    pretokenizer = configuration.get("pre_tokenizer")
    decoder = configuration.get("decoder")
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    alphabet = set(pre_tokenizers.ByteLevel.alphabet())
    full_alphabet = len(alphabet) == PUBLICATION_BPE_INITIAL_ALPHABET_SIZE and (
        alphabet <= set(vocabulary)
    )
    pretokenizer_exact = bool(
        isinstance(pretokenizer, dict)
        and pretokenizer.get("type") == "ByteLevel"
        and pretokenizer.get("add_prefix_space") is False
        and pretokenizer.get("use_regex") is True
    )
    decoder_exact = bool(
        isinstance(decoder, dict) and decoder.get("type") == "ByteLevel"
    )
    added_tokens = configuration.get("added_tokens")
    no_added_tokens = isinstance(added_tokens, list) and not added_tokens
    digit_labels_single = all(
        len(tokenizer.encode(str(digit), add_special_tokens=False).ids) == 1
        for digit in range(7)
    )
    audit_rows = tuple(audit_texts)
    if not audit_rows or any(not isinstance(text, str) for text in audit_rows):
        raise ValueError("tokenizer audit requires text rows")
    roundtrip = all(
        tokenizer.decode(
            tokenizer.encode(text, add_special_tokens=False).ids,
            skip_special_tokens=False,
        )
        == text
        for text in audit_rows
    )
    null_ids = tokenizer.encode("\x00", add_special_tokens=False).ids
    null_id = null_ids[0] if len(null_ids) == 1 else -1
    serialized = tokenizer.to_str(pretty=False).encode("utf-8")
    version_exact = tokenizers.__version__ == PINNED_TOKENIZERS_VERSION
    vocabulary_exact = tokenizer.get_vocab_size(with_added_tokens=True) == (
        expected_vocabulary_size
    )
    checks = (
        version_exact,
        vocabulary_exact,
        full_alphabet,
        configuration.get("normalizer") is None,
        pretokenizer_exact,
        decoder_exact,
        no_added_tokens,
        digit_labels_single,
        roundtrip,
        null_id >= 0,
    )
    return ByteBPEAudit(
        tokenizers_version=tokenizers.__version__,
        vocabulary_size=tokenizer.get_vocab_size(with_added_tokens=True),
        expected_vocabulary_size=expected_vocabulary_size,
        full_byte_alphabet=full_alphabet,
        normalizer_absent=configuration.get("normalizer") is None,
        pretokenizer_exact=pretokenizer_exact,
        decoder_exact=decoder_exact,
        no_added_tokens=no_added_tokens,
        digit_labels_single_token=digit_labels_single,
        roundtrip_identity=roundtrip,
        null_byte_padding_token_id=null_id,
        tokenizer_json_sha256=hashlib.sha256(serialized).hexdigest(),
        overall_pass=all(checks),
    )


def build_publication_bpe_model(
    target_millions: int,
    *,
    seed: int,
    padding_token_id: int,
    vocabulary_size: int = PUBLICATION_BPE_VOCABULARY_SIZE,
):
    """Build the preregistered cached decoder-only token baseline."""

    if (
        seed < 0
        or padding_token_id < 0
        or padding_token_id >= vocabulary_size
    ):
        raise ValueError("model seed and padding token id must be nonnegative")
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    spec = publication_bpe_spec(
        target_millions,
        vocabulary_size=vocabulary_size,
    )
    torch.manual_seed(seed)
    configuration = LlamaConfig(
        vocab_size=spec.vocabulary_size,
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
        num_hidden_layers=spec.layers,
        num_attention_heads=spec.attention_heads,
        num_key_value_heads=spec.key_value_heads,
        max_position_embeddings=spec.maximum_positions,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=True,
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=padding_token_id,
        use_cache=True,
    )
    model = LlamaForCausalLM(configuration)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != spec.expected_parameters:
        raise RuntimeError("publication BPE parameter count drifted")
    observed_match = abs(
        actual_parameters - PUBLICATION_EXPECTED_PARAMETERS[target_millions]
    ) / PUBLICATION_EXPECTED_PARAMETERS[target_millions]
    if not math.isclose(
        parameter_match_fraction(
            target_millions,
            vocabulary_size=vocabulary_size,
        ),
        observed_match,
        rel_tol=0,
        abs_tol=0,
    ):
        raise RuntimeError("publication BPE parameter match audit failed")
    return model


def validate_publication_bpe_specs() -> None:
    if set(PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY) != set(
        PUBLICATION_BPE_VOCABULARY_CANDIDATES
    ):
        raise ValueError("BPE vocabulary grid drifted")
    for vocabulary_size, specs in (
        PUBLICATION_BPE_MODEL_SPECS_BY_VOCABULARY.items()
    ):
        if set(specs) != set(PUBLICATION_EXPECTED_PARAMETERS):
            raise ValueError("BPE model scale grid drifted")
        for target, spec in specs.items():
            analytical_parameters = analytical_token_transformer_parameters(
                vocabulary_size=spec.vocabulary_size,
                hidden_size=spec.hidden_size,
                intermediate_size=spec.intermediate_size,
                layers=spec.layers,
            )
            if (
                spec.target_millions != target
                or spec.vocabulary_size != vocabulary_size
                or spec.maximum_positions != PUBLICATION_CONTEXT_BYTES
                or spec.hidden_size % spec.attention_heads
                or spec.key_value_heads != spec.attention_heads
                or spec.expected_parameters != analytical_parameters
            ):
                raise ValueError("publication BPE model spec is invalid")
            family = (
                "byte_bpe_16000_body_matched"
                if vocabulary_size == PUBLICATION_BPE_STRESS_VOCABULARY_SIZE
                else "byte_bpe_32000"
            )
            if (
                spec.expected_parameters
                != PUBLICATION_FAMILY_EXPECTED_PARAMETERS[target][family]
            ):
                raise ValueError("campaign BPE parameter identity drifted")
            if vocabulary_size == PUBLICATION_BPE_VOCABULARY_SIZE:
                if (
                    spec
                    != derive_parameter_matched_bpe_spec(target, vocabulary_size)
                    or parameter_match_fraction(
                        target,
                        vocabulary_size=vocabulary_size,
                    )
                    > PUBLICATION_PARAMETER_TOLERANCE
                ):
                    raise ValueError("32K BPE parameter match drifted")
            else:
                primary = PUBLICATION_BPE_MODEL_SPECS[target]
                if spec != body_matched_bpe_spec(primary, vocabulary_size):
                    raise ValueError("16K BPE body match drifted")


validate_publication_bpe_specs()
