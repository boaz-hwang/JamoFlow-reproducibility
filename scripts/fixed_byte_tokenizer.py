"""Fixed-vocabulary byte tokenizers with explicit segmentation semantics.

The Hugging Face ``tokenizers`` package supplies the optimized trie/Viterbi
engines, while this module fixes the alphabet and serialization contract.  It
is deliberately vocabulary-construction agnostic: the same ordered byte
pieces can be applied with either paper-style left-most-longest matching or a
minimum-token dynamic program.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Literal

from jamoflow.publication_bpe import _gpt2_byte_to_unicode, byte_bpe_token_bytes


Segmentation = Literal["leftmost_longest", "minimum_token_dp"]
BYTE_ALPHABET_SIZE = 256
MAXIMUM_PIECE_BYTES = 48


@dataclass(frozen=True, slots=True)
class FixedByteTokenizerAudit:
    segmentation: str
    vocabulary_size: int
    full_byte_alphabet: bool
    byte_alphabet_ids_are_identity: bool
    unique_pieces: bool
    maximum_piece_bytes: int
    exact_utf8_roundtrip: bool
    tokenizer_json_sha256: str
    ordered_pieces_sha256: str
    overall_pass: bool

    def to_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _EncodingResult:
    """Small compatibility surface used by the opportunity evaluator."""

    ids: tuple[int, ...]


class LeftmostLongestByteTokenizer:
    """Exact bounded-trie left-most-longest tokenizer over UTF-8 bytes.

    Hugging Face WordPiece makes the same greedy choice, but on deliberately
    unsplit long documents it searches successively shorter whole substrings
    and becomes effectively quadratic.  This adapter traverses a byte trie at
    most ``maximum_piece_bytes`` bytes from every emitted-token boundary and
    delegates vocabulary, serialization, and decoding to the canonical HF
    tokenizer.
    """

    def __init__(
        self,
        *,
        base_tokenizer,
        pieces: Sequence[bytes],
        maximum_piece_bytes: int,
    ) -> None:
        values = validate_ordered_byte_pieces(
            pieces,
            maximum_piece_bytes=maximum_piece_bytes,
        )
        children: list[dict[int, int]] = [{}]
        terminals: list[int | None] = [None]
        for token_id, piece in enumerate(values):
            node = 0
            for value in piece:
                child = children[node].get(value)
                if child is None:
                    child = len(children)
                    children[node][value] = child
                    children.append({})
                    terminals.append(None)
                node = child
            if terminals[node] is not None:
                raise ValueError("left-most-longest trie contains a duplicate piece")
            terminals[node] = token_id
        self._base_tokenizer = base_tokenizer
        self._children = tuple(children)
        self._terminals = tuple(terminals)
        self._maximum_piece_bytes = maximum_piece_bytes

    def encode_raw_bytes(self, raw: bytes) -> tuple[int, ...]:
        if not isinstance(raw, bytes):
            raise TypeError("left-most-longest input must be bytes")
        output: list[int] = []
        position = 0
        while position < len(raw):
            node = 0
            best_id: int | None = None
            best_end = position
            stop = min(len(raw), position + self._maximum_piece_bytes)
            for end in range(position, stop):
                child = self._children[node].get(raw[end])
                if child is None:
                    break
                node = child
                token_id = self._terminals[node]
                if token_id is not None:
                    best_id = token_id
                    best_end = end + 1
            if best_id is None or best_end <= position:
                raise AssertionError("full byte fallback failed in greedy trie")
            output.append(best_id)
            position = best_end
        return tuple(output)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> _EncodingResult:
        if not isinstance(text, str) or add_special_tokens:
            raise ValueError("left-most-longest adapter accepts plain text without specials")
        return _EncodingResult(self.encode_raw_bytes(text.encode("utf-8")))

    def __getattr__(self, name: str):
        return getattr(self._base_tokenizer, name)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def ordered_pieces_sha256(pieces: Sequence[bytes]) -> str:
    if not pieces:
        raise ValueError("fixed byte vocabulary must not be empty")
    digest = hashlib.sha256(b"JamoFlow/fixed-byte-pieces/v1\0")
    digest.update(len(pieces).to_bytes(8, "big"))
    for piece in pieces:
        if not isinstance(piece, bytes) or not piece:
            raise ValueError("fixed byte pieces must be non-empty bytes")
        digest.update(len(piece).to_bytes(8, "big"))
        digest.update(piece)
    return digest.hexdigest()


def validate_ordered_byte_pieces(
    pieces: Sequence[bytes],
    *,
    expected_vocabulary_size: int | None = None,
    maximum_piece_bytes: int = MAXIMUM_PIECE_BYTES,
) -> tuple[bytes, ...]:
    values = tuple(pieces)
    if expected_vocabulary_size is not None and len(values) != expected_vocabulary_size:
        raise ValueError("fixed byte vocabulary size differs")
    if len(values) < BYTE_ALPHABET_SIZE:
        raise ValueError("fixed byte vocabulary is smaller than byte fallback")
    if maximum_piece_bytes <= 0:
        raise ValueError("maximum fixed byte piece length must be positive")
    if values[:BYTE_ALPHABET_SIZE] != tuple(bytes((value,)) for value in range(256)):
        raise ValueError("fixed byte vocabulary must begin with identity byte fallback")
    if any(not isinstance(piece, bytes) or not piece for piece in values):
        raise ValueError("fixed byte vocabulary contains an invalid piece")
    if len(set(values)) != len(values):
        raise ValueError("fixed byte vocabulary pieces must be unique")
    if max(map(len, values)) > maximum_piece_bytes:
        raise ValueError("fixed byte vocabulary piece exceeds the byte limit")
    return values


def byte_piece_to_level_string(piece: bytes) -> str:
    if not isinstance(piece, bytes) or not piece:
        raise ValueError("byte piece must be non-empty")
    byte_to_unicode = _gpt2_byte_to_unicode()
    return "".join(byte_to_unicode[value] for value in piece)


def build_fixed_byte_tokenizer(
    pieces: Sequence[bytes],
    *,
    segmentation: Segmentation,
    maximum_piece_bytes: int = MAXIMUM_PIECE_BYTES,
):
    """Build an optimized reversible tokenizer for one ordered byte vocab.

    ``leftmost_longest`` uses WordPiece's maximum-munch trie with an
    unreachable byte-zero fallback as its required ``unk_token``.  The full
    byte alphabet makes that fallback unreachable for any ByteLevel string.

    ``minimum_token_dp`` uses an equal score of ``-1`` for every piece.  The
    Unigram Viterbi objective then maximizes ``-number_of_tokens``.  This is a
    segmentation ablation, not the frozen-vocabulary algorithm claimed by the
    Length-MAX paper.
    """

    values = validate_ordered_byte_pieces(
        pieces,
        maximum_piece_bytes=maximum_piece_bytes,
    )
    if segmentation not in ("leftmost_longest", "minimum_token_dp"):
        raise ValueError("unknown fixed byte segmentation")

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    token_strings = tuple(byte_piece_to_level_string(piece) for piece in values)
    if segmentation == "leftmost_longest":
        vocabulary = {token: index for index, token in enumerate(token_strings)}
        model = models.WordPiece(
            vocab=vocabulary,
            unk_token=token_strings[0],
            continuing_subword_prefix="",
            max_input_chars_per_word=max(1_000_000, maximum_piece_bytes),
        )
    else:
        model = models.Unigram(
            [(token, -1.0) for token in token_strings],
            unk_id=None,
            byte_fallback=False,
        )
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    tokenizer.decoder = decoders.ByteLevel()
    recovered = byte_bpe_token_bytes(tokenizer)
    if recovered != values:
        raise AssertionError("fixed byte tokenizer changed ordered vocabulary")
    if segmentation == "leftmost_longest":
        return LeftmostLongestByteTokenizer(
            base_tokenizer=tokenizer,
            pieces=values,
            maximum_piece_bytes=maximum_piece_bytes,
        )
    return tokenizer


def build_scored_byte_unigram_tokenizer(
    pieces: Sequence[bytes],
    scores: Sequence[float],
    *,
    maximum_piece_bytes: int = MAXIMUM_PIECE_BYTES,
):
    """Build a reversible Viterbi tokenizer from fixed pieces and scores."""

    values = validate_ordered_byte_pieces(
        pieces,
        maximum_piece_bytes=maximum_piece_bytes,
    )
    numeric_scores = tuple(float(value) for value in scores)
    if len(numeric_scores) != len(values) or not all(
        math.isfinite(value) for value in numeric_scores
    ):
        raise ValueError("fixed byte Unigram scores differ")

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    token_strings = tuple(byte_piece_to_level_string(piece) for piece in values)
    tokenizer = Tokenizer(
        models.Unigram(
            list(zip(token_strings, numeric_scores, strict=True)),
            unk_id=None,
            byte_fallback=False,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=False,
    )
    tokenizer.decoder = decoders.ByteLevel()
    if byte_bpe_token_bytes(tokenizer) != values:
        raise AssertionError("scored byte Unigram changed ordered vocabulary")
    return tokenizer


def encode_raw_bytes(tokenizer, raw: bytes) -> tuple[int, ...]:
    """Encode arbitrary bytes directly, including invalid UTF-8 sequences."""

    if not isinstance(raw, bytes):
        raise TypeError("raw byte input must be bytes")
    if not raw:
        return ()
    direct = getattr(tokenizer, "encode_raw_bytes", None)
    if direct is not None:
        return tuple(int(value) for value in direct(raw))
    transformed = byte_piece_to_level_string(raw)
    return tuple(int(token.id) for token in tokenizer.model.tokenize(transformed))


def decode_ids_to_bytes(tokenizer, token_ids: Sequence[int]) -> bytes:
    pieces = byte_bpe_token_bytes(tokenizer)
    output = bytearray()
    for token_id in token_ids:
        index = int(token_id)
        if not 0 <= index < len(pieces):
            raise ValueError("fixed byte token id is outside the vocabulary")
        output.extend(pieces[index])
    return bytes(output)


def audit_fixed_byte_tokenizer(
    tokenizer,
    *,
    pieces: Sequence[bytes],
    segmentation: Segmentation,
    utf8_samples: Sequence[str],
) -> FixedByteTokenizerAudit:
    values = validate_ordered_byte_pieces(pieces)
    recovered = byte_bpe_token_bytes(tokenizer)
    full_alphabet = set(recovered[:256]) == {bytes((value,)) for value in range(256)}
    identity_ids = recovered[:256] == tuple(bytes((value,)) for value in range(256))
    unique = len(set(recovered)) == len(recovered)
    byte_roundtrip = all(
        decode_ids_to_bytes(tokenizer, encode_raw_bytes(tokenizer, sample)) == sample
        for sample in (
            bytes(range(256)),
            bytes(reversed(range(256))),
            b"\x00\xff\x80\xc0\xaf\xf5\n\t ",
        )
    )
    utf8_roundtrip = all(
        tokenizer.decode(
            tokenizer.encode(text, add_special_tokens=False).ids,
            skip_special_tokens=False,
        )
        == text
        for text in utf8_samples
    )
    tokenizer_json = tokenizer.to_str(pretty=False).encode("utf-8")
    overall = bool(
        recovered == values
        and full_alphabet
        and identity_ids
        and unique
        and byte_roundtrip
        and utf8_roundtrip
    )
    return FixedByteTokenizerAudit(
        segmentation=segmentation,
        vocabulary_size=len(recovered),
        full_byte_alphabet=full_alphabet,
        byte_alphabet_ids_are_identity=identity_ids,
        unique_pieces=unique,
        maximum_piece_bytes=max(map(len, recovered)),
        exact_utf8_roundtrip=utf8_roundtrip and byte_roundtrip,
        tokenizer_json_sha256=hashlib.sha256(tokenizer_json).hexdigest(),
        ordered_pieces_sha256=ordered_pieces_sha256(values),
        overall_pass=overall,
    )


def unigram_piece_scores(tokenizer) -> tuple[float, ...]:
    """Read ordered Unigram scores from canonical tokenizer JSON."""

    payload = json.loads(tokenizer.to_str(pretty=False))
    model = payload.get("model")
    if not isinstance(model, dict) or model.get("type") != "Unigram":
        raise ValueError("tokenizer is not a Unigram model")
    vocabulary = model.get("vocab")
    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError("Unigram vocabulary is absent")
    scores: list[float] = []
    for row in vocabulary:
        if (
            not isinstance(row, list)
            or len(row) != 2
            or not isinstance(row[0], str)
            or not isinstance(row[1], (int, float))
            or not math.isfinite(float(row[1]))
        ):
            raise ValueError("Unigram vocabulary row differs")
        scores.append(float(row[1]))
    return tuple(scores)


def canonical_fixed_tokenizer_descriptor(
    *,
    pieces: Sequence[bytes],
    segmentation: Segmentation,
) -> dict[str, Any]:
    values = validate_ordered_byte_pieces(pieces)
    payload = {
        "byte_alphabet": "identity_ids_0_through_255",
        "kind": "fixed_byte_tokenizer_v1",
        "maximum_piece_bytes": max(map(len, values)),
        "ordered_pieces_sha256": ordered_pieces_sha256(values),
        "segmentation": segmentation,
        "vocabulary_size": len(values),
    }
    return {**payload, "descriptor_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()}
