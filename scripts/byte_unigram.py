"""Deterministic byte-exact Unigram vocabulary construction.

SentencePiece is used only as a deterministic vocabulary/score estimator over
GPT-2's whitespace-free byte-to-Unicode alphabet.  The learned pieces are then
reordered behind identity byte IDs 0..255 and installed in the project's
``tokenizers`` runtime, so model classes and byte semantics remain identical to
the BPE control.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import hashlib
import io
import math
import struct

from fixed_byte_tokenizer import (
    MAXIMUM_PIECE_BYTES,
    build_scored_byte_unigram_tokenizer,
    ordered_pieces_sha256,
    validate_ordered_byte_pieces,
)
from jamoflow.publication_bpe import _gpt2_byte_to_unicode, byte_bpe_token_bytes


PINNED_SENTENCEPIECE_VERSION = "0.2.1"
DEFAULT_VOCABULARY_SIZE = 2_048
SENTENCEPIECE_UNKNOWN_PIECE = "<unk>"
SENTENCEPIECE_UNKNOWN_COUNT = 1
SEED_SENTENCEPIECE_SIZE = 1_000_000
SHRINKING_FACTOR = 0.75
SUB_ITERATIONS = 2
MAXIMUM_SENTENCE_BYTES = 262_144


@dataclass(frozen=True, slots=True)
class ByteUnigramTrainingMetadata:
    schema_version: int
    vocabulary_size: int
    source_document_count: int
    synthetic_single_byte_rows: int
    maximum_piece_bytes: int
    sentencepiece_version: str
    sentencepiece_model_sha256: str
    ordered_pieces_sha256: str
    ordered_scores_sha256: str
    tokenizer_json_sha256: str
    full_byte_alphabet: bool
    exact_vocabulary_size: bool
    byte_exact_sample_roundtrip: bool
    missing_single_bytes_inserted: int
    learned_nonbyte_pieces_retained: int
    learned_pieces_dropped_for_fallback: int
    inserted_byte_fallback_score: float | None
    deterministic_trainer_contract: dict[str, int | float | bool | str]
    overall_pass: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def byte_level_alphabet() -> tuple[str, ...]:
    mapping = _gpt2_byte_to_unicode()
    alphabet = tuple(mapping[value] for value in range(256))
    if len(set(alphabet)) != 256 or any(character.isspace() for character in alphabet):
        raise AssertionError("GPT-2 byte alphabet must be unique and whitespace-free")
    return alphabet


def bytes_to_level_string(raw: bytes) -> str:
    if not isinstance(raw, bytes):
        raise TypeError("byte-Unigram input must be bytes")
    mapping = _gpt2_byte_to_unicode()
    return "".join(mapping[value] for value in raw)


def level_string_to_bytes(value: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("byte-Unigram piece must be a non-empty string")
    inverse = {character: byte for byte, character in _gpt2_byte_to_unicode().items()}
    try:
        return bytes(inverse[character] for character in value)
    except KeyError as error:
        raise ValueError("byte-Unigram piece is outside the byte alphabet") from error


def ordered_scores_sha256(scores: Sequence[float]) -> str:
    values = tuple(float(score) for score in scores)
    if not values or not all(math.isfinite(score) for score in values):
        raise ValueError("byte-Unigram scores must be finite")
    digest = hashlib.sha256(b"JamoFlow/byte-unigram-scores/v1\0")
    digest.update(len(values).to_bytes(8, "big"))
    for score in values:
        digest.update(struct.pack(">d", score))
    return digest.hexdigest()


def project_mandatory_byte_fallback(
    sentencepiece_rows: Sequence[tuple[bytes, float]],
    *,
    vocabulary_size: int,
) -> tuple[tuple[bytes, ...], tuple[float, ...], dict[str, int | float | None]]:
    """Project learned rows onto exact byte fallback plus best non-byte pieces.

    SentencePiece may prune rare one-character pieces even when they appear as
    synthetic rows.  We therefore reserve IDs 0..255 for all bytes and fill the
    remaining budget with the highest-score learned multi-byte pieces.  Ties
    retain SentencePiece order, making the projection deterministic.
    """

    if vocabulary_size <= 256 or not sentencepiece_rows:
        raise ValueError("byte fallback projection budget differs")
    normalized = tuple((bytes(piece), float(score)) for piece, score in sentencepiece_rows)
    if (
        any(not piece or not math.isfinite(score) for piece, score in normalized)
        or len({piece for piece, _ in normalized}) != len(normalized)
    ):
        raise ValueError("byte fallback projection rows differ")
    score_by_piece = {piece: score for piece, score in normalized}
    mandatory = tuple(bytes((value,)) for value in range(256))
    missing = tuple(piece for piece in mandatory if piece not in score_by_piece)
    fallback_score = min(score_by_piece.values()) - 10.0 if missing else None
    nonbytes = [
        (index, piece, score)
        for index, (piece, score) in enumerate(normalized)
        if len(piece) != 1
    ]
    target_nonbytes = vocabulary_size - 256
    if len(nonbytes) < target_nonbytes:
        raise ValueError("byte fallback projection lacks learned multi-byte pieces")
    selected_indices = {
        index
        for index, _, _ in sorted(
            nonbytes,
            key=lambda row: (-row[2], row[0], row[1]),
        )[:target_nonbytes]
    }
    selected = tuple(
        (piece, score)
        for index, piece, score in nonbytes
        if index in selected_indices
    )
    pieces = mandatory + tuple(piece for piece, _ in selected)
    scores = tuple(
        score_by_piece.get(piece, fallback_score) for piece in mandatory
    ) + tuple(score for _, score in selected)
    if fallback_score is not None and any(score is None for score in scores):
        raise AssertionError("byte fallback projection score is absent")
    return pieces, tuple(float(score) for score in scores), {
        "missing_single_bytes_inserted": len(missing),
        "learned_nonbyte_pieces_retained": len(selected),
        "learned_pieces_dropped_for_fallback": len(nonbytes) - len(selected),
        "inserted_byte_fallback_score": fallback_score,
    }


def deterministic_trainer_contract(
    *, vocabulary_size: int, maximum_piece_bytes: int
) -> dict[str, int | float | bool | str]:
    return {
        "add_dummy_prefix": False,
        "bos_id": -1,
        "byte_fallback": False,
        "character_coverage": 1.0,
        "hard_vocab_limit": True,
        "input_sentence_size": 0,
        "max_sentence_length": MAXIMUM_SENTENCE_BYTES,
        "max_sentencepiece_length": maximum_piece_bytes,
        "model_type": "unigram",
        "normalization_rule_name": "identity",
        "num_sub_iterations": SUB_ITERATIONS,
        "num_threads": 1,
        "pad_id": -1,
        "remove_extra_whitespaces": False,
        "seed_sentencepiece_size": SEED_SENTENCEPIECE_SIZE,
        "shuffle_input_sentence": False,
        "shrinking_factor": SHRINKING_FACTOR,
        "split_by_number": False,
        "split_by_unicode_script": False,
        "split_by_whitespace": False,
        "split_digits": False,
        "synthetic_alphabet_rows": 256,
        "target_sentencepiece_vocabulary_size": (
            vocabulary_size + SENTENCEPIECE_UNKNOWN_COUNT
        ),
        "treat_whitespace_as_suffix": False,
        "unk_id": 0,
    }


def train_deterministic_byte_unigram(
    texts: Iterable[str],
    *,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    maximum_piece_bytes: int = MAXIMUM_PIECE_BYTES,
):
    """Train and return ``(runtime_tokenizer, pieces, scores, metadata)``.

    The 256 one-character synthetic rows force byte fallback without creating
    artificial multi-byte n-grams.  They are vocabulary-training scaffolding,
    never part of the language-model training stream.
    """

    if vocabulary_size <= 256:
        raise ValueError("byte-Unigram vocabulary must exceed byte fallback")
    if not 1 <= maximum_piece_bytes <= MAXIMUM_SENTENCE_BYTES:
        raise ValueError("byte-Unigram maximum piece length differs")

    import sentencepiece as sentencepiece

    if sentencepiece.__version__ != PINNED_SENTENCEPIECE_VERSION:
        raise RuntimeError("SentencePiece version differs from the pinned contract")

    alphabet = byte_level_alphabet()
    source_iterator = iter(texts)
    try:
        first_text = next(source_iterator)
    except StopIteration as error:
        raise ValueError("byte-Unigram training requires source documents") from error
    source_count = 0

    def validated_level_text(text: str) -> str:
        nonlocal source_count
        if not isinstance(text, str) or not text:
            raise ValueError("byte-Unigram training rows must be non-empty text")
        raw = text.encode("utf-8")
        if len(raw) > MAXIMUM_SENTENCE_BYTES:
            raise ValueError("byte-Unigram source document exceeds the pinned limit")
        source_count += 1
        return bytes_to_level_string(raw)

    first_level_text = validated_level_text(first_text)

    def sentence_iterator():
        yield from alphabet
        yield first_level_text
        for text in source_iterator:
            yield validated_level_text(text)

    model_writer = io.BytesIO()
    contract = deterministic_trainer_contract(
        vocabulary_size=vocabulary_size,
        maximum_piece_bytes=maximum_piece_bytes,
    )
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=sentence_iterator(),
        model_writer=model_writer,
        model_type="unigram",
        vocab_size=vocabulary_size + SENTENCEPIECE_UNKNOWN_COUNT,
        hard_vocab_limit=True,
        unk_id=0,
        unk_piece=SENTENCEPIECE_UNKNOWN_PIECE,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        normalization_rule_name="identity",
        add_dummy_prefix=False,
        remove_extra_whitespaces=False,
        split_by_whitespace=False,
        split_by_unicode_script=False,
        split_by_number=False,
        split_digits=False,
        treat_whitespace_as_suffix=False,
        byte_fallback=False,
        character_coverage=1.0,
        input_sentence_size=0,
        shuffle_input_sentence=False,
        num_threads=1,
        max_sentence_length=MAXIMUM_SENTENCE_BYTES,
        max_sentencepiece_length=maximum_piece_bytes,
        seed_sentencepiece_size=SEED_SENTENCEPIECE_SIZE,
        shrinking_factor=SHRINKING_FACTOR,
        num_sub_iterations=SUB_ITERATIONS,
        minloglevel=2,
    )
    if source_count <= 0:
        raise AssertionError("byte-Unigram source preflight was not consumed")

    model_proto = model_writer.getvalue()
    processor = sentencepiece.SentencePieceProcessor(model_proto=model_proto)
    if (
        processor.vocab_size() != vocabulary_size + SENTENCEPIECE_UNKNOWN_COUNT
        or processor.id_to_piece(0) != SENTENCEPIECE_UNKNOWN_PIECE
    ):
        raise ValueError("SentencePiece byte-Unigram vocabulary size differs")

    sentencepiece_rows: list[tuple[bytes, float]] = []
    seen_pieces: set[bytes] = set()
    for token_id in range(1, processor.vocab_size()):
        piece = level_string_to_bytes(processor.id_to_piece(token_id))
        score = float(processor.get_score(token_id))
        if piece in seen_pieces or not math.isfinite(score):
            raise ValueError("SentencePiece byte-Unigram piece differs")
        seen_pieces.add(piece)
        sentencepiece_rows.append((piece, score))

    pieces, scores, projection = project_mandatory_byte_fallback(
        sentencepiece_rows,
        vocabulary_size=vocabulary_size,
    )
    mandatory = tuple(bytes((value,)) for value in range(256))
    validate_ordered_byte_pieces(
        pieces,
        expected_vocabulary_size=vocabulary_size,
        maximum_piece_bytes=maximum_piece_bytes,
    )
    tokenizer = build_scored_byte_unigram_tokenizer(
        pieces,
        scores,
        maximum_piece_bytes=maximum_piece_bytes,
    )
    recovered = byte_bpe_token_bytes(tokenizer)
    samples = (
        "한글 byte-Unigram 검증",
        " 공백  두 개\n탭\t끝 ",
        "ASCII, 숫자 123, 기호 !?",
    )
    roundtrip = all(
        tokenizer.decode(
            tokenizer.encode(sample, add_special_tokens=False).ids,
            skip_special_tokens=False,
        )
        == sample
        for sample in samples
    )
    tokenizer_json = tokenizer.to_str(pretty=False).encode("utf-8")
    full_alphabet = recovered[:256] == mandatory
    exact_size = len(recovered) == vocabulary_size
    overall = bool(full_alphabet and exact_size and roundtrip)
    metadata = ByteUnigramTrainingMetadata(
        schema_version=1,
        vocabulary_size=len(recovered),
        source_document_count=source_count,
        synthetic_single_byte_rows=256,
        maximum_piece_bytes=max(map(len, recovered)),
        sentencepiece_version=sentencepiece.__version__,
        sentencepiece_model_sha256=hashlib.sha256(model_proto).hexdigest(),
        ordered_pieces_sha256=ordered_pieces_sha256(pieces),
        ordered_scores_sha256=ordered_scores_sha256(scores),
        tokenizer_json_sha256=hashlib.sha256(tokenizer_json).hexdigest(),
        full_byte_alphabet=full_alphabet,
        exact_vocabulary_size=exact_size,
        byte_exact_sample_roundtrip=roundtrip,
        **projection,
        deterministic_trainer_contract=contract,
        overall_pass=overall,
    )
    if not overall:
        raise ValueError("trained byte-Unigram failed its exact audit")
    return tokenizer, pieces, scores, model_proto, metadata
