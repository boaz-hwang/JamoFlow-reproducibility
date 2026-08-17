import unittest

from jamoflow.corpus import Record, stable_record_id
from jamoflow.entropy import ByteNGramModel
from jamoflow.metrics import (
    ScoredRecord,
    build_evaluation_context,
    evaluate_policy,
    make_bootstrap_weights,
)
from jamoflow.patching import FixedStridePolicy


def scored_record(text: str, model: ByteNGramModel) -> ScoredRecord:
    raw = text.encode("utf-8")
    record = Record(
        record_id=stable_record_id(raw),
        source="fixture",
        ordinal=1,
        raw=raw,
        text=text,
    )
    return ScoredRecord(record=record, scores=tuple(model.score(raw)))


class MetricsTests(unittest.TestCase):
    def test_cached_context_preserves_structural_metrics(self) -> None:
        model = ByteNGramModel(order=2).fit(["한글 ABC".encode("utf-8")])
        records = [scored_record("한A", model), scored_record("漢B", model)]
        context = build_evaluation_context(records)

        cached = evaluate_policy(
            records,
            FixedStridePolicy(2),
            comparison_group="fixture",
            role="cached",
            runtime_repeats=1,
            context=context,
        )
        uncached = evaluate_policy(
            records,
            FixedStridePolicy(2),
            comparison_group="fixture",
            role="uncached",
            runtime_repeats=1,
        )

        self.assertEqual(len(context.ranked_positions), sum(len(r.record.raw) - 1 for r in records))
        self.assertEqual(
            cached.boundaries_inside_hangul_syllable,
            uncached.boundaries_inside_hangul_syllable,
        )
        self.assertEqual(
            cached.boundaries_inside_cjk_ideograph,
            uncached.boundaries_inside_cjk_ideograph,
        )
        self.assertAlmostEqual(
            cached.oracle_entropy_capture_ratio,
            uncached.oracle_entropy_capture_ratio,
        )

    def test_record_bootstrap_is_seeded_and_reported(self) -> None:
        model = ByteNGramModel(order=1).fit(
            [text.encode("utf-8") for text in ("가나다", "ABC", "漢字")]
        )
        records = [
            scored_record("가나다", model),
            scored_record("ABC", model),
            scored_record("漢字", model),
        ]
        first = make_bootstrap_weights(3, repeats=50, seed=1729)
        second = make_bootstrap_weights(3, repeats=50, seed=1729)
        self.assertTrue((first == second).all())

        metrics = evaluate_policy(
            records,
            FixedStridePolicy(2),
            comparison_group="fixture",
            role="bootstrap",
            runtime_repeats=1,
            bootstrap_weights=first,
        )

        self.assertIsNotNone(metrics.bootstrap_95)
        self.assertIn("average_patch_bytes", metrics.bootstrap_95)
        self.assertIn("score_evaluations_per_byte", metrics.bootstrap_95)


if __name__ == "__main__":
    unittest.main()
