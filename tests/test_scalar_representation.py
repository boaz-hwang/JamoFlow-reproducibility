import importlib.util
import json
import math
import unittest

from scripts.analyze_scalar_representation_opportunity import (
    IMPLEMENTATION_PATHS,
    PLAN_PATH,
    _validate_plan,
)
from jamoflow.compute_conversion import conversion_model_spec
from scripts.scalar_representation_core import (
    audit_bpe_encoding,
    complete_utf8_prefix,
    hangul_components,
    hangul_dependence,
    representation_counts,
    scalar_blt_opportunity_flops,
    scalar_inventory,
    train_exact_byte_bpe,
)


class ScalarRepresentationTest(unittest.TestCase):
    def test_sealed_opportunity_plan_matches_implementation(self):
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        _validate_plan(plan)
        self.assertEqual(
            set(plan["implementation_sha256"]),
            set(IMPLEMENTATION_PATHS),
        )

    def test_complete_prefix_rejects_invalid_interior_and_preserves_suffix(self):
        complete, suffix = complete_utf8_prefix(
            "가A".encode("utf-8") + "힣".encode("utf-8")[:2]
        )
        self.assertEqual(complete, "가A")
        self.assertEqual(suffix, "힣".encode("utf-8")[:2])
        with self.assertRaisesRegex(ValueError, "invalid interior"):
            complete_utf8_prefix(b"a\xffb")

    def test_representation_accounting_and_hangul_components(self):
        data = "가Aé😀각".encode("utf-8")
        counts = representation_counts(data)
        self.assertEqual(hangul_components(ord("가")), (0, 0, 0))
        self.assertEqual(hangul_components(ord("각")), (0, 0, 1))
        self.assertEqual(hangul_components(0xD7A3), (18, 20, 27))
        self.assertEqual(counts["input_bytes"], 13)
        self.assertEqual(counts["complete_scalars"], 5)
        self.assertEqual(counts["precomposed_hangul_scalars"], 2)
        self.assertEqual(
            counts["sequential_steps"]["generic_unicode_scalar_with_raw_suffix_fallback"],
            5,
        )
        self.assertEqual(
            counts["sequential_steps"]["hangul_scalar_otherwise_raw_byte"],
            9,
        )

    def test_conditional_chain_closes_joint_and_independent_heads_do_not(self):
        metrics = hangul_dependence("가가각나난난난")
        entropy = metrics["entropy_bits"]
        self.assertTrue(
            math.isclose(
                entropy["conditional_chain_total"],
                entropy["joint"],
                rel_tol=0,
                abs_tol=1e-12,
            )
        )
        self.assertGreater(entropy["independent_excess_total_correlation"], 0)
        self.assertTrue(
            metrics["interpretation"]["conditional_chain_can_represent_joint_exactly"]
        )

    def test_inventory_counts_unseen_scalars_and_hangul(self):
        inventory = scalar_inventory(("가A", "\n각"), "가힣AB")
        calibration = inventory["calibration"]
        self.assertEqual(inventory["train"]["unique_scalars"], 4)
        self.assertEqual(calibration["unseen_scalar_types"], 2)
        self.assertEqual(calibration["unseen_scalar_occurrences"], 2)
        self.assertEqual(calibration["unseen_hangul_types"], 1)

    def test_flop_opportunity_uses_fewer_steps_but_is_not_speed_evidence(self):
        text = ("가나다라마바사아자차카타파하 ABC 123.\n" * 20)
        raw = text.encode("utf-8")
        raw = (raw * (512 // len(raw) + 2))[:512]
        # Make the synthetic window strict while retaining exactly 512 bytes.
        while True:
            try:
                raw.decode("utf-8")
                break
            except UnicodeDecodeError as error:
                if error.reason != "unexpected end of data":
                    raise
                raw = raw[: error.start] + b" " * (512 - error.start)
        result = scalar_blt_opportunity_flops(
            raw,
            baseline_spec=conversion_model_spec(72),
            data_patches=72,
        )
        self.assertEqual(result["raw_windows"], 1)
        self.assertGreater(
            result["generic_unicode_scalar"]["reduction_relative_to_w72"],
            0,
        )
        self.assertGreater(
            result["hangul_scalar_otherwise_raw_byte"][
                "reduction_relative_to_w72"
            ],
            0,
        )
        self.assertIn("conditional micro-head dependencies and kernel launches", result["omitted"])


HAS_TOKENIZERS = importlib.util.find_spec("tokenizers") is not None


@unittest.skipUnless(HAS_TOKENIZERS, "optional tokenizer research dependency")
class ExactByteBPETest(unittest.TestCase):
    def test_non_nfc_source_is_preserved_without_normalization(self):
        corpus = ("가 cafe\u0301\n", "각 ASCII 123\n") * 20
        first = train_exact_byte_bpe(
            corpus,
            vocabulary_size=280,
            minimum_frequency=1,
        )
        second = train_exact_byte_bpe(
            corpus,
            vocabulary_size=280,
            minimum_frequency=1,
        )
        self.assertEqual(first.to_str(pretty=False), second.to_str(pretty=False))
        audit = audit_bpe_encoding(first, "cafe\u0301와 각")
        self.assertTrue(audit["roundtrip_identity"])
        self.assertGreaterEqual(audit["vocabulary_size"], 256)
        self.assertLessEqual(audit["vocabulary_size"], 280)


if __name__ == "__main__":
    unittest.main()
