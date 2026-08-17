import math
import unittest

from jamoflow.publication_protocol import (
    ACTUAL_INFERENCE_BYTE_MAXIMUM_OVERSHOOT,
    ACTUAL_INFERENCE_MINIMUM_VALID_OUTPUT_BYTES,
    ACTUAL_INFERENCE_VALID_OUTPUT_CONSTRAINT,
    DATASET_PINS,
    PRIMARY_DOWNSTREAM_TASKS,
    PUBLICATION_BPE_INITIAL_ALPHABET_SIZE,
    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS,
    PUBLICATION_BPE_STRESS_VOCABULARY_SIZE,
    PUBLICATION_BPE_VOCABULARY_SIZE,
    PUBLICATION_BPE_VOCABULARY_CANDIDATES,
    PUBLICATION_BPB_CONTEXT_BYTES,
    PUBLICATION_BPB_CONTEXT_CONTRACT,
    PUBLICATION_BPB_NONINFERIORITY_MARGIN,
    PUBLICATION_BPB_TARGET_BLOCK_BYTES,
    PUBLICATION_DOWNSTREAM_REFERENCE_KEYS,
    PUBLICATION_RAW_COMPARATOR_MODEL_KEY,
    SECONDARY_DOWNSTREAM_TASKS,
    choose_validation_reference,
    validate_publication_protocol,
)


class PublicationProtocolTests(unittest.TestCase):
    def test_manifest_is_internally_valid(self) -> None:
        validate_publication_protocol()
        self.assertEqual(PUBLICATION_BPE_INITIAL_ALPHABET_SIZE, 256)
        self.assertEqual(PUBLICATION_BPE_STRESS_VOCABULARY_SIZE, 16_000)
        self.assertEqual(PUBLICATION_BPE_VOCABULARY_SIZE, 32_000)
        self.assertEqual(
            PUBLICATION_BPE_VOCABULARY_CANDIDATES,
            (16_000, 32_000),
        )
        self.assertEqual(PUBLICATION_BPB_NONINFERIORITY_MARGIN, 0.010)
        self.assertEqual(PUBLICATION_BPB_CONTEXT_BYTES, 512)
        self.assertEqual(PUBLICATION_BPB_TARGET_BLOCK_BYTES, 256)
        self.assertIn("raw_capped_rolling", PUBLICATION_BPB_CONTEXT_CONTRACT)
        self.assertEqual(ACTUAL_INFERENCE_MINIMUM_VALID_OUTPUT_BYTES, 128)
        self.assertEqual(ACTUAL_INFERENCE_BYTE_MAXIMUM_OVERSHOOT, 3)
        self.assertIn("strict_rfc3629", ACTUAL_INFERENCE_VALID_OUTPUT_CONSTRAINT)

    def test_primary_suite_uses_only_redistributable_pinned_sources(self) -> None:
        for task in PRIMARY_DOWNSTREAM_TASKS.values():
            pin = DATASET_PINS[task.dataset_key]
            self.assertEqual(pin.license_spdx, "CC-BY-SA-4.0")
            self.assertEqual(len(pin.revision), 40)

    def test_unclear_or_no_derivatives_sources_are_not_primary(self) -> None:
        primary_datasets = {
            task.dataset_key for task in PRIMARY_DOWNSTREAM_TASKS.values()
        }
        self.assertNotIn("haerae_bench", primary_datasets)
        self.assertNotIn("kmmlu", primary_datasets)
        self.assertIsNone(DATASET_PINS["haerae_bench"].license_spdx)
        self.assertEqual(DATASET_PINS["kmmlu"].license_spdx, "CC-BY-ND-4.0")

    def test_every_label_is_one_ascii_byte(self) -> None:
        all_tasks = dict(PRIMARY_DOWNSTREAM_TASKS)
        all_tasks.update(SECONDARY_DOWNSTREAM_TASKS)
        for task in all_tasks.values():
            self.assertEqual(
                task.labels,
                tuple(str(index) for index in range(task.label_count)),
            )
            self.assertTrue(all(len(label.encode("ascii")) == 1 for label in task.labels))
        self.assertTrue(
            all(
                task.primary_metric in {"macro_f1", "accuracy"}
                for task in PRIMARY_DOWNSTREAM_TASKS.values()
            )
        )

    def test_reference_selection_prefers_bpe_inside_tie_band(self) -> None:
        selected = choose_validation_reference(
            {
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: (
                    0.702,
                    0.712,
                    0.692,
                ),
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: (
                    0.70,
                    0.71,
                    0.69,
                ),
                PUBLICATION_RAW_COMPARATOR_MODEL_KEY: (
                    0.704,
                    0.714,
                    0.694,
                ),
            }
        )
        self.assertEqual(
            selected,
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
        )

    def test_reference_selection_uses_clearly_stronger_raw_model(self) -> None:
        selected = choose_validation_reference(
            {
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: (
                    0.702,
                    0.712,
                    0.692,
                ),
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: (
                    0.70,
                    0.71,
                    0.69,
                ),
                PUBLICATION_RAW_COMPARATOR_MODEL_KEY: (
                    0.73,
                    0.74,
                    0.72,
                ),
            }
        )
        self.assertEqual(selected, PUBLICATION_RAW_COMPARATOR_MODEL_KEY)

    def test_dual_bpe_reference_uses_32k_as_only_near_tie_default(self) -> None:
        selected = choose_validation_reference(
            {
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: (
                    0.704,
                    0.704,
                    0.704,
                ),
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: (
                    0.700,
                    0.700,
                    0.700,
                ),
                PUBLICATION_RAW_COMPARATOR_MODEL_KEY: (0.703, 0.703, 0.703),
            },
        )
        self.assertEqual(
            selected,
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000],
        )

        clearly_stronger_16k = choose_validation_reference(
            {
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: (
                    0.720,
                    0.720,
                    0.720,
                ),
                PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: (
                    0.700,
                    0.700,
                    0.700,
                ),
                PUBLICATION_RAW_COMPARATOR_MODEL_KEY: (0.703, 0.703, 0.703),
            },
        )
        self.assertEqual(
            clearly_stronger_16k,
            PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000],
        )

    def test_reference_selection_rejects_nonfinite_score(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            choose_validation_reference(
                {
                    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000]: (
                        0.7,
                        0.7,
                        0.7,
                    ),
                    PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[32_000]: (
                        0.7,
                        0.7,
                        math.nan,
                    ),
                    PUBLICATION_RAW_COMPARATOR_MODEL_KEY: (0.7, 0.7, 0.7),
                }
            )

    def test_reference_selection_requires_all_three_sealed_controls(self) -> None:
        scores = {
            key: (0.7, 0.7, 0.7)
            for key in PUBLICATION_DOWNSTREAM_REFERENCE_KEYS
        }
        scores.pop(PUBLICATION_BPE_COMPARATOR_MODEL_KEYS[16_000])
        with self.assertRaisesRegex(ValueError, "exact raw, 16K, and 32K"):
            choose_validation_reference(scores)


if __name__ == "__main__":
    unittest.main()
