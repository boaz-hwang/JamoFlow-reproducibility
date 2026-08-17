import hashlib
import unittest

from jamoflow.downstream_data import (
    DOWNSTREAM_MAXIMUM_PROMPT_BYTES,
    DOWNSTREAM_LABEL_CONDITIONING,
    DOWNSTREAM_PROMPT_VERSION,
    benchmark_input_text,
    encode_downstream_conditioning,
    klue_internal_hash_split,
    render_downstream_prompt,
    validate_downstream_data_protocol,
)
from jamoflow.publication_protocol import PRIMARY_DOWNSTREAM_TASKS


SAMPLE_ROWS = {
    "kobest_boolq": {
        "paragraph": "물은 섭씨 100도 부근에서 끓는다.",
        "question": "물은 끓을 수 있는가?",
        "label": 1,
    },
    "kobest_copa": {
        "premise": "물을 오래 끓였다.",
        "question": "결과",
        "alternative_1": "물의 양이 늘었다.",
        "alternative_2": "물의 양이 줄었다.",
        "label": 1,
    },
    "kobest_wic": {
        "word": "양분",
        "context_1": "토양에 [양분]이 풍부하다.",
        "context_2": "모체에서 [양분]을 공급받는다.",
        "label": 1,
    },
    "kobest_sentineg": {"sentence": "정말 마음에 든다.", "label": 1},
    "klue_ynat": {"title": "과학 기술 관련 새 소식", "label": 0},
    "klue_nli": {
        "premise": "고양이가 창가에 앉아 있다.",
        "hypothesis": "동물이 창가에 있다.",
        "label": 0,
    },
}


class DownstreamPromptTests(unittest.TestCase):
    def test_every_primary_task_renders_one_ascii_label_inside_context(self) -> None:
        self.assertEqual(set(SAMPLE_ROWS), set(PRIMARY_DOWNSTREAM_TASKS))
        for task_key, row in SAMPLE_ROWS.items():
            with self.subTest(task=task_key):
                rendered = render_downstream_prompt(task_key, row)
                self.assertEqual(rendered.prompt_version, DOWNSTREAM_PROMPT_VERSION)
                self.assertEqual(len(rendered.label_digit.encode("ascii")), 1)
                self.assertLessEqual(
                    len((rendered.prompt + rendered.label_digit).encode("utf-8")),
                    512,
                )
                self.assertNotIn(rendered.label_digit, rendered.prompt[-1:])

    def test_prompt_snapshot_locks_field_order_and_wording(self) -> None:
        expected = {
            "klue_nli": "6dabeb7acc774a6965f9954ae8c4ec040e871681070ba7f869ad501873c4a03e",
            "klue_ynat": "a300d216a557d2d51843bac8534145dbfffe4ae3cfb86d5d59d2c3c327dc22ac",
            "kobest_boolq": "8a0aa851a8c7c683b8e115425ad64f3199db84d1524d323d6087679a9e6b59a4",
            "kobest_copa": "e9becf8e7067954baa3c6a539d031edd89c95d1dc2f9035e6de4b4fbec282d34",
            "kobest_sentineg": "4b10e9b4729fe4e606247885347187510f6e6e65d17397e1ef7d6a7375a31fe5",
            "kobest_wic": "b38b6016cdc0235e9839097b6a94ef2d975d47ea4b9a52fa58f4ee911d29fa9f",
        }
        for task_key, row in SAMPLE_ROWS.items():
            rendered = render_downstream_prompt(task_key, row)
            digest = hashlib.sha256(rendered.prompt.encode("utf-8")).hexdigest()
            self.assertEqual(digest, expected[task_key])

    def test_renderer_protocol_is_internally_valid(self) -> None:
        validate_downstream_data_protocol()

    def test_incorrect_wic_card_alias_is_rejected(self) -> None:
        row = dict(SAMPLE_ROWS["kobest_wic"])
        row["target_word"] = row.pop("word")
        with self.assertRaisesRegex(ValueError, "word"):
            render_downstream_prompt("kobest_wic", row)

    def test_truncation_preserves_question_and_options(self) -> None:
        row = dict(SAMPLE_ROWS["kobest_boolq"])
        row["paragraph"] = "가나다라마바사" * 100
        rendered = render_downstream_prompt("kobest_boolq", row)
        self.assertTrue(rendered.truncated)
        self.assertLessEqual(rendered.rendered_prompt_bytes, DOWNSTREAM_MAXIMUM_PROMPT_BYTES)
        self.assertIn(row["question"], rendered.prompt)
        self.assertIn("0: 아니오\n1: 예", rendered.prompt)

    def test_nfc_normalization_is_recorded(self) -> None:
        row = dict(SAMPLE_ROWS["klue_ynat"])
        row["title"] = "가나다 소식"
        rendered = render_downstream_prompt("klue_ynat", row)
        self.assertEqual(rendered.normalization_changed_fields, ("title",))
        self.assertIn("가나다", rendered.prompt)

    def test_content_free_metadata_excludes_prompt_and_label(self) -> None:
        rendered = render_downstream_prompt("klue_nli", SAMPLE_ROWS["klue_nli"])
        metadata = rendered.metadata()
        self.assertNotIn("prompt", metadata)
        self.assertNotIn("label_digit", metadata)

    def test_label_conditioning_encodes_prompt_and_digit_separately(self) -> None:
        rendered = render_downstream_prompt(
            "klue_nli",
            SAMPLE_ROWS["klue_nli"],
        )

        def boundary_merging_encoder(text: str) -> tuple[int, ...]:
            encoded = tuple(text.encode("utf-8"))
            if text == rendered.prompt + rendered.label_digit:
                return encoded[:-1] + (10_000,)
            return encoded

        conditioned = encode_downstream_conditioning(
            rendered,
            boundary_merging_encoder,
        )
        self.assertFalse(conditioned.joint_encoding_matches_separate)
        self.assertEqual(
            conditioned.gold_label_unit,
            ord(rendered.label_digit),
        )
        self.assertEqual(
            conditioned.conditioning_contract,
            DOWNSTREAM_LABEL_CONDITIONING,
        )
        metadata = conditioned.metadata()
        self.assertTrue(metadata["joint_boundary_merge_observed"])
        self.assertNotIn("prompt_units", metadata)
        self.assertNotIn("gold_label_unit", metadata)

    def test_label_conditioning_rejects_multiunit_or_colliding_digits(self) -> None:
        rendered = render_downstream_prompt(
            "klue_nli",
            SAMPLE_ROWS["klue_nli"],
        )

        def multiunit_digit(text: str) -> tuple[int, ...]:
            if text in {"0", "1", "2"}:
                return (100, 101)
            return tuple(text.encode("utf-8"))

        with self.assertRaisesRegex(ValueError, "distinct single units"):
            encode_downstream_conditioning(rendered, multiunit_digit)

        def colliding_digits(text: str) -> tuple[int, ...]:
            if text in {"0", "1", "2"}:
                return (100,)
            return tuple(text.encode("utf-8"))

        with self.assertRaisesRegex(ValueError, "distinct single units"):
            encode_downstream_conditioning(rendered, colliding_digits)

    def test_contamination_input_has_no_instruction_or_label(self) -> None:
        text = benchmark_input_text("kobest_copa", SAMPLE_ROWS["kobest_copa"])
        self.assertIn("물을 오래 끓였다.", text)
        self.assertNotIn("[정답]", text)
        self.assertNotIn("label", text)


class KLUEInternalSplitTests(unittest.TestCase):
    def _rows(self, task_key: str) -> list[dict]:
        label_count = PRIMARY_DOWNSTREAM_TASKS[task_key].label_count
        rows = []
        for label in range(label_count):
            for index in range(20):
                if task_key == "klue_ynat":
                    row = {"title": f"라벨 {label} 뉴스 제목 {index}", "label": label}
                else:
                    row = {
                        "premise": f"라벨 {label} 전제 문장 {index}",
                        "hypothesis": f"라벨 {label} 가설 문장 {index}",
                        "label": label,
                    }
                row["guid"] = f"{task_key}-{label}-{index}"
                rows.append(row)
        return rows

    def test_split_is_stratified_exact_and_order_invariant(self) -> None:
        rows = self._rows("klue_nli")
        forward = klue_internal_hash_split("klue_nli", rows)
        reverse = klue_internal_hash_split("klue_nli", list(reversed(rows)))
        self.assertEqual(forward, reverse)
        self.assertEqual(forward.counts_by_label, ((0, 18, 2), (1, 18, 2), (2, 18, 2)))
        self.assertFalse(set(forward.fit_row_ids) & set(forward.selection_row_ids))
        self.assertEqual(len(forward.assignment_sha256), 64)

    def test_duplicate_guid_is_rejected(self) -> None:
        rows = self._rows("klue_nli")
        rows[-1]["guid"] = rows[0]["guid"]
        with self.assertRaisesRegex(ValueError, "identity"):
            klue_internal_hash_split("klue_nli", rows)


if __name__ == "__main__":
    unittest.main()
