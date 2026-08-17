import unittest

from jamoflow.entropy import PositionScore
from jamoflow.patching import (
    CandidateEntropyPolicy,
    CJKIdeographPolicy,
    CodepointAlignedStridePolicy,
    EntropyPolicy,
    EojeolCappedPolicy,
    FixedStridePolicy,
    HangulSyllablePolicy,
    OrthographicCandidateEntropyPolicy,
    SpaceBytePolicy,
    assert_prefix_causal,
)
from jamoflow.utf8 import is_continuation_byte


def constant_scores(data: bytes, entropy: float = 2.0) -> list[PositionScore]:
    return [
        PositionScore(entropy, entropy, 2 ** (-entropy), 0) for _ in data
    ]


class PatchingTests(unittest.TestCase):
    def test_fixed_and_codepoint_strides_differ_on_hangul(self) -> None:
        data = "한글A".encode("utf-8")
        fixed = FixedStridePolicy(4).boundaries(data)
        aligned = CodepointAlignedStridePolicy(4).boundaries(data)

        self.assertEqual(fixed, (0, 4))
        self.assertEqual(aligned, (0, 6))

    def test_hangul_policy_places_boundary_after_each_syllable(self) -> None:
        data = "한글A".encode("utf-8")
        self.assertEqual(HangulSyllablePolicy().boundaries(data), (0, 3, 6))

    def test_cjk_policy_places_boundary_after_each_ideograph(self) -> None:
        data = "漢字A".encode("utf-8")
        self.assertEqual(CJKIdeographPolicy().boundaries(data), (0, 3, 6))

    def test_orthographic_candidate_is_aligned_and_capped(self) -> None:
        data = "한A very long ASCII phrase".encode("utf-8")
        scores = constant_scores(data, entropy=8.0)
        policy = OrthographicCandidateEntropyPolicy(
            threshold=7.0,
            script="hangul",
            max_patch_bytes=6,
        )

        boundaries = policy.boundaries(data, scores)
        ends = [*boundaries[1:], len(data)]
        lengths = [
            end - start for start, end in zip(boundaries, ends, strict=True)
        ]

        self.assertIn(3, boundaries)
        self.assertTrue(
            all(not is_continuation_byte(data[index]) for index in boundaries[1:])
        )
        self.assertLessEqual(max(lengths), 8)

    def test_spacebyte_cadence_can_split_multibyte_codepoint(self) -> None:
        data = "한글".encode("utf-8")
        boundaries = SpaceBytePolicy().boundaries(data)

        self.assertIn(1, boundaries)
        self.assertIn(4, boundaries)

    def test_all_policies_are_prefix_causal(self) -> None:
        data = "한글 test! ㅋㅋ".encode("utf-8")
        scores = constant_scores(data)
        policies = [
            FixedStridePolicy(4),
            CodepointAlignedStridePolicy(4),
            SpaceBytePolicy(),
            HangulSyllablePolicy(),
            EojeolCappedPolicy(8),
            EntropyPolicy(1.5),
            CandidateEntropyPolicy(1.5),
            OrthographicCandidateEntropyPolicy(
                threshold=1.5,
                script="hangul",
                max_patch_bytes=7,
            ),
        ]

        for policy in policies:
            with self.subTest(policy=policy.name):
                assert_prefix_causal(policy, data, scores)


if __name__ == "__main__":
    unittest.main()
