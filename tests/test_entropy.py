import unittest

from jamoflow.entropy import ByteNGramModel


class EntropyTests(unittest.TestCase):
    def test_seen_continuation_gets_higher_probability(self) -> None:
        model = ByteNGramModel(order=2, alpha=0.1).fit([b"abababababab"])
        seen_probability, order = model.probability(ord("a"), b"ab")
        unseen_probability, _ = model.probability(ord("z"), b"ab")

        self.assertEqual(order, 2)
        self.assertGreater(seen_probability, unseen_probability)

    def test_scores_are_finite_and_aligned(self) -> None:
        model = ByteNGramModel(order=3).fit(["한글 byte".encode("utf-8")])
        data = "한글".encode("utf-8")
        scores = model.score(data)

        self.assertEqual(len(scores), len(data))
        self.assertTrue(all(score.entropy_bits >= 0 for score in scores))
        self.assertTrue(all(score.surprisal_bits >= 0 for score in scores))


if __name__ == "__main__":
    unittest.main()

