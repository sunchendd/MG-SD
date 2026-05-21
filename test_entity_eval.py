import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from entity_eval import (
  build_prompt_and_gold,
  extract_entities,
  keep_prompt_tail_with_token_limit,
  score_entity_errors,
)


class EntityEvalTests(unittest.TestCase):
  def test_trims_prompt_to_recent_token_budget(self):
    class FakeTokenizer:
      def encode(self, text, add_special_tokens=False):
        return text.split()

      def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)

    text = "a b c d e f g h i j"

    trimmed = keep_prompt_tail_with_token_limit(text, FakeTokenizer(), max_prompt_tokens=5)

    self.assertEqual(trimmed, "f g h i j")

  def test_extracts_medication_dose_frequency_and_negation_entities(self):
    text = (
      "Aspirin 81 mg daily. She denies chest pain. "
      "Furosemide 40 mg bid for edema."
    )

    entities = extract_entities(text, {"aspirin", "furosemide"})

    self.assertEqual(entities["medications"], Counter({"aspirin": 1, "furosemide": 1}))
    self.assertEqual(entities["doses"], Counter({"81 mg": 1, "40 mg": 1}))
    self.assertEqual(entities["frequencies"], Counter({"daily": 1, "bid": 1}))
    self.assertEqual(entities["negations"], Counter({"denies": 1}))

  def test_scores_insertions_deletions_and_substitutions(self):
    gold = "Aspirin 81 mg daily. She denies chest pain."
    pred = "Aspirin 325 mg bid. She has chest pain."

    metrics = score_entity_errors(gold, pred, {"aspirin"})

    self.assertEqual(metrics["gold_entities"], 4)
    self.assertEqual(metrics["substitutions"], 2)
    self.assertEqual(metrics["deletions"], 1)
    self.assertEqual(metrics["insertions"], 0)
    self.assertAlmostEqual(metrics["ceer"], 0.75)
    self.assertAlmostEqual(metrics["medication_error_rate"], 0.0)
    self.assertAlmostEqual(metrics["dose_error_rate"], 1.0)
    self.assertAlmostEqual(metrics["frequency_error_rate"], 1.0)
    self.assertAlmostEqual(metrics["negation_error_rate"], 1.0)
    self.assertEqual(metrics["categories"]["doses"]["gold"], 1)
    self.assertEqual(metrics["categories"]["doses"]["substitutions"], 1)
    self.assertEqual(metrics["categories"]["negations"]["deletions"], 1)

  def test_builds_prompt_and_gold_suffix_from_note(self):
    note = (
      "History of present illness includes abdominal pain and edema. "
      "The patient takes aspirin 81 mg daily and furosemide 40 mg bid. "
      "She denies chest pain and shortness of breath. "
      "Plan is to continue diuresis and arrange follow-up."
    )

    prompt, gold = build_prompt_and_gold(note, prompt_fraction=0.5, max_gold_chars=80)

    self.assertTrue(prompt.startswith("History of present illness"))
    self.assertTrue(gold)
    self.assertLessEqual(len(gold), 80)
    self.assertNotEqual(prompt, note)
    self.assertNotIn(gold, prompt)


if __name__ == "__main__":
  unittest.main()
