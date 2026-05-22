import sys
import unittest
from collections import Counter
from math import exp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from entity_eval import (
  build_prompt_and_gold,
  summarize_margin_rows,
  tag_margin_rows_with_entities,
  completion_logprobs_to_margin_rows,
  extract_entities,
  keep_prompt_tail_with_token_limit,
  score_entity_errors,
)

from run_entity_eval_pilot import dedupe_rows_by_note_prefix, is_clean_discharge_medications_section

from medication_lexicon import normalize_medication_text, build_medication_alias_set


class EntityEvalTests(unittest.TestCase):
  def test_converts_completion_logprobs_into_margin_rows(self):
    logprobs = {
      "tokens": ["Aspirin", " 81", " mg"],
      "token_logprobs": [-0.1, -0.2, -0.05],
      "top_logprobs": [
        {"Aspirin": -0.1, "Acetaminophen": -1.2},
        {" 81": -0.2, " 325": -0.3},
        {" mg": -0.05, " mcg": -1.0},
      ],
    }

    rows = completion_logprobs_to_margin_rows(logprobs)

    self.assertEqual(len(rows), 3)
    self.assertEqual(rows[0]["token"], "Aspirin")
    self.assertEqual(rows[0]["top1_token"], "Aspirin")
    self.assertEqual(rows[0]["top2_token"], "Acetaminophen")
    self.assertAlmostEqual(rows[0]["top1_prob"], exp(-0.1))
    self.assertAlmostEqual(rows[0]["top2_prob"], exp(-1.2))
    self.assertAlmostEqual(rows[0]["margin"], exp(-0.1) - exp(-1.2))
    self.assertEqual(rows[1]["top2_token"], " 325")

  def test_tags_entity_tokens_and_summarizes_low_margin_ratio(self):
    rows = [
      {"token": "Aspirin", "margin": 0.04},
      {"token": " 81", "margin": 0.03},
      {"token": " mg", "margin": 0.02},
      {"token": " daily", "margin": 0.08},
      {"token": " stable", "margin": 0.30},
    ]

    tagged = tag_margin_rows_with_entities(rows, {"aspirin"})
    summary = summarize_margin_rows(tagged, low_margin_threshold=0.1)

    self.assertEqual([row["entity_type"] for row in tagged],
                     ["medication", "dose", "dose", "frequency", None])
    self.assertEqual(summary["entity_tokens"], 4)
    self.assertEqual(summary["non_entity_tokens"], 1)
    self.assertEqual(summary["low_margin_entity_tokens"], 4)
    self.assertEqual(summary["low_margin_non_entity_tokens"], 0)
    self.assertAlmostEqual(summary["low_margin_entity_rate"], 1.0)
    self.assertAlmostEqual(summary["low_margin_non_entity_rate"], 0.0)

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



class DatasetSelectionTests(unittest.TestCase):
  def test_keeps_only_one_row_per_note_prefix(self):
    rows = [
      {"note_id": "10000032-DS-21", "prompt": "p1", "gold": "Discharge Medications: 1. Aspirin 81 mg PO DAILY"},
      {"note_id": "10000032-DS-22", "prompt": "p2", "gold": "Discharge Medications: 1. Furosemide 40 mg PO DAILY"},
      {"note_id": "10000935-DS-18", "prompt": "p3", "gold": "Discharge Medications: 1. Metformin 500 mg PO BID"},
    ]

    deduped = dedupe_rows_by_note_prefix(rows)

    self.assertEqual([row["note_id"] for row in deduped], [
      "10000032-DS-21",
      "10000935-DS-18",
    ])

  def test_rejects_non_clean_discharge_medication_sections(self):
    dirty = (
      "Discharge Medications: 1. Aspirin 81 mg PO DAILY "
      "# CODE: full confirmed Follow-up with PCP next week"
    )
    clean = (
      "Discharge Medications: 1. Aspirin 81 mg PO DAILY "
      "2. Furosemide 40 mg PO DAILY 3. Metformin 500 mg PO BID"
    )

    self.assertFalse(is_clean_discharge_medications_section(dirty))
    self.assertTrue(is_clean_discharge_medications_section(clean))


class MedicationLexiconTests(unittest.TestCase):
  def test_normalizes_parenthesized_brand_aliases(self):
    self.assertEqual(
      normalize_medication_text("Emtricitabine-Tenofovir (Truvada)"),
      "emtricitabine tenofovir truvada",
    )

  def test_alias_set_contains_common_surface_forms(self):
    aliases = build_medication_alias_set({"acetaminophen", "furosemide"})
    self.assertIn("acetaminophen", aliases)
    self.assertIn("furosemide", aliases)

  def test_extracts_multiword_parenthesized_alias(self):
    text = "Patient is on Emtricitabine-Tenofovir (Truvada) 200 mg daily."
    entities = extract_entities(text, {"Emtricitabine-Tenofovir (Truvada)"})
    # normalize_medication_text lowercases and removes punctuation into spaces
    self.assertEqual(entities["medications"], Counter({"emtricitabine tenofovir truvada": 1}))

  def test_tags_margin_rows_with_multiword_alias(self):
    rows = [
      {"token": "Emtricitabine-Tenofovir", "margin": 0.05},
      {"token": " (Truvada)", "margin": 0.03},
      {"token": " 200", "margin": 0.02},
      {"token": " mg", "margin": 0.01},
    ]
    tagged = tag_margin_rows_with_entities(rows, {"Emtricitabine-Tenofovir (Truvada)"})
    self.assertEqual([r["entity_type"] for r in tagged], ["medication", "medication", "dose", "dose"])

  def test_does_not_tag_empty_normalized_tokens_as_medication(self):
    # Alias "A-B" normalizes to "a b"; the comma normalizes to empty string.
    # Expectation: rows for "A" and "B" are tagged as medication, comma is not.
    rows = [
      {"token": "A", "margin": 0.05},
      {"token": ",", "margin": 0.03},
      {"token": "B", "margin": 0.02},
    ]
    tagged = tag_margin_rows_with_entities(rows, {"A-B"})
    self.assertEqual([r["entity_type"] for r in tagged], ["medication", None, "medication"])


if __name__ == "__main__":
  unittest.main()
