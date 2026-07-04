import importlib.util
import json
import unittest
import uuid
from pathlib import Path


def load_script(name):
  path = Path(__file__).parent / "scripts" / f"{name}.py"
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class CompareEntityEvalRunsTests(unittest.TestCase):
  def test_builds_markdown_rows_with_baseline_deltas(self):
    compare = load_script("compare_entity_eval_runs")
    baseline = {
      "ceer": 0.80,
      "medication_error_rate": 0.70,
      "dose_error_rate": 0.90,
      "frequency_error_rate": 0.60,
      "negation_error_rate": 1.00,
      "completion_tokens_per_second": 32.0,
      "seconds_per_sample": 8.0,
    }
    candidate = dict(baseline)
    candidate["ceer"] = 0.78
    candidate["completion_tokens_per_second"] = 36.0

    rows = compare.build_comparison_rows([
      ("baseline", baseline),
      ("v7", candidate),
    ])
    markdown = compare.render_markdown(rows)

    self.assertEqual(rows[0]["method"], "baseline")
    self.assertEqual(rows[1]["delta_ceer"], -0.02)
    self.assertEqual(rows[1]["delta_tokens_per_second"], 4.0)
    self.assertIn("| method |", markdown)
    self.assertIn("v7", markdown)


class AnalyzeGateDebugTests(unittest.TestCase):
  def setUp(self):
    self.tmp_root = Path(__file__).parent / ".tmp_test_analysis_scripts"
    self.tmp_root.mkdir(exist_ok=True)

  def test_summarizes_empty_gate_debug_file(self):
    analyze = load_script("analyze_gate_debug")
    path = self.tmp_root / f"gate_debug_empty_{uuid.uuid4().hex}.jsonl"
    path.write_text("", encoding="utf-8")

    summary = analyze.summarize_gate_debug(path)

    self.assertEqual(summary["total_rows"], 0)
    self.assertEqual(summary["groups"], {})

  def test_summarizes_risk_and_non_risk_gate_debug_rows(self):
    analyze = load_script("analyze_gate_debug")
    rows = [
      {
        "is_risk_token": True,
        "risk_is_medication": True,
        "accepted": True,
        "risk_gate": 0.2,
        "safe_gate": None,
      },
      {
        "is_risk_token": False,
        "accepted": False,
        "risk_gate": None,
        "safe_gate": 0.9,
      },
    ]
    path = self.tmp_root / f"gate_debug_rows_{uuid.uuid4().hex}.jsonl"
    path.write_text(
      "".join(json.dumps(row) + "\n" for row in rows),
      encoding="utf-8",
    )

    summary = analyze.summarize_gate_debug(path)

    self.assertEqual(summary["total_rows"], 2)
    self.assertEqual(summary["groups"]["risk"]["rows"], 1)
    self.assertEqual(summary["groups"]["risk"]["acceptance_rate"], 1.0)
    self.assertEqual(summary["groups"]["non_risk"]["acceptance_rate"], 0.0)
    self.assertEqual(summary["groups"]["medication"]["mean_gate"], 0.2)


if __name__ == "__main__":
  unittest.main()
