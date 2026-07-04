import unittest
from pathlib import Path

from v6_eval_runtime import (
  apply_method_env_overrides,
  build_method_env,
  build_risk_token_ids,
  build_run_manifest,
  load_fixed_dataset,
  validate_gpu_config,
)


class V6EvalRuntimeTests(unittest.TestCase):
  def setUp(self):
    self.fixture_root = Path(__file__).parent / "testdata"
    self.dataset_path = self.fixture_root / "v6_runtime_valid.jsonl"

  def test_loads_and_slices_fixed_dataset(self):
    rows = load_fixed_dataset(self.dataset_path, sample_count=1)
    self.assertEqual([row["note_id"] for row in rows], ["n1"])

  def test_rejects_missing_dataset_fields(self):
    with self.assertRaisesRegex(ValueError, "missing fields"):
      load_fixed_dataset(
        self.fixture_root / "v6_runtime_missing_fields.jsonl",
        sample_count=1,
      )

  def test_rejects_sample_count_larger_than_fixed_dataset(self):
    with self.assertRaisesRegex(ValueError, "contains only 2 rows"):
      load_fixed_dataset(self.dataset_path, sample_count=3)

  def test_requires_one_visible_gpu_per_tp_rank(self):
    with self.assertRaisesRegex(ValueError, "tensor parallel"):
      validate_gpu_config("0,1", tensor_parallel_size=4)

  def test_baseline_disables_all_custom_gates(self):
    env = build_method_env("baseline", base_tolerance=0.2, risk_token_ids={})
    self.assertEqual(env, {})

  def test_v5_disables_v6_and_uses_requested_tolerance(self):
    env = build_method_env("v5", base_tolerance=0.2, risk_token_ids={})
    self.assertEqual(env["VLLM_EARS_BASE_TOLERANCE"], "0.2")
    self.assertEqual(env["VLLM_MGSD_ENABLED"], "1")
    self.assertEqual(env["VLLM_MGSD_V6_ENABLED"], "0")

  def test_v6_default_exports_sorted_risk_ids(self):
    env = build_method_env(
      "v6-default",
      base_tolerance=0.2,
      risk_token_ids={"negation": {3, 1}},
    )
    self.assertEqual(env["VLLM_MGSD_V6_ENABLED"], "1")
    self.assertEqual(env["VLLM_MGSD_NEGATION_TOKEN_IDS"], "1,3")
    self.assertEqual(env["VLLM_MGSD_V6_SAFE_FLOOR"], "0.75")
    self.assertEqual(env["VLLM_MGSD_V6_RHO_RISK"], "0.85")

  def test_v7_risk_score_exports_balanced_defaults(self):
    env = build_method_env(
      "v7-risk-score",
      base_tolerance=0.2,
      risk_token_ids={"medication": {9}, "frequency": {7}},
    )

    self.assertEqual(env["VLLM_MGSD_V6_ENABLED"], "1")
    self.assertEqual(env["VLLM_MGSD_V7_ENABLED"], "1")
    self.assertEqual(env["VLLM_MGSD_V7_LAMBDA"], "0.75")
    self.assertEqual(env["VLLM_MGSD_V7_MIN_GATE"], "0.10")
    self.assertEqual(env["VLLM_MGSD_V7_RHO"], "0.90")
    self.assertEqual(env["VLLM_MGSD_V7_SAFE_FLOOR"], "0.85")
    self.assertEqual(env["VLLM_MGSD_V7_WEIGHT_NEGATION"], "1.00")
    self.assertEqual(env["VLLM_MGSD_V7_WEIGHT_NUMERIC"], "0.80")
    self.assertEqual(env["VLLM_MGSD_V7_WEIGHT_UNIT"], "0.60")
    self.assertEqual(env["VLLM_MGSD_V7_WEIGHT_FREQUENCY"], "0.80")
    self.assertEqual(env["VLLM_MGSD_V7_WEIGHT_MEDICATION"], "0.70")
    self.assertEqual(env["VLLM_MGSD_MEDICATION_TOKEN_IDS"], "9")
    self.assertEqual(env["VLLM_MGSD_FREQUENCY_TOKEN_IDS"], "7")

  def test_applies_only_explicit_tuning_overrides(self):
    env = build_method_env(
      "v6-default",
      base_tolerance=0.2,
      risk_token_ids={"negation": {3, 1}},
    )
    overridden = apply_method_env_overrides(
      env,
      {
        "VLLM_MGSD_ENABLED": "0",
        "VLLM_MGSD_NEGATION_TOKEN_IDS": "999",
        "VLLM_MGSD_FREQUENCY_BACKOFF": "0.25",
        "VLLM_MGSD_V6_RHO_RISK": "0.90",
      },
    )

    self.assertEqual(overridden["VLLM_MGSD_ENABLED"], "1")
    self.assertEqual(overridden["VLLM_MGSD_NEGATION_TOKEN_IDS"], "1,3")
    self.assertEqual(overridden["VLLM_MGSD_FREQUENCY_BACKOFF"], "0.25")
    self.assertEqual(overridden["VLLM_MGSD_V6_RHO_RISK"], "0.90")

  def test_applies_v7_tuning_overrides(self):
    env = build_method_env(
      "v7-risk-score",
      base_tolerance=0.2,
      risk_token_ids={"negation": {3, 1}},
    )
    overridden = apply_method_env_overrides(
      env,
      {
        "VLLM_MGSD_V7_LAMBDA": "0.60",
        "VLLM_MGSD_V7_WEIGHT_MEDICATION": "0.90",
        "VLLM_MGSD_V7_RHO": "0.95",
        "VLLM_MGSD_V7_ENABLED": "0",
      },
    )

    self.assertEqual(overridden["VLLM_MGSD_V7_ENABLED"], "1")
    self.assertEqual(overridden["VLLM_MGSD_V7_LAMBDA"], "0.60")
    self.assertEqual(overridden["VLLM_MGSD_V7_WEIGHT_MEDICATION"], "0.90")
    self.assertEqual(overridden["VLLM_MGSD_V7_RHO"], "0.95")

  def test_builds_stable_risk_ids_and_skips_multi_token_medications(self):
    class FakeTokenizer:
      def encode(self, text, add_special_tokens=False):
        mapping = {
          "aspirin": [11],
          " aspirin": [12],
          "long medicine": [21, 22],
          " long medicine": [23, 24],
        }
        return mapping.get(text, [100 + sum(text.encode("utf-8")) % 1000])

    risk_ids, metadata = build_risk_token_ids(
      FakeTokenizer(),
      {"aspirin", "long medicine"},
    )

    self.assertIn(11, risk_ids["medication"])
    self.assertIn(12, risk_ids["medication"])
    self.assertNotIn(21, risk_ids["medication"])
    self.assertEqual(metadata["skipped_multi_token_medications"], 1)
    self.assertEqual(len(metadata["sha256"]), 64)

  def test_builds_manifest_with_dataset_and_runtime_identity(self):
    manifest = build_run_manifest(
      config={"preset": "v6-default"},
      method_env={"VLLM_MGSD_V6_ENABLED": "1"},
      dataset_path=self.dataset_path,
      risk_metadata={"sha256": "abc", "counts": {"negation": 2}},
      package_versions={"python": "3.12", "vllm": "0.21.0"},
      sampler_path="/tmp/rejection_sampler.py",
      git_commit="deadbeef",
    )

    self.assertEqual(manifest["git_commit"], "deadbeef")
    self.assertEqual(manifest["config"]["preset"], "v6-default")
    self.assertEqual(len(manifest["dataset"]["sha256"]), 64)
    self.assertEqual(manifest["method_env"]["VLLM_MGSD_V6_ENABLED"], "1")


if __name__ == "__main__":
  unittest.main()
