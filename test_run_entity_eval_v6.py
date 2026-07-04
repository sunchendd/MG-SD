import unittest
from pathlib import Path

from run_entity_eval_v6 import (
  attach_runtime_metrics,
  build_server_env,
  build_vllm_command,
  ensure_fresh_output_dir,
  parse_args,
  select_gold_window,
)


class RunEntityEvalV6Tests(unittest.TestCase):
  def test_builds_vllm_command_with_reproducible_model_configuration(self):
    command = build_vllm_command(
      target_model="/data/models/Qwen3-32B",
      draft_model="/data/models/Qwen3-8B",
      port=8000,
      tensor_parallel_size=4,
      max_model_len=4096,
      gpu_memory_utilization=0.85,
      num_speculative_tokens=5,
    )

    rendered = " ".join(command)
    self.assertIn("/data/models/Qwen3-32B", rendered)
    self.assertIn("/data/models/Qwen3-8B", rendered)
    self.assertIn("--tensor-parallel-size 4", rendered)
    self.assertIn('"num_speculative_tokens": 5', rendered)

  def test_rejects_output_directory_with_existing_manifest(self):
    occupied = Path(__file__).parent / "testdata" / "occupied_v6_output"
    with self.assertRaises(FileExistsError):
      ensure_fresh_output_dir(occupied)

  def test_selects_fixed_gold_window_independent_of_output_length(self):
    gold = "abcdefghij"

    self.assertEqual("abcd", select_gold_window(gold, eval_gold_chars=4))
    self.assertEqual("abcdefghij", select_gold_window(gold, eval_gold_chars=50))

  def test_accepts_v7_risk_score_preset(self):
    args = parse_args([
      "--dataset-path", "dataset.jsonl",
      "--output-dir", "out",
      "--preset", "v7-risk-score",
    ])

    self.assertEqual(args.preset, "v7-risk-score")

  def test_build_server_env_strips_stale_custom_gate_variables(self):
    env = build_server_env(
      {
        "PATH": "/usr/bin",
        "VLLM_MGSD_ENABLED": "1",
        "VLLM_EARS_BASE_TOLERANCE": "0.2",
      },
      method_env={},
      cuda_visible_devices="1,2,3,4",
    )

    self.assertEqual(env["PATH"], "/usr/bin")
    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1,2,3,4")
    self.assertNotIn("VLLM_MGSD_ENABLED", env)
    self.assertNotIn("VLLM_EARS_BASE_TOLERANCE", env)

  def test_attach_runtime_metrics_uses_completion_tokens_from_usage(self):
    summary = {"samples": 2, "avg_completion_tokens": 10.0}
    records = [
      {"usage": {"completion_tokens": 7}},
      {"usage": {"completion_tokens": 13}},
    ]

    attach_runtime_metrics(
      summary,
      records,
      generation_wall_time_seconds=4.0,
      server_ready_wall_time_seconds=6.0,
    )

    self.assertEqual(summary["total_completion_tokens"], 20)
    self.assertEqual(summary["generation_wall_time_seconds"], 4.0)
    self.assertEqual(summary["server_ready_wall_time_seconds"], 6.0)
    self.assertEqual(summary["completion_tokens_per_second"], 5.0)
    self.assertEqual(summary["seconds_per_sample"], 2.0)


if __name__ == "__main__":
  unittest.main()
