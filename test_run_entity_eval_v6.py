import unittest
from pathlib import Path

from run_entity_eval_v6 import build_vllm_command, ensure_fresh_output_dir


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


if __name__ == "__main__":
  unittest.main()
