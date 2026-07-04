import argparse
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from entity_eval import (
  completion_logprobs_to_margin_rows,
  keep_prompt_tail_with_token_limit,
  score_entity_errors,
  summarize_margin_rows,
  tag_margin_rows_with_entities,
)
from run_entity_eval_pilot import MEDICATION_LEXICON, aggregate_records
from v6_eval_runtime import (
  apply_method_env_overrides,
  build_method_env,
  build_risk_token_ids,
  build_run_manifest,
  load_fixed_dataset,
  validate_gpu_config,
)


def build_vllm_command(
  *,
  target_model,
  draft_model,
  port,
  tensor_parallel_size,
  max_model_len,
  gpu_memory_utilization,
  num_speculative_tokens,
):
  served_model = Path(target_model).name
  speculative_config = json.dumps({
    "model": draft_model,
    "method": "draft_model",
    "num_speculative_tokens": num_speculative_tokens,
    "parallel_drafting": False,
  })
  return [
    "vllm", "serve", target_model,
    "--host", "127.0.0.1",
    "--port", str(port),
    "--served-model-name", served_model,
    "--tensor-parallel-size", str(tensor_parallel_size),
    "--max-model-len", str(max_model_len),
    "--gpu-memory-utilization", str(gpu_memory_utilization),
    "--trust-remote-code",
    "--speculative-config", speculative_config,
  ]


def ensure_fresh_output_dir(path):
  path = Path(path)
  if path.exists():
    occupied = (
      (path / "run_manifest.json").exists()
      or any(path.glob("*_outputs.jsonl"))
      or any(path.glob("*_summary.json"))
      or any(path.glob("*_server.log"))
    )
    if occupied:
      raise FileExistsError(f"output directory already contains a run: {path}")
  path.mkdir(parents=True, exist_ok=True)
  return path


def select_gold_window(gold_text, eval_gold_chars):
  if eval_gold_chars <= 0:
    raise ValueError("eval_gold_chars must be positive")
  return gold_text[:eval_gold_chars]


def build_server_env(base_env, method_env, cuda_visible_devices):
  env = {
    key: value
    for key, value in base_env.items()
    if not (key.startswith("VLLM_MGSD_") or key.startswith("VLLM_EARS_"))
  }
  env.update(method_env)
  env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
  return env


def attach_runtime_metrics(
    summary,
    records,
    *,
    generation_wall_time_seconds,
    server_ready_wall_time_seconds,
):
  total_completion_tokens = sum(
    int(record.get("usage", {}).get("completion_tokens", 0) or 0)
    for record in records
  )
  if not total_completion_tokens:
    total_completion_tokens = int(
      round(summary.get("samples", 0) * summary.get("avg_completion_tokens", 0.0))
    )
  summary["total_completion_tokens"] = total_completion_tokens
  summary["generation_wall_time_seconds"] = generation_wall_time_seconds
  summary["server_ready_wall_time_seconds"] = server_ready_wall_time_seconds
  summary["completion_tokens_per_second"] = (
    total_completion_tokens / generation_wall_time_seconds
    if generation_wall_time_seconds > 0 else 0.0
  )
  summary["seconds_per_sample"] = (
    generation_wall_time_seconds / summary["samples"]
    if summary.get("samples") else 0.0
  )
  return summary


def start_server(command, env, log_path):
  log_handle = Path(log_path).open("x", encoding="utf-8")
  process = subprocess.Popen(
    command,
    env=env,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    text=True,
  )
  process.log_handle = log_handle
  return process


def wait_ready(port, process, timeout_seconds=900):
  import requests
  deadline = time.time() + timeout_seconds
  while time.time() < deadline:
    if process.poll() is not None:
      raise RuntimeError(f"server exited early with code {process.returncode}")
    try:
      response = requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=2)
      if response.ok:
        return
    except requests.RequestException:
      pass
    time.sleep(3)
  raise RuntimeError("server readiness timeout")


def stop_server(process):
  if process.poll() is None:
    process.send_signal(signal.SIGTERM)
    try:
      process.wait(timeout=30)
    except subprocess.TimeoutExpired:
      process.kill()
      process.wait(timeout=10)
  process.log_handle.close()


def generate_one(port, model_name, prompt, max_tokens, temperature, seed, logprobs):
  import requests
  response = requests.post(
    f"http://127.0.0.1:{port}/v1/completions",
    json={
      "model": model_name,
      "prompt": prompt,
      "max_tokens": max_tokens,
      "temperature": temperature,
      "seed": seed,
      "logprobs": logprobs,
    },
    timeout=1800,
  )
  response.raise_for_status()
  data = response.json()
  choice = data["choices"][0]
  return choice["text"], data.get("usage", {}), choice.get("logprobs")


def _git_commit():
  try:
    return subprocess.check_output(
      ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()
  except (OSError, subprocess.CalledProcessError):
    return "unknown"


def _package_versions():
  versions = {"python": sys.version.split()[0]}
  for package in ("vllm", "transformers", "requests"):
    try:
      versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      versions[package] = "not-installed"
  return versions


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset-path", required=True)
  parser.add_argument("--output-dir", required=True)
  parser.add_argument(
    "--preset",
    choices=("baseline", "v5", "v6-default", "v7-risk-score"),
    required=True,
  )
  parser.add_argument("--target-model", default="/data/models/Qwen3-32B")
  parser.add_argument("--draft-model", default="/data/models/Qwen3-8B")
  parser.add_argument("--tensor-parallel-size", type=int, default=4)
  parser.add_argument("--cuda-visible-devices", default="0,1,2,3")
  parser.add_argument("--base-tolerance", type=float, default=0.2)
  parser.add_argument("--sample-count", type=int, default=20)
  parser.add_argument("--max-tokens", type=int, default=256)
  parser.add_argument("--temperature", type=float, default=0.9)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--logprobs", type=int, default=2)
  parser.add_argument("--num-speculative-tokens", type=int, default=5)
  parser.add_argument("--max-model-len", type=int, default=4096)
  parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
  parser.add_argument("--port", type=int, default=8000)
  parser.add_argument("--low-margin-threshold", type=float, default=0.1)
  parser.add_argument("--gate-debug-max-rows", type=int, default=200000)
  parser.add_argument(
    "--eval-gold-chars",
    type=int,
    default=2048,
    help="Fixed number of gold characters used for entity scoring.",
  )
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  if args.eval_gold_chars <= 0:
    raise SystemExit("--eval-gold-chars must be positive")
  rows = load_fixed_dataset(args.dataset_path, args.sample_count)
  validate_gpu_config(args.cuda_visible_devices, args.tensor_parallel_size)
  output_dir = ensure_fresh_output_dir(args.output_dir)

  from transformers import AutoTokenizer
  tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
  max_prompt_tokens = args.max_model_len - args.max_tokens - 32
  if max_prompt_tokens <= 0:
    raise ValueError("max_model_len must exceed max_tokens by at least 33")
  for row in rows:
    row["prompt"] = keep_prompt_tail_with_token_limit(
      row["prompt"], tokenizer, max_prompt_tokens=max_prompt_tokens
    )

  risk_ids, risk_metadata = build_risk_token_ids(tokenizer, MEDICATION_LEXICON)
  method_env = build_method_env(args.preset, args.base_tolerance, risk_ids)
  method_env = apply_method_env_overrides(method_env, os.environ)
  method_env["VLLM_MGSD_DEBUG_LOG_PATH"] = str(output_dir / "gate_debug.jsonl")
  method_env["VLLM_MGSD_DEBUG_MAX_ROWS"] = str(args.gate_debug_max_rows)

  sampler_spec = importlib.util.find_spec("vllm.v1.sample.rejection_sampler")
  sampler_path = sampler_spec.origin if sampler_spec else "not-found"
  command = build_vllm_command(
    target_model=args.target_model,
    draft_model=args.draft_model,
    port=args.port,
    tensor_parallel_size=args.tensor_parallel_size,
    max_model_len=args.max_model_len,
    gpu_memory_utilization=args.gpu_memory_utilization,
    num_speculative_tokens=args.num_speculative_tokens,
  )
  config = vars(args).copy()
  config["vllm_command"] = command
  manifest = build_run_manifest(
    config=config,
    method_env=method_env,
    dataset_path=args.dataset_path,
    risk_metadata=risk_metadata,
    package_versions=_package_versions(),
    sampler_path=sampler_path,
    git_commit=_git_commit(),
  )
  (output_dir / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
    encoding="utf-8",
  )

  env = build_server_env(os.environ, method_env, args.cuda_visible_devices)
  server_log = output_dir / f"{args.preset}_server.log"
  process = start_server(command, env, server_log)
  output_path = output_dir / f"{args.preset}_outputs.jsonl"
  model_name = Path(args.target_model).name
  records = []
  server_start_time = time.time()
  server_ready_time = None
  generation_start_time = None
  generation_end_time = None
  try:
    wait_ready(args.port, process)
    server_ready_time = time.time()
    generation_start_time = server_ready_time
    with output_path.open("x", encoding="utf-8") as output_handle:
      for index, row in enumerate(rows, start=1):
        print(f"preset={args.preset} sample={index}/{len(rows)}", flush=True)
        output, usage, completion_logprobs = generate_one(
          args.port,
          model_name,
          row["prompt"],
          args.max_tokens,
          args.temperature,
          args.seed + index,
          args.logprobs,
        )
        gold_window = select_gold_window(row["gold"], args.eval_gold_chars)
        metrics = score_entity_errors(gold_window, output, MEDICATION_LEXICON)
        margin_rows = completion_logprobs_to_margin_rows(completion_logprobs or {})
        tagged_rows = tag_margin_rows_with_entities(margin_rows, MEDICATION_LEXICON)
        record = {
          "note_id": row["note_id"],
          "prompt": row["prompt"],
          "gold": row["gold"],
          "gold_window": gold_window,
          "output": output,
          "usage": usage,
          "metrics": metrics,
          "margin_summary": summarize_margin_rows(
            tagged_rows, low_margin_threshold=args.low_margin_threshold
          ),
          "margin_rows": tagged_rows,
        }
        records.append(record)
        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        output_handle.flush()
    generation_end_time = time.time()
  finally:
    stop_server(process)

  summary = aggregate_records(records)
  if generation_start_time is not None and generation_end_time is not None:
    attach_runtime_metrics(
      summary,
      records,
      generation_wall_time_seconds=generation_end_time - generation_start_time,
      server_ready_wall_time_seconds=server_ready_time - server_start_time,
    )
  summary_path = output_dir / f"{args.preset}_summary.json"
  summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
  )
  print(f"summary_path={summary_path}")


if __name__ == "__main__":
  main()
