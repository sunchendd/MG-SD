import argparse
import csv
import gzip
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer

from entity_eval import (
  build_prompt_and_gold,
  extract_entities,
  keep_prompt_tail_with_token_limit,
  score_entity_errors,
)


MEDICATION_LEXICON = {
  "acetaminophen", "acyclovir", "alprazolam", "amiodarone", "amlodipine",
  "apixaban", "aspirin", "atenolol", "atorvastatin", "bisacodyl",
  "carvedilol", "ciprofloxacin", "citalopram", "clonazepam",
  "clopidogrel", "cyanocobalamin", "dexAMETHasone".lower(), "diazepam",
  "digoxin", "duloxetine", "enoxaparin", "famotidine", "fluoxetine",
  "furosemide", "gabapentin", "glipizide", "hydralazine", "hydrochlorothiazide",
  "ibuprofen", "lasix", "levetiracetam", "levofloxacin", "levothyroxine",
  "lisinopril", "lorazepam", "metformin", "metronidazole", "mirtazapine",
  "morphine", "omeprazole", "ondansetron", "oxycodone", "pantoprazole",
  "pravastatin", "prednisone", "ranitidine", "senna", "sertraline",
  "simvastatin", "spironolactone", "tacrolimus", "tamsulosin", "torsemide",
  "tramadol", "trazodone", "valsartan", "vancomycin", "warfarin",
}

SECTION_MARKERS = [
  "discharge medications:",
  "medications on discharge:",
  "discharge disposition:",
  "discharge diagnosis:",
  "medications:",
]


def prepare_dataset(output_path, sample_count, prompt_fraction, max_gold_chars):
  source = Path("/home/scd/mimic-iv-note/note/discharge.csv.gz")
  rows = []
  with gzip.open(source, "rt") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
      text = re.sub(r"\s+", " ", row["text"]).strip()
      if len(text) < 1800:
        continue
      lower = text.lower()
      marker_positions = [
        lower.find(marker) for marker in SECTION_MARKERS if lower.find(marker) > 700
      ]
      if marker_positions:
        split_at = min(marker_positions)
        prompt = text[:split_at].strip()
        gold = text[split_at:split_at + max_gold_chars].strip()
      else:
        prompt, gold = build_prompt_and_gold(
          text,
          prompt_fraction=prompt_fraction,
          max_gold_chars=max_gold_chars,
        )
      if not prompt or not gold:
        continue
      score_region = gold[:800]
      gold_entities = extract_entities(score_region, MEDICATION_LEXICON)
      if sum(gold_entities["medications"].values()) < 2:
        continue
      if sum(gold_entities["doses"].values()) < 2:
        continue
      if (sum(gold_entities["frequencies"].values())
          + sum(gold_entities["negations"].values())) < 1:
        continue

      rows.append({
        "note_id": row["note_id"],
        "prompt": prompt,
        "gold": gold,
        "gold_entity_counts": {
          key: int(sum(value.values()))
          for key, value in gold_entities.items()
        },
      })
      if len(rows) >= sample_count:
        break

  output_path.parent.mkdir(parents=True, exist_ok=True)
  with output_path.open("w", encoding="utf-8") as handle:
    for row in rows:
      handle.write(json.dumps(row, ensure_ascii=False) + "\n")
  return rows


def start_server(mode, port, log_path):
  env = os.environ.copy()
  env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  env["CUDA_VISIBLE_DEVICES"] = "0,1"
  env["HF_ENDPOINT"] = "https://hf-mirror.com"
  env.pop("VLLM_EARS_BASE_TOLERANCE", None)
  env.pop("VLLM_MGSD_ENABLED", None)
  env.pop("VLLM_MGSD_MARGIN_DELTA", None)

  if mode in {"ears", "mgsd"}:
    env["VLLM_EARS_BASE_TOLERANCE"] = "0.1"
  if mode == "mgsd":
    env["VLLM_MGSD_ENABLED"] = "1"
    env["VLLM_MGSD_MARGIN_DELTA"] = "0.1"

  cmd = [
    "vllm", "serve", "/data/models/Qwen3-32B",
    "--host", "127.0.0.1",
    "--port", str(port),
    "--served-model-name", "Qwen3-32B",
    "--tensor-parallel-size", "2",
    "--max-model-len", "4096",
    "--gpu-memory-utilization", "0.85",
    "--trust-remote-code",
    "--speculative-config",
    '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}',
  ]
  log_handle = log_path.open("w", encoding="utf-8")
  process = subprocess.Popen(
    cmd,
    env=env,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    text=True,
  )
  process.log_handle = log_handle
  return process


def wait_ready(port, process):
  deadline = time.time() + 900
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
  if process.poll() is not None:
    log_handle = getattr(process, "log_handle", None)
    if log_handle:
      log_handle.close()
    return
  process.send_signal(signal.SIGTERM)
  try:
    process.wait(timeout=30)
  except subprocess.TimeoutExpired:
    process.kill()
    process.wait(timeout=10)
  log_handle = getattr(process, "log_handle", None)
  if log_handle:
    log_handle.close()


def generate_one(port, prompt, max_tokens, temperature, seed):
  payload = {
    "model": "Qwen3-32B",
    "prompt": prompt,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "seed": seed,
  }
  response = requests.post(
    f"http://127.0.0.1:{port}/v1/completions",
    json=payload,
    timeout=1800,
  )
  response.raise_for_status()
  data = response.json()
  return data["choices"][0]["text"], data.get("usage", {})


def aggregate_records(records):
  summary = {
    "samples": len(records),
    "gold_entities": 0,
    "substitutions": 0,
    "deletions": 0,
    "insertions": 0,
    "categories": {
      "medications": {"gold": 0, "errors": 0},
      "doses": {"gold": 0, "errors": 0},
      "frequencies": {"gold": 0, "errors": 0},
      "negations": {"gold": 0, "errors": 0},
    },
    "avg_prompt_tokens": 0.0,
    "avg_completion_tokens": 0.0,
  }
  for record in records:
    metrics = record["metrics"]
    summary["gold_entities"] += metrics["gold_entities"]
    summary["substitutions"] += metrics["substitutions"]
    summary["deletions"] += metrics["deletions"]
    summary["insertions"] += metrics["insertions"]
    summary["avg_prompt_tokens"] += record["usage"].get("prompt_tokens", 0)
    summary["avg_completion_tokens"] += record["usage"].get("completion_tokens", 0)
    for key, category in metrics["categories"].items():
      summary["categories"][key]["gold"] += category["gold"]
      summary["categories"][key]["errors"] += (
        category["substitutions"] + category["deletions"] + category["insertions"]
      )

  if records:
    summary["avg_prompt_tokens"] /= len(records)
    summary["avg_completion_tokens"] /= len(records)
  total_errors = summary["substitutions"] + summary["deletions"] + summary["insertions"]
  summary["ceer"] = (
    total_errors / summary["gold_entities"] if summary["gold_entities"] else 0.0
  )
  summary["medication_error_rate"] = (
    summary["categories"]["medications"]["errors"]
    / summary["categories"]["medications"]["gold"]
    if summary["categories"]["medications"]["gold"] else 0.0
  )
  summary["dose_error_rate"] = (
    summary["categories"]["doses"]["errors"]
    / summary["categories"]["doses"]["gold"]
    if summary["categories"]["doses"]["gold"] else 0.0
  )
  summary["frequency_error_rate"] = (
    summary["categories"]["frequencies"]["errors"]
    / summary["categories"]["frequencies"]["gold"]
    if summary["categories"]["frequencies"]["gold"] else 0.0
  )
  summary["negation_error_rate"] = (
    summary["categories"]["negations"]["errors"]
    / summary["categories"]["negations"]["gold"]
    if summary["categories"]["negations"]["gold"] else 0.0
  )
  return summary


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--sample-count", type=int, default=12)
  parser.add_argument("--prompt-fraction", type=float, default=0.45)
  parser.add_argument("--max-gold-chars", type=int, default=1400)
  parser.add_argument("--max-tokens", type=int, default=256)
  parser.add_argument("--temperature", type=float, default=0.9)
  parser.add_argument("--seed", type=int, default=123)
  parser.add_argument("--prepare-only", action="store_true")
  args = parser.parse_args()

  root = Path("/home/scd/MG-SD/entity_eval")
  root.mkdir(parents=True, exist_ok=True)
  dataset_path = root / "pilot_dataset.jsonl"
  dataset = prepare_dataset(
    dataset_path,
    sample_count=args.sample_count,
    prompt_fraction=args.prompt_fraction,
    max_gold_chars=args.max_gold_chars,
  )
  tokenizer = AutoTokenizer.from_pretrained("/data/models/Qwen3-32B", trust_remote_code=True)
  max_prompt_tokens = 4096 - args.max_tokens - 32
  for row in dataset:
    row["prompt"] = keep_prompt_tail_with_token_limit(
      row["prompt"],
      tokenizer,
      max_prompt_tokens=max_prompt_tokens,
    )
  print(f"prepared_dataset={dataset_path}")
  print(f"prepared_samples={len(dataset)}")

  if args.prepare_only:
    return

  summaries = {}
  for mode in ("baseline", "ears", "mgsd"):
    print(f"running_mode={mode}")
    log_path = root / f"{mode}_server.log"
    output_path = root / f"{mode}_outputs.jsonl"
    process = start_server(mode, 8000, log_path)
    try:
      wait_ready(8000, process)
      records = []
      for idx, row in enumerate(dataset, start=1):
        print(f"mode={mode} sample={idx}/{len(dataset)}")
        output, usage = generate_one(
          8000,
          row["prompt"],
          max_tokens=args.max_tokens,
          temperature=args.temperature,
          seed=args.seed + idx,
        )
        gold_window = row["gold"][:max(len(output), 1)]
        metrics = score_entity_errors(gold_window, output, MEDICATION_LEXICON)
        record = {
          "note_id": row["note_id"],
          "prompt": row["prompt"],
          "gold": row["gold"],
          "gold_window": gold_window,
          "output": output,
          "usage": usage,
          "metrics": metrics,
        }
        records.append(record)
      with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
          handle.write(json.dumps(record, ensure_ascii=False) + "\n")
      summaries[mode] = aggregate_records(records)
      print(json.dumps({mode: summaries[mode]}, ensure_ascii=False, indent=2))
    finally:
      stop_server(process)

  summary_path = root / "pilot_summary.json"
  summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"summary_path={summary_path}")


if __name__ == "__main__":
  main()
