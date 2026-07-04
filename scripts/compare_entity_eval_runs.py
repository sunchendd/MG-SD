#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


METRICS = (
  ("ceer", "CEER"),
  ("medication_error_rate", "Medication"),
  ("dose_error_rate", "Dose"),
  ("frequency_error_rate", "Frequency"),
  ("negation_error_rate", "Negation"),
  ("completion_tokens_per_second", "tokens/s"),
  ("seconds_per_sample", "sec/sample"),
)


def load_summary(path):
  path = Path(path)
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def infer_method_name(path):
  path = Path(path)
  name = path.name
  if name.endswith("_summary.json"):
    return name[:-len("_summary.json")]
  return path.parent.name or path.stem


def _round(value):
  if value is None:
    return None
  return round(float(value), 6)


def build_comparison_rows(named_summaries):
  if not named_summaries:
    return []
  baseline = named_summaries[0][1]
  baseline_ceer = baseline.get("ceer")
  baseline_tps = baseline.get("completion_tokens_per_second")
  rows = []
  for method, summary in named_summaries:
    row = {"method": method}
    for key, _ in METRICS:
      row[key] = _round(summary.get(key))
    row["delta_ceer"] = _round(
      summary.get("ceer") - baseline_ceer
      if summary.get("ceer") is not None and baseline_ceer is not None else None
    )
    row["delta_tokens_per_second"] = _round(
      summary.get("completion_tokens_per_second") - baseline_tps
      if (
        summary.get("completion_tokens_per_second") is not None
        and baseline_tps is not None
      ) else None
    )
    rows.append(row)
  return rows


def render_markdown(rows):
  headers = [
    "method",
    "CEER ↓",
    "Medication ↓",
    "Dose ↓",
    "Frequency ↓",
    "Negation ↓",
    "tokens/s ↑",
    "sec/sample ↓",
    "ΔCEER",
    "Δtokens/s",
  ]
  keys = [
    "method",
    "ceer",
    "medication_error_rate",
    "dose_error_rate",
    "frequency_error_rate",
    "negation_error_rate",
    "completion_tokens_per_second",
    "seconds_per_sample",
    "delta_ceer",
    "delta_tokens_per_second",
  ]
  lines = [
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join("---" for _ in headers) + " |",
  ]
  for row in rows:
    values = ["" if row.get(key) is None else str(row.get(key)) for key in keys]
    lines.append("| " + " | ".join(values) + " |")
  return "\n".join(lines)


def write_csv(rows, path):
  if not rows:
    return
  with Path(path).open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("summary_paths", nargs="+")
  parser.add_argument("--csv-output")
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  named_summaries = [
    (infer_method_name(path), load_summary(path)) for path in args.summary_paths
  ]
  rows = build_comparison_rows(named_summaries)
  print(render_markdown(rows))
  if args.csv_output:
    write_csv(rows, args.csv_output)


if __name__ == "__main__":
  main()
