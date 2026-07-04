#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


CATEGORY_FLAGS = {
  "negation": "risk_is_negation",
  "numeric": "risk_is_numeric",
  "unit": "risk_is_unit",
  "frequency": "risk_is_frequency",
  "medication": "risk_is_medication",
}


def _new_group():
  return {"rows": 0, "accepted": 0, "gate_sum": 0.0, "gate_count": 0}


def _add(groups, name, row):
  group = groups.setdefault(name, _new_group())
  group["rows"] += 1
  if row.get("accepted") is True:
    group["accepted"] += 1
  gate = row.get("risk_gate") if row.get("is_risk_token") else row.get("safe_gate")
  if gate is None:
    gate = row.get("gate_scale")
  if gate is not None:
    group["gate_sum"] += float(gate)
    group["gate_count"] += 1


def _finalize(groups):
  finalized = {}
  for name, group in sorted(groups.items()):
    rows = group["rows"]
    gate_count = group["gate_count"]
    finalized[name] = {
      "rows": rows,
      "accepted": group["accepted"],
      "acceptance_rate": group["accepted"] / rows if rows else 0.0,
      "mean_gate": group["gate_sum"] / gate_count if gate_count else None,
    }
  return finalized


def summarize_gate_debug(path):
  path = Path(path)
  groups = {}
  total_rows = 0
  if not path.is_file():
    raise FileNotFoundError(f"gate debug file not found: {path}")
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
      total_rows += 1
      if row.get("is_risk_token"):
        _add(groups, "risk", row)
        for category, flag_name in CATEGORY_FLAGS.items():
          if row.get(flag_name):
            _add(groups, category, row)
      else:
        _add(groups, "non_risk", row)
  return {"total_rows": total_rows, "groups": _finalize(groups)}


def render_markdown(summary):
  lines = [
    "| group | rows | accepted | acceptance_rate | mean_gate |",
    "| --- | --- | --- | --- | --- |",
  ]
  for name, group in summary["groups"].items():
    mean_gate = group["mean_gate"]
    lines.append(
      "| {group} | {rows} | {accepted} | {rate:.6f} | {gate} |".format(
        group=name,
        rows=group["rows"],
        accepted=group["accepted"],
        rate=group["acceptance_rate"],
        gate="" if mean_gate is None else f"{mean_gate:.6f}",
      )
    )
  return "\n".join(lines)


def parse_args(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("gate_debug_path")
  parser.add_argument("--json-output")
  return parser.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  summary = summarize_gate_debug(args.gate_debug_path)
  print(render_markdown(summary))
  if args.json_output:
    Path(args.json_output).write_text(
      json.dumps(summary, ensure_ascii=False, indent=2),
      encoding="utf-8",
    )


if __name__ == "__main__":
  main()
