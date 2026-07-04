import hashlib
import json
from pathlib import Path


REQUIRED_DATASET_FIELDS = {"note_id", "prompt", "gold", "gold_entity_counts"}
RISK_ENV_NAMES = {
  "negation": "VLLM_MGSD_NEGATION_TOKEN_IDS",
  "numeric": "VLLM_MGSD_NUMERIC_TOKEN_IDS",
  "unit": "VLLM_MGSD_UNIT_TOKEN_IDS",
  "frequency": "VLLM_MGSD_FREQUENCY_TOKEN_IDS",
  "medication": "VLLM_MGSD_MEDICATION_TOKEN_IDS",
}
NEGATION_TERMS = ("no", "not", "denies", "without", "negative for", "free of")
UNIT_TERMS = ("mg", "mcg", "g", "ml", "l", "unit", "units", "meq", "mmol")
FREQUENCY_TERMS = ("daily", "bid", "tid", "qid", "qhs", "prn", "nightly", "weekly")
NUMERIC_TERMS = tuple(str(value) for value in range(1001))
OVERRIDABLE_METHOD_ENV_NAMES = {
  "VLLM_MGSD_SOFT_TAU",
  "VLLM_MGSD_DRAFT_MIN_RATIO",
  "VLLM_MGSD_RATIO_TAU",
  "VLLM_MGSD_NEGATION_BACKOFF",
  "VLLM_MGSD_NUMERIC_BACKOFF",
  "VLLM_MGSD_UNIT_BACKOFF",
  "VLLM_MGSD_FREQUENCY_BACKOFF",
  "VLLM_MGSD_MEDICATION_BACKOFF",
  "VLLM_MGSD_V6_SAFE_FLOOR",
  "VLLM_MGSD_V6_DELTA_SAFE",
  "VLLM_MGSD_V6_TAU_M_SAFE",
  "VLLM_MGSD_V6_RHO_SAFE",
  "VLLM_MGSD_V6_TAU_R_SAFE",
  "VLLM_MGSD_V6_DELTA_RISK",
  "VLLM_MGSD_V6_TAU_M_RISK",
  "VLLM_MGSD_V6_RHO_RISK",
  "VLLM_MGSD_V6_TAU_R_RISK",
  "VLLM_MGSD_V7_LAMBDA",
  "VLLM_MGSD_V7_MIN_GATE",
  "VLLM_MGSD_V7_DELTA",
  "VLLM_MGSD_V7_TAU_M",
  "VLLM_MGSD_V7_RHO",
  "VLLM_MGSD_V7_TAU_R",
  "VLLM_MGSD_V7_SAFE_FLOOR",
  "VLLM_MGSD_V7_WEIGHT_NEGATION",
  "VLLM_MGSD_V7_WEIGHT_NUMERIC",
  "VLLM_MGSD_V7_WEIGHT_UNIT",
  "VLLM_MGSD_V7_WEIGHT_FREQUENCY",
  "VLLM_MGSD_V7_WEIGHT_MEDICATION",
}


def load_fixed_dataset(path, sample_count):
  path = Path(path)
  if sample_count <= 0:
    raise ValueError("sample_count must be positive")
  if not path.is_file():
    raise FileNotFoundError(f"dataset not found: {path}")

  rows = []
  with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, start=1):
      if not line.strip():
        continue
      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
      missing = REQUIRED_DATASET_FIELDS - set(row)
      if missing:
        raise ValueError(
          f"dataset line {line_number} missing fields: {sorted(missing)}"
        )
      rows.append(row)

  if not rows:
    raise ValueError(f"dataset is empty: {path}")
  if sample_count > len(rows):
    raise ValueError(
      f"requested {sample_count} samples, but dataset contains only {len(rows)} rows"
    )
  return rows[:sample_count]


def validate_gpu_config(cuda_visible_devices, tensor_parallel_size):
  gpu_ids = [value.strip() for value in cuda_visible_devices.split(",") if value.strip()]
  if len(gpu_ids) != tensor_parallel_size:
    raise ValueError(
      "tensor parallel size must equal the number of visible GPU IDs: "
      f"tp={tensor_parallel_size}, gpu_ids={gpu_ids}"
    )
  if len(set(gpu_ids)) != len(gpu_ids):
    raise ValueError(f"visible GPU IDs must be unique: {gpu_ids}")
  return gpu_ids


def build_method_env(preset, base_tolerance, risk_token_ids):
  if preset not in {"baseline", "v5", "v6-default", "v7-risk-score"}:
    raise ValueError(f"unsupported preset: {preset}")
  if preset == "baseline":
    return {}

  env = {
    "VLLM_EARS_BASE_TOLERANCE": str(base_tolerance),
    "VLLM_MGSD_ENABLED": "1",
    "VLLM_MGSD_MARGIN_DELTA": "0.0",
    "VLLM_MGSD_SOFT_TAU": "0.03",
    "VLLM_MGSD_DRAFT_MIN_RATIO": "0.85",
    "VLLM_MGSD_RATIO_TAU": "0.05",
    "VLLM_MGSD_RISK_ONLY": "0",
    "VLLM_MGSD_V6_ENABLED": "1" if preset in {"v6-default", "v7-risk-score"} else "0",
    "VLLM_MGSD_V7_ENABLED": "1" if preset == "v7-risk-score" else "0",
    "VLLM_MGSD_NEGATION_BACKOFF": "0.0",
    "VLLM_MGSD_NUMERIC_BACKOFF": "0.35",
    "VLLM_MGSD_UNIT_BACKOFF": "0.35",
    "VLLM_MGSD_FREQUENCY_BACKOFF": "0.50",
    "VLLM_MGSD_MEDICATION_BACKOFF": "1.0",
  }
  if preset == "v6-default":
    env.update({
      "VLLM_MGSD_V6_SAFE_FLOOR": "0.75",
      "VLLM_MGSD_V6_DELTA_SAFE": "-0.02",
      "VLLM_MGSD_V6_TAU_M_SAFE": "0.05",
      "VLLM_MGSD_V6_RHO_SAFE": "0.75",
      "VLLM_MGSD_V6_TAU_R_SAFE": "0.12",
      "VLLM_MGSD_V6_DELTA_RISK": "0.00",
      "VLLM_MGSD_V6_TAU_M_RISK": "0.03",
      "VLLM_MGSD_V6_RHO_RISK": "0.85",
      "VLLM_MGSD_V6_TAU_R_RISK": "0.05",
    })
  if preset == "v7-risk-score":
    env.update({
      "VLLM_MGSD_V6_SAFE_FLOOR": "0.75",
      "VLLM_MGSD_V6_DELTA_SAFE": "-0.02",
      "VLLM_MGSD_V6_TAU_M_SAFE": "0.05",
      "VLLM_MGSD_V6_RHO_SAFE": "0.75",
      "VLLM_MGSD_V6_TAU_R_SAFE": "0.12",
      "VLLM_MGSD_V6_DELTA_RISK": "0.00",
      "VLLM_MGSD_V6_TAU_M_RISK": "0.03",
      "VLLM_MGSD_V6_RHO_RISK": "0.85",
      "VLLM_MGSD_V6_TAU_R_RISK": "0.05",
      "VLLM_MGSD_V7_LAMBDA": "0.75",
      "VLLM_MGSD_V7_MIN_GATE": "0.10",
      "VLLM_MGSD_V7_DELTA": "0.00",
      "VLLM_MGSD_V7_TAU_M": "0.03",
      "VLLM_MGSD_V7_RHO": "0.90",
      "VLLM_MGSD_V7_TAU_R": "0.05",
      "VLLM_MGSD_V7_SAFE_FLOOR": "0.85",
      "VLLM_MGSD_V7_WEIGHT_NEGATION": "1.00",
      "VLLM_MGSD_V7_WEIGHT_NUMERIC": "0.80",
      "VLLM_MGSD_V7_WEIGHT_UNIT": "0.60",
      "VLLM_MGSD_V7_WEIGHT_FREQUENCY": "0.80",
      "VLLM_MGSD_V7_WEIGHT_MEDICATION": "0.70",
    })
  for category, env_name in RISK_ENV_NAMES.items():
    values = risk_token_ids.get(category, set())
    env[env_name] = ",".join(str(value) for value in sorted(values))
  return env


def apply_method_env_overrides(method_env, override_env):
  env = dict(method_env)
  for name in OVERRIDABLE_METHOD_ENV_NAMES:
    if name in env and name in override_env:
      env[name] = str(override_env[name])
  return env


def _single_token_ids(tokenizer, terms):
  token_ids = set()
  represented_terms = set()
  for term in sorted(set(terms)):
    for surface in (term, f" {term}"):
      encoded = tokenizer.encode(surface, add_special_tokens=False)
      if len(encoded) == 1:
        token_ids.add(int(encoded[0]))
        represented_terms.add(term)
  return token_ids, represented_terms


def build_risk_token_ids(tokenizer, medication_lexicon):
  category_terms = {
    "negation": NEGATION_TERMS,
    "numeric": NUMERIC_TERMS,
    "unit": UNIT_TERMS,
    "frequency": FREQUENCY_TERMS,
  }
  risk_ids = {}
  for category, terms in category_terms.items():
    risk_ids[category], _ = _single_token_ids(tokenizer, terms)

  normalized_medications = {
    str(term).strip().lower() for term in medication_lexicon if str(term).strip()
  }
  medication_ids, represented_medications = _single_token_ids(
    tokenizer, normalized_medications
  )
  risk_ids["medication"] = medication_ids

  canonical = {
    category: sorted(values) for category, values in sorted(risk_ids.items())
  }
  digest = hashlib.sha256(
    json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
  ).hexdigest()
  metadata = {
    "counts": {category: len(values) for category, values in canonical.items()},
    "sha256": digest,
    "skipped_multi_token_medications": len(
      normalized_medications - represented_medications
    ),
  }
  return risk_ids, metadata


def _sha256_file(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def build_run_manifest(
  *,
  config,
  method_env,
  dataset_path,
  risk_metadata,
  package_versions,
  sampler_path,
  git_commit,
):
  dataset_path = Path(dataset_path)
  return {
    "git_commit": git_commit,
    "config": dict(config),
    "dataset": {
      "path": str(dataset_path),
      "sha256": _sha256_file(dataset_path),
    },
    "method_env": dict(sorted(method_env.items())),
    "risk_token_ids": risk_metadata,
    "package_versions": dict(package_versions),
    "sampler_path": str(sampler_path),
  }
