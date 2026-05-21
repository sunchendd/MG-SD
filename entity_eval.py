import re
from collections import Counter


DOSE_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|l|units?|meq|mmol)\b", re.I)
FREQ_RE = re.compile(
  r"\b(?:daily|bid|tid|qid|qhs|q\d+h|prn|nightly|weekly|every\s+\d+\s+hours?)\b",
  re.I,
)
NEG_RE = re.compile(r"\b(?:no|not|denies|without|negative\s+for|free\s+of)\b", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")


def keep_prompt_tail_with_token_limit(text, tokenizer, max_prompt_tokens):
  token_ids = tokenizer.encode(text, add_special_tokens=False)
  if len(token_ids) <= max_prompt_tokens:
    return text
  trimmed_ids = token_ids[-max_prompt_tokens:]
  return tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()


def build_prompt_and_gold(note, prompt_fraction=0.4, max_gold_chars=1200):
  text = re.sub(r"\s+", " ", note).strip()
  if not text:
    return "", ""

  split_at = max(1, min(len(text) - 1, int(len(text) * prompt_fraction)))
  boundary = text.rfind(" ", 0, split_at)
  if boundary <= 0:
    boundary = split_at

  prompt = text[:boundary].strip()
  gold = text[boundary:].strip()[:max_gold_chars].strip()
  return prompt, gold


def extract_entities(text, medication_lexicon):
  norm_text = re.sub(r"\s+", " ", text).strip()
  lower_text = norm_text.lower()
  entities = {
    "medications": Counter(),
    "doses": Counter(m.group(0).lower() for m in DOSE_RE.finditer(norm_text)),
    "frequencies": Counter(m.group(0).lower() for m in FREQ_RE.finditer(norm_text)),
    "negations": Counter(m.group(0).lower() for m in NEG_RE.finditer(norm_text)),
  }

  med_words = {word.lower() for word in medication_lexicon}
  for token in WORD_RE.findall(lower_text):
    if token in med_words:
      entities["medications"][token] += 1

  return entities


def _category_errors(gold_counter, pred_counter):
  overlap = gold_counter & pred_counter
  common = sum(overlap.values())
  gold_total = sum(gold_counter.values())
  pred_total = sum(pred_counter.values())
  missing = gold_total - common
  extra = pred_total - common
  substitutions = min(missing, extra)
  deletions = missing - substitutions
  insertions = extra - substitutions
  return substitutions, deletions, insertions


def score_entity_errors(gold_text, pred_text, medication_lexicon):
  gold = extract_entities(gold_text, medication_lexicon)
  pred = extract_entities(pred_text, medication_lexicon)

  substitutions = 0
  deletions = 0
  insertions = 0
  gold_entities = 0
  category_rates = {}
  categories = {}

  for key in ("medications", "doses", "frequencies", "negations"):
    g = gold[key]
    p = pred[key]
    category_gold = sum(g.values())
    gold_entities += category_gold
    s, d, i = _category_errors(g, p)
    substitutions += s
    deletions += d
    insertions += i
    category_total_errors = s + d + i
    category_rates[key] = (category_total_errors / category_gold) if category_gold else 0.0
    categories[key] = {
      "gold": category_gold,
      "substitutions": s,
      "deletions": d,
      "insertions": i,
      "error_rate": category_rates[key],
    }

  total_errors = substitutions + deletions + insertions
  ceer = (total_errors / gold_entities) if gold_entities else 0.0
  return {
    "gold_entities": gold_entities,
    "substitutions": substitutions,
    "deletions": deletions,
    "insertions": insertions,
    "ceer": ceer,
    "medication_error_rate": category_rates["medications"],
    "dose_error_rate": category_rates["doses"],
    "frequency_error_rate": category_rates["frequencies"],
    "negation_error_rate": category_rates["negations"],
    "categories": categories,
  }
