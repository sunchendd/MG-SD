import re


def normalize_medication_text(text):
  lowered = text.lower()
  lowered = re.sub(r"[()/,-]+", " ", lowered)
  lowered = re.sub(r"\s+", " ", lowered).strip()
  return lowered


def build_medication_alias_set(base_terms):
  aliases = set()
  for term in base_terms:
    aliases.add(normalize_medication_text(term))
  return aliases
