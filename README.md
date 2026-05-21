# MG-SD / EARS Benchmark Notes

Date: 2026-05-21

## 1. Service startup commands

### Baseline (Standard SD)

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
unset VLLM_EARS_BASE_TOLERANCE VLLM_MGSD_ENABLED VLLM_MGSD_MARGIN_DELTA

vllm serve /data/models/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --speculative-config '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}'
```

### EARS

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_EARS_BASE_TOLERANCE=0.1
unset VLLM_MGSD_ENABLED VLLM_MGSD_MARGIN_DELTA

vllm serve /data/models/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --speculative-config '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}'
```

### MG-SD

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_EARS_BASE_TOLERANCE=0.1
export VLLM_MGSD_ENABLED=1
export VLLM_MGSD_MARGIN_DELTA=0.1

vllm serve /data/models/Qwen3-32B \
  --host 127.0.0.1 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --tensor-parallel-size 2 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code \
  --speculative-config '{"model":"/data/models/Qwen3-0.6B","method":"draft_model","num_speculative_tokens":5,"parallel_drafting":false}'
```

## 2. Dataset and benchmark method

- Dataset file: `/home/scd/MG-SD/mimic_evalscope_messages.jsonl`
- Source: `/home/scd/mimic-iv-note/note/discharge.csv.gz`
- Prompt construction: take the note prefix, ask the model to continue the de-identified clinical note in a consistent style.

### Temperature 0 smoke benchmark

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --tokenizer-path /data/models/Qwen3-32B \
  --dataset line_by_line \
  --dataset-path /home/scd/MG-SD/mimic_evalscope_messages.jsonl \
  --number 20 \
  --parallel 1 \
  --max-tokens 256 \
  --temperature 0.0 \
  --stream
```

### Temperature 0.9 throughput benchmark

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --tokenizer-path /data/models/Qwen3-32B \
  --dataset line_by_line \
  --dataset-path /home/scd/MG-SD/mimic_evalscope_messages.jsonl \
  --number 10 \
  --parallel 1 \
  --max-tokens 512 \
  --temperature 0.9 \
  --seed 123 \
  --stream
```

## 3. Temperature 0 results

### Summary table

| Method | Output Throughput | Total Throughput | TPOT | Avg Latency | Accept Rate | Decoded Tok/Iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 36.07 tok/s | 70.60 tok/s | 27.28 ms | 7.09 s | 0.59 | 2.44 |
| EARS | 36.30 tok/s | 71.04 tok/s | 27.15 ms | 7.05 s | 0.59 | 2.44 |
| MG-SD (`δ=0.1`) | 35.74 tok/s | 69.95 tok/s | 27.57 ms | 7.16 s | 0.59 | 2.44 |

### Interpretation

1. This setup is too deterministic to separate EARS and MG-SD clearly.
2. Baseline, EARS, and MG-SD have nearly identical accept rate and decoded tokens per iteration, which means the tolerance path is barely being exercised.
3. The saved sample outputs were also nearly identical. This is a sanity-check result, not evidence for the medical safety claim.
4. Conclusion: `temperature=0` is useful for smoke validation, but not strong enough for proving the margin gate benefit.

## 4. Temperature 0.9 results

### Summary table

| Method | Output Throughput | Total Throughput | TPOT | Avg Latency | Accept Rate | Decoded Tok/Iter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 30.53 tok/s | 45.65 tok/s | 32.53 ms | 16.76 s | 0.54 | 2.18 |
| EARS | 33.13 tok/s | 49.53 tok/s | 29.97 ms | 15.45 s | 0.56 | 2.29 |
| MG-SD (`δ=0.05`) | 32.78 tok/s | 49.01 tok/s | 30.28 ms | 15.61 s | 0.57 | 2.30 |
| MG-SD (`δ=0.10`) | 31.87 tok/s | 47.65 tok/s | 31.14 ms | 16.05 s | 0.56 | 2.28 |
| MG-SD (`δ=0.20`) | 32.24 tok/s | 48.20 tok/s | 30.80 ms | 15.88 s | 0.56 | 2.26 |

### Delta vs baseline

| Method | Output Throughput | Total Throughput | TPOT | Avg Latency |
| --- | ---: | ---: | ---: | ---: |
| EARS | +8.5% | +8.5% | -7.9% | -7.8% |
| MG-SD (`δ=0.05`) | +7.4% | +7.4% | -6.9% | -6.9% |
| MG-SD (`δ=0.10`) | +4.4% | +4.4% | -4.3% | -4.2% |
| MG-SD (`δ=0.20`) | +5.6% | +5.6% | -5.3% | -5.3% |

### Delta vs EARS

| Method | Output Throughput | Total Throughput | TPOT | Avg Latency |
| --- | ---: | ---: | ---: | ---: |
| MG-SD (`δ=0.05`) | -1.1% | -1.1% | +1.0% | +1.0% |
| MG-SD (`δ=0.10`) | -3.8% | -3.8% | +3.9% | +3.9% |
| MG-SD (`δ=0.20`) | -2.7% | -2.7% | +2.8% | +2.8% |

### Interpretation

1. At `temperature=0.9`, both EARS and MG-SD clearly outperform baseline in throughput and TPOT.
2. MG-SD still preserves throughput gain relative to baseline for all tested `δ` values.
3. `δ=0.05` is the best speed point among the tested MG-SD settings and is very close to EARS.
4. `δ=0.10` is the more conservative default for a medical-safety pilot: it still beats baseline, but leaves more room for the margin gate to block risky low-margin positions.
5. `δ=0.20` is stricter than `0.10`, but in this small run it is not clearly better than `0.10` in throughput or latency. Without entity-error evidence, it is hard to justify as the default.

### Practical recommendation

- **Speed-oriented default:** `VLLM_MGSD_MARGIN_DELTA=0.05`
- **Safety-oriented pilot default:** `VLLM_MGSD_MARGIN_DELTA=0.10`
- **Not recommended as first default:** `VLLM_MGSD_MARGIN_DELTA=0.20`

## 5. How to prove MG-SD reduces entity errors

The paper claim is not "MG-SD is faster". The real claim is:

> MG-SD reduces medical entity errors compared with confidence-only relaxation (EARS), while preserving most of the speculative decoding speed gain.

To prove that, the experiment must include both **speed metrics** and **entity-error metrics**.

### 5.1 Main comparison

Compare these methods on the same prompt set:

1. Standard SD (baseline)
2. EARS / confidence-relaxed SD
3. MG-SD (`δ=0.10` as main setting, `δ=0.05` as speed-oriented ablation)

### 5.2 Generation setup

- Data: MIMIC-IV discharge note continuation
- Prompt: first 30%-50% of each note
- Gold reference: remaining suffix
- Output length: 256-512 tokens
- Temperature: `0.9`
- Use fixed seeds for repeatability in the pilot, then repeat with 3 seeds in the main experiment

### 5.3 Entity categories

Track at least:

1. drug names
2. dosage / number / unit / frequency
3. negation markers
4. diagnosis / procedure / body part / time expressions

### 5.4 Metrics

- **CEER**: clinical entity error rate
- **Med-CEER**: medical entity error rate
- **Dose/Number Error**
- **Negation Error**
- **Answer Flip Rate** on MedQA

### 5.5 What counts as evidence

The core result should show:

1. EARS improves speed over baseline
2. EARS increases entity errors or at least fails to control them
3. MG-SD lowers entity errors relative to EARS
4. MG-SD still keeps a clear throughput advantage over baseline

### 5.6 Mechanism evidence

This is the key part for the paper story. In addition to output evaluation, log:

- `p1`
- `p2`
- `margin = p1 - p2`
- whether the token is accepted via relaxation

Then analyze:

1. low-margin ratio for entity tokens vs non-entity tokens
2. low-margin ratio for error tokens vs correct tokens
3. whether MG-SD suppresses acceptance exactly in those risky low-margin regions

If erroneous entity positions are significantly more likely to be low-margin than ordinary positions, then the margin-gate hypothesis is supported.

### 5.7 Suggested pilot design

- 300-500 MIMIC samples
- methods: baseline / EARS / MG-SD (`δ=0.10`)
- optional ablation: MG-SD (`δ=0.05`)
- outputs: speed table + CEER table + 5-10 case studies + margin histogram

### 5.8 If the effect is still too small

If MG-SD does not separate clearly from EARS, increase task difficulty:

1. use smaller draft models
2. use longer generations
3. keep `temperature=0.9`
4. focus on medication-dense or number-dense note subsets

## 6. Result directories

- Temperature 0: `/home/scd/MG-SD/evalscope/`
- Temperature 0.9 sweep: `/home/scd/MG-SD/temp09/evalscope/`
- Logs: `/home/scd/MG-SD/logs/` and `/home/scd/MG-SD/temp09/logs/`
- Entity-eval pilot: `/home/scd/MG-SD/entity_eval/`

## 7. Medication-section entity-error pilot

### Pilot method

- Data: 12 MIMIC-IV discharge notes
- Slice strategy: align the prompt/gold boundary near `Discharge Medications:` or nearby medication-heavy sections
- Generation: `temperature=0.9`, `max_tokens=256`
- Compared methods: baseline / EARS / MG-SD (`δ=0.10`)
- Scoring: regex-based CEER-style proxy over four categories:
  1. medications
  2. doses
  3. frequencies
  4. negations

### Reproduction command

```bash
cd /home/scd/MG-SD
python3 run_entity_eval_pilot.py --sample-count 12 --max-tokens 256 --temperature 0.9
```

### Pilot results

| Method | Samples | Gold Entities | CEER | Med Error | Dose Error | Freq Error | Negation Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 12 | 300 | 0.773 | 0.943 | 0.859 | 0.639 | 1.125 |
| EARS | 12 | 285 | 0.698 | 0.755 | 0.852 | 0.559 | 1.000 |
| MG-SD (`δ=0.10`) | 12 | 275 | 0.647 | 0.654 | 0.795 | 0.523 | 1.125 |

### Key takeaways

1. MG-SD beats EARS on the aggregate CEER-style proxy: `0.698 -> 0.647` (**-7.3%**).
2. The strongest gain is on medication entities: `0.755 -> 0.654` (**-13.4%** vs EARS).
3. Dose and frequency errors also improve: dose `-6.7%`, frequency `-6.5%` vs EARS.
4. Negation does **not** improve in this small pilot.
5. Relative to baseline, both EARS and MG-SD reduce this proxy metric, but the paper claim should still focus on **MG-SD vs EARS**.

### Stability note

- Across 12 notes, MG-SD is better than EARS on 4 notes, worse on 3 notes, and tied on 5 notes.
- This is enough for a **positive pilot signal**, but not enough yet for a publication-grade claim.

### Representative case

For note `10000032-DS-21`, EARS drifts into many extra medications and doses, while MG-SD stays much closer to the gold medication list:

- baseline CEER: `0.476`
- EARS CEER: `1.190`
- MG-SD CEER: `0.476`

In this case, MG-SD removes the large medication/dose insertion burst that appears under EARS.

### Important caveat

This pilot uses a **regex-based CEER proxy**, not a manually annotated clinical entity benchmark. So the current result should be presented as:

> MG-SD shows an encouraging pilot trend of reducing medication-section entity errors relative to EARS.

It should **not** yet be overstated as the final proof.

### What would make the claim strong

To make the paper claim convincing, the next experiment should:

1. expand to 300-500 medication-section samples
2. keep `temperature=0.9`
3. fix `δ=0.10` as the main setting and `δ=0.05` as ablation
4. add per-token `p1 / p2 / margin` logging
5. show that the positions corrected by MG-SD are enriched in low-margin entity tokens