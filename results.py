"""
Compute per-stratum metrics from the HuggingFace episodes parquet.

Run this on a machine with pyarrow/pandas available (e.g. Shannon).

Output:
    Printed tables (overall + 4 strata, 6 metrics each)
    stratum_results_from_hf.json

Usage:
    python results.py
    # run from the release/ directory, or set HF_DIR / SECRETS_PATH below
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import pyarrow.parquet as pq

try:
    _BASE = Path(__file__).parent
except NameError:
    _BASE = Path.cwd()   # notebook fallback

HF_DIR       = _BASE / "hf_public_csg_dataset"
SECRETS_PATH = _BASE / "eval_results" / "canonical_secrets.json"

# ── load stratum membership ───────────────────────────────────────────────────

with open(SECRETS_PATH) as f:
    canonical = json.load(f)

STRATA      = ["favorable", "hard_in_dist", "ood_range", "structural"]
stratum_sets= {s: set(canonical[s]) for s in STRATA}

# ── load episodes parquet ─────────────────────────────────────────────────────

ep_table = pq.read_table(HF_DIR / "episodes.parquet")
episodes  = ep_table.to_pydict()
n         = len(episodes["secret"])
print(f"Loaded {n:,} episode rows")
print(f"Columns: {list(episodes.keys())}\n")

model_keys = sorted(set(episodes["model_key"]))
print(f"Model keys ({len(model_keys)}):")
for k in model_keys:
    print(f"  {k!r}")
print()

by_model = defaultdict(list)
for i in range(n):
    by_model[episodes["model_key"][i]].append(i)

# ── metrics ───────────────────────────────────────────────────────────────────

EPISODE_METRICS = [
    ("resolved",              "Resolve%",  True),
    ("zero_rate",             "Zero%",     True),
    ("quality_mean",          "Qual",      False),
    ("grounded_rate",         "Ground%",   True),
    ("grounded_recover_rate", "GRecover%", True),
    ("reason_pct",            "Reason%",   True),
]

def mean(v):
    return sum(v) / len(v) if v else float("nan")

def stratum_mean(indices, col, stratum):
    vals = [
        float(episodes[col][i])
        for i in indices
        if episodes["secret"][i] in stratum_sets[stratum]
        and episodes[col][i] is not None
    ]
    return mean(vals) if vals else None

def overall_mean(indices, col):
    vals = [float(episodes[col][i]) for i in indices if episodes[col][i] is not None]
    return mean(vals) if vals else None

# ── model groups ──────────────────────────────────────────────────────────────

GROUPS = [
    ("ECHO",                    ["multiturn__cross_generation"]),
    ("Episode-return GRPO",     ["multiturn__episode_return"]),
    ("RLOO per-turn",           ["multiturn__rloo_per_turn"]),
    ("Offline SFT+RL",          ["singleturn__offline_continued"]),
    ("Claude Sonnet 4.6",       ["frontier__claude-sonnet-4-6"]),
    ("o3-mini",                 ["frontier__o3-mini"]),
    ("GPT-4o",                  ["frontier__gpt-4o"]),
    ("GPT-4.1-mini",            ["frontier__gpt-4.1-mini"]),
    ("GPT-4o-mini",             ["frontier__gpt-4o-mini"]),
    ("GPT-4.1-nano",            ["frontier__gpt-4.1-nano"]),
    ("Qwen2.5-1.5B base",       ["base__qwen2.5-1.5b-instruct"]),
    ("Qwen2.5-1.5B base (CoT)", ["base__qwen2.5-1.5b-instruct__cot"]),
    ("Qwen2.5-1.5B base (ReAct)",["base__qwen2.5-1.5b-instruct__react_proper"]),
    ("Qwen2.5-3B base",         ["base__qwen2.5-3b-instruct"]),
    ("Qwen2.5-7B base",         ["base__qwen2.5-7b-instruct"]),
    ("Qwen2.5-14B base",        ["base__qwen2.5-14b-instruct"]),
    ("Greedy oracle",           ["python__greedy_oracle"]),
]

# ── print tables ──────────────────────────────────────────────────────────────

def fmt(v, pct=True):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v*100:.1f}" if pct else f"{v:.3f}"

results = {}

for stratum_label, stratum_filter in [
    ("ALL SECRETS (overall)", None),
    ("FAVORABLE",             "favorable"),
    ("HARD IN-DIST",          "hard_in_dist"),
    ("OOD RANGE (101-200)",   "ood_range"),
    ("STRUCTURAL",            "structural"),
]:
    print(f"\n=== {stratum_label} ===\n")

    col_headers = "  ".join(f"{lbl:>10}" for _, lbl, _ in EPISODE_METRICS)
    header      = f"{'Model':<40}  {col_headers}  {'n_eps':>6}"
    print(header)
    print("-" * len(header))

    for label, key_patterns in GROUPS:
        indices      = []
        matched_keys = []
        for mk in model_keys:
            if mk in key_patterns:
                indices.extend(by_model[mk])
                matched_keys.append(mk)

        if not indices:
            print(f"  {label:<38}  (no match)")
            continue

        row      = f"  {label:<38}"
        row_data = {}
        for col, lbl, pct in EPISODE_METRICS:
            v    = overall_mean(indices, col) if stratum_filter is None else stratum_mean(indices, col, stratum_filter)
            row += f"  {fmt(v, pct):>10}"
            row_data[lbl] = v

        n_eps = len(indices) if stratum_filter is None else sum(
            1 for i in indices if episodes["secret"][i] in stratum_sets[stratum_filter]
        )
        row += f"  {n_eps:>6}"
        print(row)

        results.setdefault(label, {})
        results[label][stratum_label] = row_data
        results[label].setdefault("matched_keys", matched_keys)

# ── save ──────────────────────────────────────────────────────────────────────

out_path = _BASE / "stratum_results_from_hf.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}")
