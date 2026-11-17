"""
Convert eval.py per-target results into the DatasetDict layout read by
recompute_csg_main_metrics_from_hf.py.

Usage:
    python export_hf_dataset.py --results-dir eval_results/per_target --out ./csg_eval_dataset
    python recompute_csg_main_metrics_from_hf.py --dataset ./csg_eval_dataset --output-dir ./csg_main_metrics
"""

import json
import pickle
import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict

SECRETS_PATH = Path("eval_results/canonical_secrets.json")
STRATA       = ["favorable", "hard_in_dist", "ood_range", "structural"]
TURN_FIELDS  = ["candidates_before", "candidates_after", "eliminated", "redundant",
                "hint", "raw_response", "completion", "arm", "question"]


def stratum_of(secret, canonical):
    return next((s for s in STRATA if secret in canonical[s]), None)


def build(results_dir, targets=None):
    with open(SECRETS_PATH) as f:
        canonical = json.load(f)

    turns, episodes = [], []
    for p in sorted(Path(results_dir).glob("*.json")):
        with open(p) as f:
            meta = json.load(f)
        if targets and meta["target_id"] not in targets:
            continue
        # episodes_pkl was recorded relative to wherever eval.py ran, so resolve it next to the json
        with open(p.with_suffix(".pkl"), "rb") as f:
            logs = pickle.load(f)
        if len(logs) != meta["n_episodes"]:
            raise ValueError(f"{p.name}: {len(logs)} episode logs but n_episodes={meta['n_episodes']}")

        model_key = meta["target_id"]
        run_id    = f"{model_key}__run{meta['run_idx']}"
        for ep_idx, log in enumerate(logs):
            secret     = log[0]["secret"]
            episode_id = f"{run_id}__ep{ep_idx}"
            resolved   = any(t["resolved"] for t in log)
            episodes.append({
                "model_key": model_key, "group": meta["category"], "run_id": run_id,
                "run_index": meta["run_idx"], "episode_id": episode_id, "episode_index": ep_idx,
                "secret": secret, "stratum": stratum_of(secret, canonical),
                "resolved": resolved, "turns_taken": len(log),
            })
            for t in log:
                turns.append({
                    "model_key": model_key, "group": meta["category"], "run_id": run_id,
                    "run_index": meta["run_idx"], "episode_id": episode_id, "episode_index": ep_idx,
                    "turn_index": t["turn"], "secret": secret, "episode_resolved": resolved,
                    **{k: t[k] for k in TURN_FIELDS},
                })

    if not turns:
        raise ValueError(f"no results found in {results_dir}")
    return DatasetDict({"turns": Dataset.from_list(turns), "episodes": Dataset.from_list(episodes)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export eval.py results as a CSG HF DatasetDict")
    parser.add_argument("--results-dir", default="eval_results/per_target")
    parser.add_argument("--out",         required=True, help="load_from_disk directory to write")
    parser.add_argument("--targets",     nargs="+", default=None,
                        help="target_ids to include (default: all), e.g. base__llama-3.1-8b-instruct")
    args = parser.parse_args()

    ds = build(args.results_dir, set(args.targets) if args.targets else None)
    ds.save_to_disk(args.out)
    models = sorted(set(ds["episodes"]["model_key"]))
    print(f"saved -> {args.out}: {len(ds['turns'])} turns, {len(ds['episodes'])} episodes, models={models}")
