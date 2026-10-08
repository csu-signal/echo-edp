"""
plot_craft_metrics_by_model.py
-----------------------------
Plots CRAFT metrics across separate model-combination directories.

Expected directory structure:
ROOT_DIR/
    qwen-72b_gpt-4o-mini,,1773456132000/
        craft_structure_001_2.json
        craft_structure_002_2.json
        ...
    qwen-32b_gpt-4o-mini,,1773401844000/
        craft_structure_001_2.json
        ...
    ...

This script produces:

  Figure 1 — Raw metrics anchored at t=0
  Figure 2 — Delta metrics: value[t] - value[turn_1]

Curves are grouped by MODEL COMBINATION DIRECTORY, not by partialCompletionCategory.
Aggregation is over structures within each model directory, with SEM bands.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# ── Config ────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(

    "craft_gricean_simulations_open_weight_testing_20test_notools"
).parent
# "craft_gricean_simulations_open_weight_testing_20test_notools"  


RUN_FILTER = 3
N_TURNS = 20
SAVE_PREFIX = "craft_metrics_by_model"

METRIC_KEYS = [
    "overall_progress",
    "completion_percentage",
    "iou_score",
    "position_accuracy",
    "distance_score",
]

BINARY_KEYS = [
    "move_executed",
    "failed_move",
    "correct_structure_placement",
    "correct_side_placement",
]

ALL_KEYS = METRIC_KEYS + BINARY_KEYS
DELTA_KEYS = [f"{k}_delta" for k in METRIC_KEYS]
TOTAL_LEN = N_TURNS + 1
TURNS_AXIS = np.arange(0, N_TURNS + 1)

def recompute_all_flat(root_dir: Path, run_filter: int):
    """
    Same recomputation as recompute_all but returns flat structure:
        data[model_label][metric] = list of series   (no category dimension)
        data["ALL"][metric] = list of series          (pooled across models)
    """
    from collections import defaultdict
    import re

    def clean_model_label(dirname):
        name = dirname.split(",,")[0]
        parts = name.split("_")
        model_map = {
            "qwen-72b": "Qwen72B", "qwen-32b": "Qwen32B",
            "qwen-14b": "Qwen14B", "qwen-7b": "Qwen7B",
            "mistral-7b": "Mistral-7B", "llama-8b": "Llama-8B",
            "gemma-9b": "Gemma-9B", "deepseek-v2-lite": "DeepSeek-Lite",
        }
        builder_map = {"gpt-4o-mini": "4o-mini"}
        model   = model_map.get(parts[0], parts[0])
        builder = builder_map.get(parts[1], parts[1]) if len(parts) > 1 else ""
        return f"{model} + {builder}" if builder else model

    METRIC_KEYS = [
        "overall_progress", "completion_percentage",
        "iou_score", "position_accuracy", "distance_score",
    ]
    BINARY_KEYS = [
        "move_executed", "failed_move",
        "correct_structure_placement", "correct_side_placement",
    ]
    ALL_KEYS   = METRIC_KEYS + BINARY_KEYS
    DELTA_KEYS = [f"{k}_delta" for k in METRIC_KEYS]

    data = defaultdict(lambda: defaultdict(list))
    data["ALL"]  # touch it so it exists
    model_labels = []

    for model_dir in sorted(root_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_label = clean_model_label(model_dir.name)
        found = False

        for fpath in sorted(model_dir.glob("*.json")):
            m = re.match(r"craft_structure_(\d+)_(\d+)\.json", fpath.name)
            if not m or int(m.group(2)) != run_filter:
                continue

            with open(fpath) as f:
                d = json.load(f)

            if "games" not in d or not d["games"]:
                print(f"Warning: skipping {fpath.name} — no games")
                continue

            game     = d["games"][0]
            # recompute from snapshots — no dependency on logged metric values
            struct_vals = recompute_metrics_from_logs(game)

            for k in ALL_KEYS + DELTA_KEYS:
                data[model_label][k].append(struct_vals[k])
                data["ALL"][k].append(struct_vals[k])  # pool across models

            found = True

        if found and model_label not in model_labels:
            model_labels.append(model_label)
            print(f"Recomputed: {model_label:<28} ({len(data[model_label]['overall_progress'])} structures)")
             
    return dict(data), model_labels

                  
# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_model_label(dirname: str) -> str:
    name = dirname.split(",,")[0]

    parts = name.split("_")

    model = parts[0]
    builder = parts[1] if len(parts) > 1 else ""

    # shorten names
    model = model.replace("qwen-", "Qwen").replace("b", "B")
    model = model.replace("mistral-", "Mistral-")
    model = model.replace("gemma-", "Gemma-")
    model = model.replace("llama-", "Llama-")
    model = model.replace("deepseek-v2-lite", "DeepSeek-Lite")

    builder = builder.replace("gpt-", "").replace("mini", "-mini")

    return f"{model} + {builder}"


def get_metric_at_turn(turn, key):
    if key in METRIC_KEYS:
        pd = turn.get("progress_data", {})
        if isinstance(pd, dict) and "metrics" in pd:
            return pd["metrics"].get(key, None)
    elif key in BINARY_KEYS:
        v = turn.get(key, None)
        return float(v) if v is not None else None
    return None


def compute_stats(series_list, total_len=TOTAL_LEN):
    means = np.full(total_len, np.nan)
    sems = np.full(total_len, np.nan)

    for t in range(total_len):
        vals = [s[t] for s in series_list if s[t] is not None]
        if len(vals) >= 2:
            arr = np.array(vals, dtype=float)
            means[t] = np.mean(arr)
            sems[t] = np.std(arr, ddof=1) / np.sqrt(len(arr))
        elif len(vals) == 1:
            means[t] = vals[0]
            sems[t] = 0.0

    return means, sems


def build_struct_series(game):
    """
    Build one structure's metric time series.
    Index layout:
        0 = synthetic t=0
        1..N_TURNS = actual turns
    """
    turns = game["turns"]

    struct_vals = {}

    # Continuous metrics: synthetic t=0 baseline = 0.0
    for k in METRIC_KEYS:
        struct_vals[k] = [0.0] + [None] * N_TURNS

    # Binary metrics: no synthetic value at t=0
    for k in BINARY_KEYS:
        struct_vals[k] = [None] + [None] * N_TURNS

    for t in turns:
        tn = t.get("turn_number", None)
        if tn is None or tn < 1 or tn > N_TURNS:
            continue

        for k in ALL_KEYS:
            v = get_metric_at_turn(t, k)
            if v is not None:
                struct_vals[k][tn] = v

    # Delta metrics relative to turn 1
    for k in METRIC_KEYS:
        dk = f"{k}_delta"
        baseline = struct_vals[k][1]

        if baseline is not None:
            struct_vals[dk] = [None] + [
                (struct_vals[k][t] - baseline) if struct_vals[k][t] is not None else None
                for t in range(1, TOTAL_LEN)
            ]
            struct_vals[dk][1] = 0.0
        else:
            struct_vals[dk] = [None] * TOTAL_LEN

    return struct_vals


def discover_run_files_by_model(root_dir: Path, run_filter: int):
    """
    Returns:
        run_files_by_model: dict[model_label][struct_id] = json_path
        model_labels: list[str]
    """
    model_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])
    run_files_by_model = {}
    model_labels = []

    print(f"Scanning root dir: {root_dir}")
    print(f"Found {len(model_dirs)} model directories\n")

    for model_dir in model_dirs:
        model_label = clean_model_label(model_dir.name)
        files = sorted(model_dir.glob("*.json"))

        groups = defaultdict(dict)
        for f in files:
            m = re.match(r"craft_structure_(\d+)_(\d+)\.json", f.name)
            if m:
                struct_id, run = m.group(1), int(m.group(2))
                groups[struct_id][run] = f

        run_files = {
            sid: groups[sid][run_filter]
            for sid in groups
            if run_filter in groups[sid]
        }

        if len(run_files) == 0:
            print(f"Skipping {model_label:<28} : no run {run_filter} files found")
            continue

        run_files_by_model[model_label] = run_files
        model_labels.append(model_label)
        print(f"Loaded  {model_label:<28} : {len(run_files)} structures")

    print()
    return run_files_by_model, model_labels


def print_summary_tables(data, model_labels):
    print(f"\n{'=' * 90}")
    print("RAW METRICS — turn-by-turn means (ALL structures pooled across models)")
    print(f"{'=' * 90}")
    for k in ALL_KEYS:
        means, sems = compute_stats(data["ALL"][k])
        print(f"\n{k}")
        print(f"{'turn':>5}  {'mean':>8}  {'sem':>8}  {'n':>5}")
        print("-" * 35)
        for t in range(TOTAL_LEN):
            n_obs = sum(1 for s in data["ALL"][k] if s[t] is not None)
            m, se = means[t], sems[t]
            ms = f"{m:.4f}" if not np.isnan(m) else "   n/a"
            ss = f"{se:.4f}" if not np.isnan(se) else "   n/a"
            print(f"{t:>5}  {ms:>8}  {ss:>8}  {n_obs:>5}")

    print(f"\n{'=' * 90}")
    print("DELTA METRICS — gain from turn 1 baseline (ALL structures pooled across models)")
    print(f"{'=' * 90}")
    for dk in DELTA_KEYS:
        means, sems = compute_stats(data["ALL"][dk])
        print(f"\n{dk}")
        print(f"{'turn':>5}  {'mean':>8}  {'sem':>8}  {'n':>5}")
        print("-" * 35)
        for t in range(1, TOTAL_LEN):
            n_obs = sum(1 for s in data["ALL"][dk] if s[t] is not None)
            m, se = means[t], sems[t]
            ms = f"{m:.4f}" if not np.isnan(m) else "   n/a"
            ss = f"{se:.4f}" if not np.isnan(se) else "   n/a"
            print(f"{t:>5}  {ms:>8}  {ss:>8}  {n_obs:>5}")

    print(f"\n{'=' * 90}")
    print("PER-MODEL COUNTS")
    print(f"{'=' * 90}")
    for model in model_labels:
        n_structs = len(data[model]["overall_progress"])
        print(f"{model:<28} : {n_structs} structures")

def plot_metric_grid(
    data,
    model_labels,
    colors,
    keys,
    title,
    fname,
    x_start=0,
    hline_val=0.0,
    ylabel="Value",
):
    n_cols = 2 if len(keys) > 1 else 1
    n_rows = int(np.ceil(len(keys) / n_cols))

    model_labels = sorted(
        model_labels,
        key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0
    )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 7 * n_rows),
        constrained_layout=False,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # linestyles = ['-',  '--', '-.',  ':',  '-',  '--', '-.',  ':']
     
    linestyles = [
        (0, ()),           # solid
        (0, (5, 2)),       # dashed
        (0, (3, 1, 1, 1)), # dashdot tight
        (0, (1, 1)),       # dotted tight
        (0, (5, 1)),       # dashed tight
        (0, (3, 2, 1, 2)), # dashdot loose
        (0, (1, 2)),       # dotted loose
        (0, (5, 5)),       # dashed loose
    ]
    markers    = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    markersize = 5

    for ax_idx, key in enumerate(keys):
        ax = axes_flat[ax_idx]

        for idx, model in enumerate(model_labels):   # ← idx tracked here
            series_list = data[model].get(key, [])   # ← series_list defined here
            if len(series_list) == 0:
                continue

            means, sems = compute_stats(series_list)
            mask = (TURNS_AXIS >= x_start) & ~np.isnan(means)
            x = TURNS_AXIS[mask]
            y = means[mask]
            e = sems[mask]

            ax.plot(
                x, y,
                label=f"{model} (n={len(series_list)})",
                color=colors[model],
                linestyle=linestyles[idx % len(linestyles)],
                marker=markers[idx % len(markers)],
                linewidth=2,
                markersize=3,
                zorder=3,
            )
            # ax.fill_between(
            #     x,
            #     y - e,
            #     y + e,
            #     color=colors[model],
            #     alpha=0.15,
            #     zorder=2,
            # )

            #error bars at specific turns
            ax.errorbar(x[::2], y[::2], yerr=e[::2], fmt='none',
            color=colors[model], capsize=2, linewidth=0.5, alpha=0.2)

            #95 percent CI 
            # final_idx = np.where(mask)[0][-1]
            # ax.errorbar(x[-1], y[-1], yerr=1.96*e[-1],
            # fmt='none', color=colors[model], capsize=3, linewidth=1.2)

        display = key.replace("_delta", "").replace("_", " ").title()
        if key.endswith("_delta"):
            display += " (Δ from turn 1)"

        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.set_xlabel("Turn", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlim(x_start - 0.3, N_TURNS + 0.3)
        ax.set_xticks(range(x_start, N_TURNS + 1, 2))
        ax.axhline(
            hline_val,
            color="gray",
            linewidth=0.8,
            linestyle="--",
            alpha=0.6,
            zorder=1,
        )
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax_idx in range(len(keys), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(bottom=0.22, top=0.88, hspace=0.30, wspace=0.20)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, -0.02),
            fontsize=10,
        )

    plt.show()
    plt.close(fig)
# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    run_files_by_model, model_labels = discover_run_files_by_model(ROOT_DIR, RUN_FILTER)

    if not model_labels:
        raise ValueError(
            f"No model directories with run {RUN_FILTER} files found under {ROOT_DIR}"
        )
    data, model_labels = recompute_all_flat(ROOT_DIR, RUN_FILTER)  # in some turns the dynamic progress tracker fails; but we can recompute the progress metrics entirely from the structure snapshot vs target

    print_summary_tables(data, model_labels)

    if len(model_labels) > 1:
        cmap = cm.get_cmap("tab10", len(model_labels))
        colors = {model: cmap(i) for i, model in enumerate(model_labels)}
    else:
        colors = {model_labels[0]: "#2563EB"}

    plot_metric_grid(
        data=data,
        model_labels=model_labels,
        colors=colors,
        keys=ALL_KEYS,
        title=(
            f"CRAFT Raw Metrics by Model Combination — Turn by Turn "
            f"(Run {RUN_FILTER}, t=0 anchored, SEM over structures)"
        ),
        fname=f"{SAVE_PREFIX}_raw.png",
        x_start=0,
        hline_val=0.0,
        ylabel="Value",
    )

    plot_metric_grid(
        data=data,
        model_labels=model_labels,
        colors=colors,
        keys=DELTA_KEYS,
        title=(
            f"CRAFT Metric Deltas by Model Combination — Gain from Turn 1 Baseline "
            f"(Run {RUN_FILTER}, turn 1 = 0, SEM over structures)"
        ),
        fname=f"{SAVE_PREFIX}_delta.png",
        x_start=1,
        hline_val=0.0,
        ylabel="Δ Value (from turn 1)",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()