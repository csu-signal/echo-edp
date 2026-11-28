"""
Plots and tables for CSG eval results (eval_results*/per_target), with the
paper's metric definitions from recompute_csg_main_metrics_from_hf.py.

Usage:
    python analyze_csg_evals.py --results eval_results --out analysis_out_csg
    python analyze_csg_evals.py --results eval_results=gpt-4o-mini \
        eval_results_oracle_claude-haiku-5-5=haiku --out analysis_out_csg_oracles

Each --results entry is a results dir, optionally `dir=tag`; tagged conditions
are labeled `<target_id> (<tag>)` so the same model under different oracles
stays separate. Error bars are SEM across eval passes (run_idx).
"""
import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

import recompute_csg_main_metrics_from_hf as rc

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
NEUTRAL = "#8a8984"
OURS_EDGE = dict(edgecolor="black", linewidth=1.3)
BAR_W = 0.78
PANEL_SIZE = (4.6, 3.4)


@dataclass(frozen=True)
class Metric:
    key: str
    display: str
    percent: bool
    higher_is_better: bool
    decimals: int


METRICS = [
    Metric("Resolve", "Resolve", True, True, 1),
    Metric("Zero", "Zero-elimination", True, False, 1),
    Metric("Qual", "Quality", False, True, 3),
    Metric("Ground", "Groundedness", False, True, 3),
    Metric("GRecover", "Grounded recovery", False, True, 3),
    Metric("ResAfterZero", "Resolve after zero", False, True, 3),
    Metric("LateCalib", "Late redundancy", True, False, 1),
    Metric("Rec1", "Recovery (next turn)", False, True, 3),
    Metric("RecQ", "Recovery quality", False, True, 3),
    Metric("NextZero", "Repeat zero", False, False, 3),
    Metric("TTRsucc", "Eventual recovery", False, True, 3),
    Metric("ReasonPct", "Reasoning text", True, None, 1),
]
METRICS_BY_KEY = {m.key: m for m in METRICS}
DEFAULT_METRICS = ["Resolve", "Zero", "Qual", "Ground", "GRecover", "ResAfterZero"]


def use_paper_style():
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 11, "axes.edgecolor": "#222222", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
        "axes.axisbelow": True, "grid.color": "#dddddd", "grid.linewidth": 0.6, "grid.linestyle": ":",
        "legend.frameon": False, "legend.fontsize": 9, "figure.facecolor": "#ffffff",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })


def save(fig, out_dir, name, formats):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"{name}.{fmt}"
        fig.savefig(path)
        print(f"  wrote {path}")
    plt.close(fig)


def load_turns(results_specs, labels=None):
    """Flatten every per-target episode pickle into one raw turn table."""
    rows = []
    for spec in results_specs:
        path, _, tag = spec.partition("=")
        for jpath in sorted(Path(path, "per_target").glob("*.json")):
            meta = json.loads(jpath.read_text())
            label = f"{meta['target_id']} ({tag})" if tag else meta["target_id"]
            if labels and label not in labels and meta["target_id"] not in labels:
                continue
            with open(jpath.with_suffix(".pkl"), "rb") as f:
                episodes = pickle.load(f)
            run_id = f"{label}::run{meta['run_idx']}"
            for ep_idx, episode in enumerate(episodes):
                resolved = any(bool(t.get("resolved")) for t in episode)
                for t in episode:
                    rows.append({
                        "model_key": label, "model": label, "group": meta.get("category"),
                        "run_id": run_id, "run_index": meta["run_idx"],
                        "episode_id": f"{run_id}::ep{ep_idx}", "episode_index": ep_idx,
                        "turn": t["turn"], "secret": t["secret"],
                        "candidates_before": t["candidates_before"], "candidates_after": t["candidates_after"],
                        "eliminated": t["eliminated"], "redundant": t.get("redundant"), "hint": t.get("hint"),
                        "raw_response": t.get("raw_response"), "completion": t.get("completion"),
                        "episode_resolved": resolved,
                    })
    if not rows:
        raise SystemExit(f"no per_target results found in {results_specs}")
    return pd.DataFrame(rows)


def compute_metrics(raw):
    turns = rc.add_turn_derivatives(rc.normalize_turns(raw))
    episodes = rc.build_episode_table(turns, rc.normalize_episodes(None))
    events = rc.build_recovery_events(turns)
    return turns, episodes, rc.build_run_metrics(turns, episodes, events)


def assign_colors(labels, ours):
    others = [l for l in labels if l not in ours]
    colors = {l: SERIES[i % len(SERIES)] for i, l in enumerate(ours)}
    colors.update({l: (SERIES[(len(ours) + i) % len(SERIES)] if len(labels) <= len(SERIES) else NEUTRAL)
                   for i, l in enumerate(others)})
    return colors


def fmt(metric, v):
    return f"{v * 100:.{metric.decimals}f}" if metric.percent else f"{v:.{metric.decimals}f}"


def axis_label(metric):
    arrow = {True: " ↑", False: " ↓", None: ""}[metric.higher_is_better]
    return f"{metric.display}{' (%)' if metric.percent else ''}{arrow}"


def summary(run_metrics, metric):
    g = run_metrics.groupby("model_key")[metric.key]
    return pd.DataFrame({"mean": g.mean(), "sem": g.sem(), "n_runs": g.count()})


def metric_bar(run_metrics, metric, colors, ours, display, panel=False):
    s = summary(run_metrics, metric).dropna(subset=["mean"]).sort_values("mean", ascending=False)
    if s.empty:
        return None
    scale = 100 if metric.percent else 1
    fig, ax = plt.subplots(figsize=PANEL_SIZE if panel else (max(3.6, 0.42 * len(s) + 1.4), 3.4))
    xs = np.arange(len(s))
    means, sems = s["mean"].to_numpy() * scale, s["sem"].fillna(0).to_numpy() * scale
    for x, label, m in zip(xs, s.index, means):
        ax.bar(x, m, width=BAR_W, color=colors[label], zorder=2, **(OURS_EDGE if label in ours else {}))
    ax.errorbar(xs, means, yerr=sems, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=3)
    pad = 0.02 * (max(means + sems) - min(0, min(means - sems)) or 1)
    for x, m, e, v in zip(xs, means, sems, s["mean"]):
        ax.text(x, m + e + pad, fmt(metric, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([display(l) for l in s.index], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(axis_label(metric))
    ax.set_ylim(top=ax.get_ylim()[1] * 1.08)
    return fig


def candidate_curve(turns, episodes, colors, ours, display, panel=False):
    """Mean surviving candidates after each turn; finished episodes carry their final set forward."""
    t = turns.sort_values(["run_id", "episode_id", "turn_position"])
    per_run = {}
    for (label, run_id), g in t.groupby(["model_key", "run_id"]):
        curves = []
        for _, ep in g.groupby("episode_id"):
            after = ep["candidates_after"].to_numpy(dtype=float)
            curves.append(np.concatenate([after, np.full(rc.MAX_TURNS - len(after), after[-1])]))
        per_run.setdefault(label, []).append(np.mean(curves, axis=0))
    fig, ax = plt.subplots(figsize=PANEL_SIZE if panel else (5.2, 3.6))
    xs = np.arange(1, rc.MAX_TURNS + 1)
    for label in sorted(per_run, key=lambda l: (l not in ours, l)):
        arr = np.array(per_run[label])
        mean = arr.mean(axis=0)
        ax.plot(xs, mean, color=colors[label], lw=2.2 if label in ours else 1.5, label=display(label))
        if len(arr) > 1:
            err = arr.std(axis=0, ddof=1) / np.sqrt(len(arr))
            ax.fill_between(xs, mean - err, mean + err, color=colors[label], alpha=0.18, lw=0)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Turn")
    ax.set_ylabel("Surviving candidates ↓")
    ax.set_xticks(xs)
    if not panel:
        ax.legend(fontsize=7, loc="upper right")
    return fig


def paired_resolve_test(episodes, a, b, n_perm=10000, seed=0):
    """Sign-flip permutation test on per-secret resolve rates (averaged over passes)."""
    per = episodes.groupby(["model_key", "secret"])["Resolve_episode"].mean().unstack(0)
    if a not in per or b not in per:
        return np.nan, np.nan, 0
    d = (per[a] - per[b]).dropna().to_numpy()
    if len(d) == 0:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice([-1, 1], size=(n_perm, len(d))) * d).mean(axis=1))
    return d.mean(), (np.sum(null >= abs(d.mean())) + 1) / (n_perm + 1), len(d)


def markdown_table(run_metrics, metrics, display):
    labels = list(summary(run_metrics, METRICS_BY_KEY["Resolve"]).sort_values("mean", ascending=False).index)
    head = "| Condition | n | " + " | ".join(axis_label(m) for m in metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 2)]
    stats = {m.key: summary(run_metrics, m) for m in metrics}
    for l in labels:
        cells = []
        for m in metrics:
            r = stats[m.key].loc[l]
            cells.append("–" if np.isnan(r["mean"]) else
                         fmt(m, r["mean"]) + ("" if np.isnan(r["sem"]) else f" ± {fmt(m, r['sem'])}"))
        lines.append(f"| {display(l)} | {int(stats[metrics[0].key].loc[l, 'n_runs'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", nargs="+", default=["eval_results"], help="results dirs, each optionally dir=tag")
    parser.add_argument("--out", default="analysis_out_csg")
    parser.add_argument("--labels", nargs="*", default=None, help="only these target_ids / tagged labels")
    parser.add_argument("--ours", nargs="*", default=None,
                        help="labels drawn with a black outline (default: echo__ labels whose name contains 'echo')")
    parser.add_argument("--reference", default=None, help="label each condition's Resolve is tested against")
    parser.add_argument("--rename", nargs="*", default=[], help="display names as label=Name")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS, choices=list(METRICS_BY_KEY))
    parser.add_argument("--figure_panels", nargs="*", default=["Resolve", "Zero", "Ground", "candidates"],
                        help="panel_*.png figures: metric keys and/or candidates")
    parser.add_argument("--formats", nargs="*", default=["png"])
    args = parser.parse_args()

    use_paper_style()
    raw = load_turns(args.results, args.labels)
    turns, episodes, run_metrics = compute_metrics(raw)
    secrets = raw.groupby("episode_id")["secret"].first()
    episodes["secret"] = episodes["episode_id"].map(secrets)

    labels = sorted(run_metrics["model_key"].unique())
    ours = args.ours if args.ours is not None else [
        l for l in labels if l.startswith("echo__") and re.search(r"echo", l[6:], re.I)
        and not re.search(r"grpo|ppo|rtg|igpo", l, re.I)]
    names = dict(r.split("=", 1) for r in args.rename)
    display = lambda l: names.get(l, l.removeprefix("echo__").removeprefix("base__").removeprefix("frontier__"))
    colors = assign_colors(labels, ours)
    out = Path(args.out)

    print(f"[csg-analysis] {len(episodes)} episodes, {len(labels)} conditions: "
          + ", ".join(f"{l} (runs={run_metrics[run_metrics.model_key == l].shape[0]})" for l in labels))

    metrics = [METRICS_BY_KEY[k] for k in args.metrics]
    for m in metrics:
        fig = metric_bar(run_metrics, m, colors, ours, display)
        if fig is not None:
            save(fig, out, f"bar_{m.key}", args.formats)
    save(candidate_curve(turns, episodes, colors, ours, display), out, "curve_candidates", args.formats)

    for key in args.figure_panels:
        fig = (candidate_curve(turns, episodes, colors, ours, display, panel=True) if key == "candidates"
               else metric_bar(run_metrics, METRICS_BY_KEY[key], colors, ours, display, panel=True))
        if fig is not None:
            save(fig, out, f"panel_{key}", ["png"])

    fig = plt.figure(figsize=(6.5, 0.4 + 0.25 * ((len(labels) + 3) // 4)))
    fig.legend(handles=[Patch(facecolor=colors[l], label=display(l), **(OURS_EDGE if l in ours else {}))
                        for l in labels], loc="center", ncol=min(4, len(labels)))
    save(fig, out, "legend", ["png"])

    run_metrics.to_csv(out / "run_metrics.csv", index=False)
    rows = []
    for m in METRICS:
        s = summary(run_metrics, m)
        rows += [{"label": l, "metric": m.key, **s.loc[l].to_dict()} for l in s.index]
    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)

    md = markdown_table(run_metrics, metrics, display)
    md += "\n\nCells: mean ± SEM across eval passes."
    if args.reference:
        md += f"\n\nResolve vs `{display(args.reference)}` (paired by secret, sign-flip permutation test):\n\n"
        md += "| Condition | Δ Resolve (pts) | p | secrets |\n|---|---|---|---|\n"
        for l in labels:
            if l == args.reference:
                continue
            diff, p, n = paired_resolve_test(episodes, l, args.reference)
            md += f"| {display(l)} | {diff * 100:+.1f} | {p:.4f} | {n} |\n"
    (out / "results.md").write_text(md + "\n")
    print(f"  wrote {out / 'results.md'}\n\n{md}")


if __name__ == "__main__":
    main()
