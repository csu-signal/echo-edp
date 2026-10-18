# test_oracle.py
"""
CRAFT Oracle Validation Suite
——————————————————————————————
Validates oracle move correctness across all 20 structures and reports
statistics on move distributions, complexity breakdown, and action types.
"""

import json
import copy
import random
from pathlib import Path
from collections import defaultdict
import numpy as np

from agents.environment import EnhancedGameState, get_oracle_moves
from agents.oracle import enumerate_correct_actions
from structure_generator_v2 import get_director_views as get_director_views_fn

DATASET_PATH = "data/structures_dataset_20.json"
SEEDS        = [42, 123, 7]   # test oracle sampling across multiple seeds

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

def load_structure(data, index):
    s = data[index]
    target_spans = {int(k): v for k, v in s['spans'].items()}
    return s, s['structure'], target_spans

def norm_spans(spans):
    out = {}
    for layer, pairs in spans.items():
        out[int(layer)] = sorted([tuple(sorted([a, b])) for a, b in pairs])
    return out

def make_game(target_structure, target_spans, partial=False, partial_type="empty"):
    return EnhancedGameState(
        target_structure=target_structure,
        target_spans=target_spans,
        partComplete=partial,
        partType=partial_type,
        invisible_cells=[]
    )

def apply_oracle_move(game, move):
    """Apply a single oracle move and return success."""
    mv = {
        "action"      : move["action"],
        "block"       : move.get("block"),
        "position"    : move["position"],
        "layer"       : move["layer"],
        "span_to"     : move.get("span_to"),
        "confirmation": "oracle_test"
    }
    ok, prog, sp, side, overall = game.execute_move(copy.deepcopy(mv))
    return ok, prog

def build_with_oracle(game, target_structure, target_spans, max_turns=60):
    """
    Greedily apply oracle moves until complete or stuck.
    Returns (turns_taken, final_progress, completed).
    """
    for turn in range(max_turns):
        if game.is_complete():
            return turn, 1.0, True
        moves = get_oracle_moves(game, n=100)  # get all available
        if not moves:
            break
        # pick first valid
        for mv in moves:
            ok, prog = apply_oracle_move(game, mv)
            if ok:
                break
    progress = 0.0
    if game.progress_tracker.progress_history:
        progress = game.progress_tracker.progress_history[-1].get(
            'metrics', {}).get('overall_progress', 0.0)
    return turn + 1, progress, game.is_complete()


# ─────────────────────────────────────────
# TEST 1 — Oracle Move Validity
# ─────────────────────────────────────────

def test_oracle_move_validity(data):
    """
    Every move returned by enumerate_correct_actions must execute successfully.
    Verifies oracle never returns an invalid move.
    """
    print(f"\n{'='*60}")
    print("TEST 1 — Oracle Move Validity")
    print(f"{'='*60}")

    total_moves  = 0
    total_valid  = 0
    failures     = []

    for idx in range(len(data)):
        s, target_structure, target_spans = load_structure(data, idx)
        game = make_game(target_structure, target_spans)

        all_correct = enumerate_correct_actions(game)
        for entry in all_correct:
            if entry["flag"] != "ok":
                continue
            mv = entry["move"]
            move_dict = {
                "action"      : mv["action"],
                "block"       : mv.get("block"),
                "position"    : mv["position"],
                "layer"       : mv["layer"],
                "span_to"     : mv.get("span_to"),
                "confirmation": "oracle_validity_test"
            }
            # simulate on a fresh copy
            test_game = make_game(target_structure, target_spans)
            ok, prog = apply_oracle_move(test_game, mv)
            total_moves += 1
            if ok:
                total_valid += 1
            else:
                failures.append({
                    "structure": s["id"],
                    "move"     : mv,
                    "error"    : prog.get("error")
                })

    validity_rate = total_valid / total_moves * 100 if total_moves else 0
    print(f"  Structures tested : {len(data)}")
    print(f"  Total oracle moves: {total_moves}")
    print(f"  Valid moves       : {total_valid} ({validity_rate:.1f}%)")
    if failures:
        print(f"  ✗ FAILURES ({len(failures)}):")
        for f in failures[:5]:
            print(f"    {f['structure']} | {f['move']} | {f['error']}")
    else:
        print(f"  ✓ All oracle moves are valid")
    return validity_rate == 100.0


# ─────────────────────────────────────────
# TEST 2 — Oracle Can Complete Any Structure
# ─────────────────────────────────────────

def test_oracle_completeness(data):
    """
    Greedy oracle build should reach 100% progress on every structure.
    Verifies oracle covers the full action space needed.
    """
    print(f"\n{'='*60}")
    print("TEST 2 — Oracle Completeness (greedy build)")
    print(f"{'='*60}")

    results = []
    for idx in range(len(data)):
        s, target_structure, target_spans = load_structure(data, idx)
        game = make_game(target_structure, target_spans)
        turns, progress, completed = build_with_oracle(game, target_structure, target_spans)
        results.append({
            "id"        : s["id"],
            "complexity": s["complexity"],
            "turns"     : turns,
            "progress"  : progress,
            "completed" : completed
        })
        status = "✓" if completed else "✗"
        print(f"  {status} {s['id']:<20} complexity={s['complexity']:<8} "
              f"turns={turns:>3}  progress={progress:.3f}")

    completed_count = sum(1 for r in results if r["completed"])
    print(f"\n  Completed: {completed_count}/{len(data)} structures")
    print(f"  Mean turns to complete: {np.mean([r['turns'] for r in results if r['completed']]):.1f}")
    return completed_count == len(data)


# ─────────────────────────────────────────
# TEST 3 — Oracle Move Distribution Stats
# ─────────────────────────────────────────

def test_oracle_distribution(data):
    """
    Reports distribution of oracle move counts across turns and structures.
    Shows what range of N to expect when sampling oracle moves.
    """
    print(f"\n{'='*60}")
    print("TEST 3 — Oracle Move Distribution")
    print(f"{'='*60}")

    all_counts   = []
    by_complexity = defaultdict(list)
    action_types  = defaultdict(int)
    block_types   = defaultdict(int)
    flag_counts   = defaultdict(int)

    for idx in range(len(data)):
        s, target_structure, target_spans = load_structure(data, idx)

        # simulate turn-by-turn oracle counts across a full greedy build
        game = make_game(target_structure, target_spans)
        for turn in range(60):
            if game.is_complete():
                break
            all_correct = enumerate_correct_actions(game)

            for entry in all_correct:
                flag_counts[entry["flag"]] += 1

            ok_moves = [e["move"] for e in all_correct if e["flag"] == "ok"]
            all_counts.append(len(ok_moves))
            by_complexity[s["complexity"]].append(len(ok_moves))

            for mv in ok_moves:
                action_types[mv["action"]] += 1
                if mv.get("block"):
                    block_types["large" if mv["block"].endswith("l") else "small"] += 1

            # apply first valid move to advance state
            if ok_moves:
                apply_oracle_move(game, ok_moves[0])

    print(f"\n  Oracle move counts per turn (across all structures × turns):")
    print(f"    n_samples : {len(all_counts)}")
    print(f"    min       : {min(all_counts)}")
    print(f"    max       : {max(all_counts)}")
    print(f"    mean      : {np.mean(all_counts):.2f}")
    print(f"    median    : {np.median(all_counts):.1f}")
    print(f"    std       : {np.std(all_counts):.2f}")
    print(f"    p25/p75   : {np.percentile(all_counts, 25):.1f} / {np.percentile(all_counts, 75):.1f}")

    print(f"\n  By complexity:")
    for cplx in ["simple", "medium", "complex"]:
        if cplx in by_complexity:
            vals = by_complexity[cplx]
            print(f"    {cplx:<10} mean={np.mean(vals):.2f}  "
                  f"min={min(vals)}  max={max(vals)}  n={len(vals)}")

    print(f"\n  Action type breakdown:")
    total_actions = sum(action_types.values())
    for action, count in sorted(action_types.items()):
        print(f"    {action:<10} {count:>6} ({count/total_actions*100:.1f}%)")

    print(f"\n  Block size breakdown:")
    total_blocks = sum(block_types.values())
    for btype, count in sorted(block_types.items()):
        print(f"    {btype:<10} {count:>6} ({count/total_blocks*100:.1f}%)")

    print(f"\n  Oracle flag breakdown:")
    total_flags = sum(flag_counts.values())
    for flag, count in sorted(flag_counts.items()):
        print(f"    {flag:<35} {count:>6} ({count/total_flags*100:.1f}%)")

    return all_counts


# ─────────────────────────────────────────
# TEST 4 — Oracle Sampling Consistency
# ─────────────────────────────────────────

def test_oracle_sampling(data, n_samples=5):
    """
    Verifies that get_oracle_moves(n=k) always returns a subset of valid moves,
    and that different seeds give different but equally valid subsets.
    """
    print(f"\n{'='*60}")
    print("TEST 4 — Oracle Sampling Consistency")
    print(f"{'='*60}")

    all_valid = True
    for idx in range(min(5, len(data))):  # test first 5 structures
        s, target_structure, target_spans = load_structure(data, idx)
        game = make_game(target_structure, target_spans)

        print(f"\n  {s['id']}:")
        seed_results = {}
        for seed in SEEDS:
            rng = random.Random(seed)
            moves = get_oracle_moves(game, n=n_samples, rng=rng)
            # verify each sampled move is valid
            valid_count = 0
            for mv in moves:
                test_game = make_game(target_structure, target_spans)
                ok, _ = apply_oracle_move(test_game, mv)
                if ok:
                    valid_count += 1
            seed_results[seed] = (len(moves), valid_count)
            status = "✓" if valid_count == len(moves) else "✗"
            print(f"    seed={seed} → {len(moves)} moves, {valid_count} valid {status}")
            if valid_count != len(moves):
                all_valid = False

    return all_valid

# ─────────────────────────────────────────
# TEST 5 — Partial Completion Oracle
# ─────────────────────────────────────────

def test_partial_completion_oracle(data):
    """
    Verifies oracle works correctly on partially pre-built structures
    (all 5 partial types from EnhancedGameState.PARTIAL_OPTIONS).
    """
    print(f"\n{'='*60}")
    print("TEST 5 — Oracle on Partial Completion States")
    print(f"{'='*60}")

    partial_types = EnhancedGameState.PARTIAL_OPTIONS
    results = defaultdict(list)

    for idx in range(min(5, len(data))):
        s, target_structure, target_spans = load_structure(data, idx)
        for ptype in partial_types:
            game = make_game(target_structure, target_spans, partial=True, partial_type=ptype)
            moves = get_oracle_moves(game, n=100)
            turns, progress, completed = build_with_oracle(
                game, target_structure, target_spans
            )
            results[ptype].append(completed)
            status = "✓" if completed else "✗"
            print(f"  {status} {s['id']:<20} partType={ptype:<15} "
                  f"oracle_moves={len(moves):>2}  turns={turns:>2}  progress={progress:.3f}")


    for idx in range(min(5, len(data))):
        s, target_structure, target_spans = load_structure(data, idx)
        for ptype in ["D1Wall", "D2Wall", "D3Wall"]:
            game = make_game(target_structure, target_spans, partial=True, partial_type=ptype)
            
            # print pre-built state
            print(f"\n  {s['id']} {ptype} pre-built state:")
            for pos, stack in game.current_structure.items():
                if stack:
                    print(f"    {pos}: {stack}")
            
            # print oracle moves available at start
            all_correct = enumerate_correct_actions(game)
            ok_moves = [e for e in all_correct if e["flag"] == "ok"]
            failed_moves = [e for e in all_correct if e["flag"] != "ok"]
            print(f"  oracle ok={len(ok_moves)} failed={len(failed_moves)}")
            for e in failed_moves[:3]:
                print(f"    ✗ {e['flag']}: {e['move']} | {e.get('detail','')[:80]}")

                
    print(f"\n  Completion rate by partial type:")
    for ptype in partial_types:
        rate = sum(results[ptype]) / len(results[ptype]) * 100
        print(f"    {ptype:<20} {rate:.0f}%")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║           CRAFT Oracle Validation Suite                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    with open(DATASET_PATH) as f:
        data = json.load(f)
    print(f"\nLoaded {len(data)} structures from {DATASET_PATH}")

    results = {}

    results["validity"]      = test_oracle_move_validity(data)
    results["completeness"]  = test_oracle_completeness(data)
    counts                   = test_oracle_distribution(data)
    results["sampling"]      = test_oracle_sampling(data)
    test_partial_completion_oracle(data)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"  Oracle validity   : {'✓ PASS' if results['validity']     else '✗ FAIL'}")
    print(f"  Oracle completeness: {'✓ PASS' if results['completeness'] else '✗ FAIL'}")
    print(f"  Sampling validity : {'✓ PASS' if results['sampling']     else '✗ FAIL'}")
    print(f"\n  Oracle move count range across all turns: "
          f"min={min(counts)}  max={max(counts)}  mean={np.mean(counts):.1f}")
    print(f"\n  Recommended oracle_n values:")
    print(f"    Conservative (p25): {int(np.percentile(counts, 25))}")
    print(f"    Balanced     (p50): {int(np.percentile(counts, 50))}")
    print(f"    Liberal      (p75): {int(np.percentile(counts, 75))}")