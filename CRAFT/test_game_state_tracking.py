# test_game_state_tracking.py
import json
import copy
from typing import Dict, List, Tuple, Any
from agents.environment import EnhancedGameState

# This test file validates the CRAFT physics engine via three things about EnhancedGameState.execute_move:
# 1. Basic correctness (main first block) — given a deterministically generated sequence of correct 
# PLACE moves that builds the target structure from empty, every move succeeds and the 
# final structure exactly matches the target including spans.
# 2. Recovery after bad moves (test_recovery_with_bad_moves) — bad moves (wrong layer, non-adjacent span, remove from empty) 
# correctly fail AND don't mutate game state. After injecting bad moves, correct moves can 
# still continue and reach completion. Tests that failed moves are truly no-ops.
# 3. Reverse dismantle (reverse_dismantle_test) — builds the full target then removes 
# every block in exact reverse order. Verifies that the remove logic is the 
# inverse of place logic — after dismantling completely, structure is empty and spans are empty.


# ----------------------------
# Helpers
# ----------------------------
def norm_pos(p: str) -> str:
    # normalize "(0, 1)" -> "(0,1)"
    p = p.strip()
    if p.startswith("(") and p.endswith(")"):
        inside = p[1:-1].replace(" ", "")
        a, b = inside.split(",")
        return f"({int(a)},{int(b)})"
    return p

def load_target_from_dataset(dataset_path: str, structure_index: int):
    with open(dataset_path, "r") as f:
        loaded = json.load(f)
    sample = loaded[structure_index]

    target_structure = {norm_pos(k): v for k, v in sample["structure"].items()}
    target_spans = {int(k): [(norm_pos(a), norm_pos(b)) for (a, b) in v]
                   for k, v in sample.get("spans", {}).items()}
    return sample, target_structure, target_spans


def spans_by_layer(target_spans: Dict[int, List[Tuple[str, str]]]) -> Dict[int, Dict[str, str]]:
    """
    Build a lookup:
      span_partner[layer][pos] = partner_pos
    """
    partner = {}
    for layer, pairs in target_spans.items():
        partner[layer] = {}
        for a, b in pairs:
            a, b = norm_pos(a), norm_pos(b)
            partner[layer][a] = b
            partner[layer][b] = a
    return partner


def generate_exact_correct_place_moves(
    target_structure: Dict[str, List[str]],
    target_spans: Dict[int, List[Tuple[str, str]]],
) -> List[Dict[str, Any]]:
    """
    Deterministically generates a sequence of PLACE moves that builds the target from empty.
    - Iterates layers 0->2
    - For large blocks: place once per span-pair (anchor = lexicographically smaller endpoint)
    - For small blocks: place at that cell
    """
    span_partner = spans_by_layer(target_spans)

    # Determine max layers present in target
    max_layer = 0
    for stack in target_structure.values():
        max_layer = max(max_layer, len(stack) - 1)
    max_layer = max(max_layer, 0)

    moves: List[Dict[str, Any]] = []

    # Track which large-span anchors already emitted per layer
    emitted_span = {layer: set() for layer in range(max_layer + 1)}

    for layer in range(max_layer + 1):
        # iterate coords in stable order for reproducibility
        for pos in sorted(target_structure.keys()):
            stack = target_structure[pos]
            if layer >= len(stack):
                continue
            block = stack[layer]
            if not block or block == "n" or block == "none":
                continue

            # Large block (endswith 'l') must be placed via span
            if isinstance(block, str) and block.endswith("l"):
                if layer not in span_partner or pos not in span_partner[layer]:
                    raise ValueError(
                        f"Target says '{block}' at {pos} layer {layer} but no span pair found in target_spans."
                    )
                other = span_partner[layer][pos]
                a, b = sorted([pos, other])  # canonical
                anchor = a
                if anchor in emitted_span[layer]:
                    continue
                emitted_span[layer].add(anchor)

                moves.append({
                    "action": "place",
                    "block": block,
                    "position": anchor,
                    "layer": layer,
                    "span_to": b,  # the other end
                    "confirmation": f"TEST placing {block} at {anchor}↔{b} layer {layer}"
                })
            else:
                moves.append({
                    "action": "place",
                    "block": block,
                    "position": pos,
                    "layer": layer,
                    "span_to": None,
                    "confirmation": f"TEST placing {block} at {pos} layer {layer}"
                })

    return moves


def run_move(game: EnhancedGameState, mv: Dict[str, Any], expect_success: bool):
    ok, info, sp, side, overall = game.execute_move(copy.deepcopy(mv))
    print(f"\nMOVE: {mv}")
    print(f"  success={ok} expect={expect_success}")
    if not ok:
        print(f"  error={info.get('error')}")
    else:
        # progress_data shape depends on your TaskProgressTracker;
        # keep this generic
        metrics = info.get("metrics", {})
        if metrics:
            print(f"  overall_progress={metrics.get('overall_progress')}")
    assert ok == expect_success, f"Mismatch on success for move {mv}"
    return ok, info


def pretty_board(board: Dict[str, List[str]]):
    # compact view
    for pos in sorted(board.keys()):
        if board[pos]:
            print(pos, board[pos])

def test_recovery_with_bad_moves(game, correct_moves):
    """
    Runs: correct move -> inject a few bad moves (must fail) -> then continue correct moves.
    Ensures:
      - bad moves do NOT mutate state (structure + spans)
      - we can recover and still reach completion (progress ~ 1.0)
    """

    def snapshot(g):
        return (copy.deepcopy(g.current_structure), copy.deepcopy(g.current_spans))

    def assert_state_unchanged(before, after, why):
        b_struct, b_spans = before
        a_struct, a_spans = after
        assert b_struct == a_struct, f"{why}: structure mutated on failed move"
        assert b_spans == a_spans, f"{why}: spans mutated on failed move"

    def run_expect_fail(g, mv, why):
        before = snapshot(g)
        ok, prog, *_ = g.execute_move(mv)
        after = snapshot(g)
        assert ok is False, f"{why}: expected failure but succeeded: {mv}"
        # execute_move should return an error dict on failure
        assert isinstance(prog, dict) and "error" in prog, f"{why}: missing error for failed move: {mv}"
        assert_state_unchanged(before, after, why)

    def run_expect_success(g, mv, why):
        ok, prog, *_ = g.execute_move(mv)
        assert ok is True, f"{why}: expected success but failed: {mv} | err={prog.get('error') if isinstance(prog,dict) else prog}"
        return prog

    # 1) Do a few correct moves first to create non-trivial state
    assert len(correct_moves) >= 5, "Need at least 5 correct moves for recovery test."
    for i in range(5):
        run_expect_success(game, correct_moves[i], why=f"warmup-correct-{i}")

    # 2) Inject BAD moves that should fail but not change state
    # Bad A: wrong layer (place at higher layer than stack height)
    bad_wrong_layer = {
        "action": "place",
        "block": correct_moves[0]["block"],
        "position": correct_moves[0]["position"],
        "layer": 2,                 # intentionally wrong
        "span_to": None,
        "confirmation": "BAD wrong layer"
    }

    # Bad B: illegal span_to (non-adjacent) for a large block
    # Pick a large move from correct_moves if available; else craft one using 'bl' and a non-adjacent span
    large_mv = next((m for m in correct_moves if m.get("block","").endswith("l")), None)
    if large_mv:
        bad_non_adjacent_span = {
            "action": "place",
            "block": large_mv["block"],
            "position": large_mv["position"],
            "layer": large_mv["layer"],
            "span_to": "(2,2)" if large_mv["position"] != "(2,2)" else "(0,0)",  # almost surely non-adjacent
            "confirmation": "BAD illegal/non-adjacent span_to"
        }
    else:
        bad_non_adjacent_span = {
            "action": "place",
            "block": "bl",
            "position": "(0,1)",
            "layer": 0,
            "span_to": "(2,2)",
            "confirmation": "BAD illegal/non-adjacent span_to"
        }

    # Bad C: remove from empty cell (pick a cell you know is empty; (2,1) is usually invisible/empty)
    bad_remove_empty = {
        "action": "remove",
        "position": "(2,1)",
        "layer": 0,
        "span_to": None,
        "block": None,
        "confirmation": "BAD remove from empty"
    }

    run_expect_fail(game, bad_wrong_layer, "bad-wrong-layer")
    run_expect_fail(game, bad_non_adjacent_span, "bad-non-adjacent-span")
    run_expect_fail(game, bad_remove_empty, "bad-remove-empty")

    # 3) Continue remaining correct moves — should recover and finish
    last_prog = None
    for i in range(5, len(correct_moves)):
        last_prog = run_expect_success(game, correct_moves[i], why=f"recovery-correct-{i}")

    # 4) Sanity: should be basically complete
    if last_prog and "metrics" in last_prog:
        p = last_prog["metrics"].get("overall_progress", None)
        assert p is None or p >= 0.95, f"Expected near-complete progress after recovery, got {p}"

    assert game.is_complete(), "Game should be complete after recovery sequence."


 

def reverse_dismantle_test(game, correct_place_moves, target_spans):
    """
    Build the target using `correct_place_moves`, then dismantle by removing
    in exact reverse order. For large blocks, includes span_to so removal is legal.

    Asserts:
      - build completes
      - reverse removals all succeed
      - final structure is empty
      - final spans are empty (or at least equal to {} per layer)
    """

    # -------------------------
    # A) Build
    # -------------------------
    print("\n=== REVERSE DISMANTLE: BUILD PHASE ===")
    for mv in correct_place_moves:
        mv2 = copy.deepcopy(mv)
        ok, prog, *_ = game.execute_move(mv2)
        assert ok, f"[BUILD] Correct move failed unexpectedly: {mv}"

    assert game.is_complete(), "[BUILD] Game did not reach complete state after correct moves."
    print("✅ Build complete.")

    # -------------------------
    # B) Make reverse remove moves
    # -------------------------
    print("\n=== REVERSE DISMANTLE: GENERATE REMOVE MOVES ===")

    remove_moves = []
    for mv in reversed(correct_place_moves):
        # We remove from the same 'position' stack top layer that was placed last.
        # For small blocks: no span_to.
        # For large blocks: we MUST include span_to used in placement so _validate_move/_remove_block can remove atomically.
        rm = {
            "action": "remove",
            "position": mv["position"],
            "layer": mv["layer"],  # on successful build, that layer should now be the top at that position
            "span_to": mv.get("span_to", None),
            "block": None,
            "confirmation": f"TEST remove at {mv['position']} layer {mv['layer']}"
        }
        remove_moves.append(rm)

    print(f"Generated {len(remove_moves)} remove moves.")

    # -------------------------
    # C) Dismantle
    # -------------------------
    print("\n=== REVERSE DISMANTLE: REMOVE PHASE ===")
    for rm in remove_moves:
        rm2 = copy.deepcopy(rm)
        ok, prog, *_ = game.execute_move(rm2)
        assert ok, f"[REMOVE] Remove failed unexpectedly: {rm}\nError: {prog.get('error')}"

    # -------------------------
    # D) Assertions: empty structure + empty spans
    # -------------------------
    print("\n=== REVERSE DISMANTLE: FINAL ASSERTS ===")

    # Structure empty
    nonempty = {pos: stack for pos, stack in game.current_structure.items() if len(stack) > 0}
    assert len(nonempty) == 0, f"Expected empty structure, but found nonempty stacks: {nonempty}"

    # Spans empty (normalize: some code leaves empty lists per layer)
    spans_left = {layer: v for layer, v in game.current_spans.items() if v}
    assert spans_left == {}, f"Expected no spans left, but found: {spans_left}"

    print("✅ Reverse dismantle passed: structure + spans are empty.")

# ----------------------------
# Main test
# ----------------------------
def main():
    dataset_path = "data/structures_dataset_v2.json"
    structure_index = 0

    sample, target_structure, target_spans = load_target_from_dataset(dataset_path, structure_index)
    print(f"Testing structure id={sample.get('id')} complexity={sample.get('complexity')}")

    # strict_target=True is useful for testing legality against target (optional).
    # If you want to test "free play" legality only, set strict_target=False.
    game = EnhancedGameState(target_structure, target_spans, strict_target=True, invisible_cells=[])

    correct_moves = generate_exact_correct_place_moves(target_structure, target_spans)
    print(f"Generated {len(correct_moves)} correct PLACE moves.")

    # ---- Inject incorrect moves (should FAIL), then recover ----
    bad_moves = []

    # 1) Wrong layer (placing on layer 1 into empty stack should fail)
    if correct_moves:
        first = correct_moves[0]
        bad_moves.append({**first, "layer": first["layer"] + 1, "confirmation": "BAD wrong layer"})

    # 2) Illegal span_to for a large block (if we have any large block move)
    large = next((m for m in correct_moves if m.get("span_to") is not None), None)
    if large:
        # pick a non-adjacent span_to (guaranteed wrong in 3x3)
        bad_moves.append({**large, "span_to": "(2,2)" if large["position"] != "(2,2)" else "(0,0)",
                          "confirmation": "BAD illegal/non-adjacent span_to"})

    # 3) Remove from empty (always fail at start)
    bad_moves.append({
        "action": "remove",
        "position": "(0,0)",
        "layer": 0,
        "span_to": None,
        "block": None,  # your execute_move computes removed_block from current stack; block is not needed
        "confirmation": "BAD remove from empty"
    })

    # Run the bad moves first: all should fail (expect_success=False)
    print("\n=== BAD MOVES (expect failures) ===")
    for mv in bad_moves:
        run_move(game, mv, expect_success=False)

    # Now run correct moves: all should succeed
    print("\n=== CORRECT MOVES (expect success) ===")
    for mv in correct_moves:
        run_move(game, mv, expect_success=True)

    # Final check: current_structure should match target_structure (and spans)
    print("\n=== FINAL STATE ===")
    print("Current structure (non-empty stacks):")
    pretty_board(game.current_structure)

    # Exact match check
    assert game.current_structure == target_structure, "Final structure mismatch vs target_structure"
    # assert game.current_spans == target_spans, "Final spans mismatch vs target_spans"

    def normalize_spans(spans_dict):
        norm = {}
        for layer, spans in spans_dict.items():
            layer = int(layer)
            canon = []
            for s in spans:
                a, b = s  # works for tuple or list
                canon.append(tuple(sorted([a, b])))
            norm[layer] = sorted(canon)
        return norm

    assert normalize_spans(game.current_spans) == normalize_spans(target_spans), \
    "Final spans mismatch vs target_spans"



    print("\n✅ PASS: execute_move + _validate_move can build target and rejects bad moves, then recovers.")


    sample, target_structure, target_spans = load_target_from_dataset(dataset_path, structure_index)
    print(f"Testing structure id={sample.get('id')} complexity={sample.get('complexity')}")

    # Fresh game for normal correctness test
    game = EnhancedGameState(
    target_structure=target_structure,
    target_spans=target_spans,
    strict_target=True,
    invisible_cells=[] ) # ← disable invisible cell constraint 

    correct_moves = generate_exact_correct_place_moves(target_structure, target_spans)
    print(f"Generated {len(correct_moves)} correct PLACE moves.")

    # ------------------------------------------------
    # 1️⃣ Normal full build test
    # ------------------------------------------------
    print("\n=== FULL BUILD TEST ===")
    for mv in correct_moves:
        ok, prog, *_ = game.execute_move(copy.deepcopy(mv))
        assert ok, f"Correct move failed unexpectedly: {mv}"

    assert game.current_structure == target_structure
    print("✅ Full build test passed.")

    # ------------------------------------------------
    # 2️⃣ Recovery test (fresh game)
    # ------------------------------------------------
    print("\n=== RECOVERY TEST ===")

    recovery_game = EnhancedGameState(
    target_structure=target_structure,
    target_spans=target_spans,
    strict_target=True,
    invisible_cells=[]  # ← disable invisible cell constraint  
)

    test_recovery_with_bad_moves(recovery_game, correct_moves)

    print("✅ Recovery test passed.")

    print("\n=== REVERSE DISMANTLE TEST ===")
    dismantle_game = EnhancedGameState(
    target_structure=target_structure,
    target_spans=target_spans,
    strict_target=True,
    invisible_cells=[]  # ← disable invisible cell constraint  
)
    reverse_dismantle_test(dismantle_game, correct_moves, target_spans)
    print("✅ reverse_dismantle_test passed.")


if __name__ == "__main__":
    main()