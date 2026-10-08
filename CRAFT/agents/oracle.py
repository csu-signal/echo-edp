

import copy
import json
from typing import Any, Dict, List, Optional, Tuple

# ── project imports (same as game_agents.py) ─────────────────────────────────
from agents.environment import EnhancedGameState
from agents.builder_tools import simulate_move
from structure_generator_v2 import ALL_COORDS

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — State Reconstructor
# ═════════════════════════════════════════════════════════════════════════════

def reconstruct_state(turn_data: Dict, game_data: Dict) -> EnhancedGameState:
    """
    Reconstruct an EnhancedGameState from a logged turn so that
    simulate_move / _validate_move behave exactly as they did during the game.

    Parameters
    ----------
    turn_data : dict
        One element of game["turns"] — contains structure_before, spans_before.
    game_data : dict
        The game dict (games[0]) — contains target_structure, target_spans.

    Returns
    -------
    EnhancedGameState
        Ready-to-use state. current_structure and current_spans reflect the
        board BEFORE the turn's move was attempted.
    """
    target_structure = game_data["target_structure"]
    target_spans     = game_data.get("target_spans", {})
    structure_before = turn_data["structure_before"]
    spans_before     = turn_data.get("spans_before", {})

    # ── Normalise target_spans keys: JSON stores them as strings ("0","1","2")
    #    EnhancedGameState._validate_move uses integer keys internally.
    target_spans_norm = _normalise_span_keys(target_spans)

    # ── Build a fresh state with the correct target ───────────────────────────
    state = EnhancedGameState(
        target_structure=copy.deepcopy(target_structure),
        target_spans=target_spans_norm,
        strict_target=False,   # we want to enumerate freely; validation is
                               # done via simulate_move, not strict mode
    )

    # ── Overwrite internals with logged pre-move board ────────────────────────
    # Normalise position keys: logs use "(0,0)" (no spaces), state uses same —
    # but guard against any whitespace variants just in case.
    state.current_structure = {
        _norm_pos(k): list(v)
        for k, v in structure_before.items()
    }

    # spans_before keys are string layer indices in JSON
    state.current_spans = _normalise_span_keys(spans_before)

    return state


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Correct Intended Action Enumerator
# ═════════════════════════════════════════════════════════════════════════════

# Sentinel strings used in the "flag" field of returned dicts.
FLAG_OK                  = "ok"
FLAG_MISSING_SPAN        = "missing_span_info"
FLAG_BLOCKED_REMOVE      = "blocked_correct_remove"
FLAG_SIM_FAILED          = "sim_failed"
FLAG_LARGE_SKIP_ENDPOINT = "large_block_secondary_endpoint_skipped"


def enumerate_correct_actions(state: EnhancedGameState) -> List[Dict]:
    """
    Return all currently-valid moves that make progress toward the target.

    Each returned dict has the shape:
    {
        "move": {action, block/None, position, layer, span_to},
        "structurePlacement": True,     # always True for items in this list
        "sidePlacement": bool,
        "overall_progress": float,
        "flag": FLAG_OK,
        "source": "target_place" | "excess_remove",
    }

    Items that are logically correct but currently un-executable (e.g. need
    to remove a block above first) are returned with flag=FLAG_BLOCKED_REMOVE
    and no simulation results — useful for analysis but should be excluded
    from the LLM labeling ground-truth.

    Items where span partner cannot be determined from target_spans are
    returned with flag=FLAG_MISSING_SPAN — also excluded from ground truth.
    """
    from agents.environment import EnhancedGameState
    results: List[Dict] = []
    seen_spans: set = set()   # avoid enumerating both endpoints of a large block

    current  = state.current_structure
    target   = state.target_structure
    t_spans  = state.target_spans   # {int_layer: [(pos_a, pos_b), ...]}
    c_spans  = state.current_spans  # same shape, for current board

    # for pos in ALL_COORDS: 
    for pos in state.target_structure.keys(): #making it generic for oracle scaling experiments
        pos = _norm_pos(pos)
        current_stack = current.get(pos, [])
        target_stack  = target.get(pos, [])
        current_depth = len(current_stack)
        target_depth  = len(target_stack)

        # ── Case A: target wants more blocks here ─────────────────────────────
        if current_depth < target_depth:
            needed_block = target_stack[current_depth]
            move = {
                "action"  : "place",
                "block"   : needed_block,
                "position": pos,
                "layer"   : current_depth,
                "span_to" : None,
            }

            if needed_block.endswith("l"):
                # Large block — find span partner from target_spans
                partner, span_key = _find_span_partner(pos, current_depth, t_spans)

                if partner is None:
                    results.append(_make_unrunnable(
                        move, FLAG_MISSING_SPAN, "target_place",
                        detail=f"No span partner found for {pos} layer {current_depth} in target_spans"
                    ))
                    continue

                # Deduplicate: only enumerate from the lexicographically smaller endpoint
                canonical = tuple(sorted([pos, partner]))
                if canonical in seen_spans:
                    results.append(_make_unrunnable(
                        {**move, "span_to": partner},
                        FLAG_LARGE_SKIP_ENDPOINT, "target_place",
                        detail=f"Secondary endpoint of span {canonical} — already enumerated"
                    ))
                    continue
                seen_spans.add(canonical)
                move["span_to"] = partner

            # ── Simulate ──────────────────────────────────────────────────────
            sim = simulate_move(state, copy.deepcopy(move))
            if sim["ok"] and sim["correctness"]["structurePlacement"]:
                results.append({
                    "move"               : move,
                    "structurePlacement" : True,
                    "sidePlacement"      : sim["correctness"]["sidePlacement"],
                    "overall_progress"   : sim.get("overall_progress", 0.0),
                    "flag"               : FLAG_OK,
                    "source"             : "target_place",
                })
            else:
                # Physically valid candidate that simulate rejected — record why
                results.append(_make_unrunnable(
                    move, FLAG_SIM_FAILED, "target_place",
                    detail=sim.get("hint", "simulate_move returned ok=False")
                ))

        # ── Case B: current has blocks target does NOT want ───────────────────
        elif current_depth > target_depth:
            # Only the TOP block of the stack is currently removable.
            # If the wrong block is buried under correct ones we flag it.
            top_layer = current_depth - 1
            top_block = current_stack[top_layer]

            # Is the block at current_depth-1 actually inconsistent with target?
            # (target_depth < current_depth means target wants fewer layers here)
            move = {
                "action"  : "remove",
                "block"   : top_block,   # informational only — _validate_move ignores it
                "position": pos,
                "layer"   : top_layer,
                "span_to" : None,
            }

            if top_block.endswith("l"):
                partner, _ = _find_span_partner(pos, top_layer, c_spans)
                if partner is None:
                    results.append(_make_unrunnable(
                        move, FLAG_MISSING_SPAN, "excess_remove",
                        detail=f"No span partner found for large block removal at {pos} layer {top_layer}"
                    ))
                    continue

                canonical = tuple(sorted([pos, partner]))
                if canonical in seen_spans:
                    results.append(_make_unrunnable(
                        {**move, "span_to": partner},
                        FLAG_LARGE_SKIP_ENDPOINT, "excess_remove",
                        detail=f"Secondary endpoint of span {canonical}"
                    ))
                    continue
                seen_spans.add(canonical)
                move["span_to"] = partner

            sim = simulate_move(state, copy.deepcopy(move))
            if sim["ok"] and sim["correctness"]["structurePlacement"]:
                results.append({
                    "move"               : move,
                    "structurePlacement" : True,
                    "sidePlacement"      : sim["correctness"]["sidePlacement"],
                    "overall_progress"   : sim.get("overall_progress", 0.0),
                    "flag"               : FLAG_OK,
                    "source"             : "excess_remove",
                })
            else:
                # results.append(_make_unrunnable(
                #     move, FLAG_SIM_FAILED, "excess_remove",
                #     detail=sim.get("hint", "simulate_move returned ok=False")
                # ))

          
                results.append(_make_unrunnable(
                    move, FLAG_SIM_FAILED, "excess_remove",
                    detail=sim.get("hint", "simulate_move returned ok=False")
                ))
                print(f"  [ENUM DEBUG] SIM_FAILED remove at {pos} layer {top_layer}: {sim.get('hint', sim.get('error', '?'))}")

        # ── Case C: current_depth == target_depth ─────────────────────────────
        # else:
            # ── Case C: current_depth == target_depth ─────────────────────────────
        else:
            # Check block-by-block for the first wrong layer
            for layer_idx in range(current_depth):
                if current_stack[layer_idx] != target_stack[layer_idx]:
                    
                    if layer_idx == current_depth - 1:
                        # ── Wrong block is at the TOP — removable now ──────────
                        top_block = current_stack[layer_idx]
                        move = {
                            "action"  : "remove",
                            "block"   : top_block,
                            "position": pos,
                            "layer"   : layer_idx,
                            "span_to" : None,
                        }
                        if top_block.endswith("l"):
                            partner, _ = _find_span_partner(pos, layer_idx, c_spans)
                            if partner is None:
                                results.append(_make_unrunnable(
                                    move, FLAG_MISSING_SPAN, "wrong_block_remove",
                                    detail=f"No span partner for wrong large block at {pos} layer {layer_idx}"
                                ))
                                break
                            canonical = tuple(sorted([pos, partner]))
                            if canonical in seen_spans:
                                break
                            seen_spans.add(canonical)
                            move["span_to"] = partner

                        sim = simulate_move(state, copy.deepcopy(move))
                        if sim["ok"] and sim["correctness"]["structurePlacement"]:
                            results.append({
                                "move"               : move,
                                "structurePlacement" : True,
                                "sidePlacement"      : sim["correctness"]["sidePlacement"],
                                "overall_progress"   : sim.get("overall_progress", 0.0),
                                "flag"               : FLAG_OK,
                                "source"             : "wrong_block_remove",
                            })
                        else:
                            results.append(_make_unrunnable(
                                move, FLAG_SIM_FAILED, "wrong_block_remove",
                                detail=sim.get("hint", "simulate_move returned ok=False")
                            ))
                            print(f"  [ENUM DEBUG] SIM_FAILED wrong_block_remove at {pos} "
                                f"layer {layer_idx}: {sim.get('hint', sim.get('error', '?'))}")

                    else:
                        # ── Wrong block is BURIED — need to remove correct blocks above it ──
                        # The immediate next correct action is to remove the TOP block
                        # (which is correct but sits above a wrong block)
                        top_layer = current_depth - 1
                        top_block = current_stack[top_layer]
                        move = {
                            "action"  : "remove",
                            "block"   : top_block,
                            "position": pos,
                            "layer"   : top_layer,
                            "span_to" : None,
                        }
                        if top_block.endswith("l"):
                            partner, _ = _find_span_partner(pos, top_layer, c_spans)
                            if partner is None:
                                results.append(_make_unrunnable(
                                    move, FLAG_MISSING_SPAN, "expose_buried_wrong",
                                    detail=f"No span partner for top block at {pos} layer {top_layer}"
                                ))
                                break
                            canonical = tuple(sorted([pos, partner]))
                            if canonical in seen_spans:
                                break
                            seen_spans.add(canonical)
                            move["span_to"] = partner

                        sim = simulate_move(state, copy.deepcopy(move))
                        if sim["ok"]:
                            results.append({
                                "move"               : move,
                                "structurePlacement" : True,
                                "sidePlacement"      : sim["correctness"]["sidePlacement"],
                                "overall_progress"   : sim.get("overall_progress", 0.0),
                                "flag"               : FLAG_OK,
                                "source"             : "expose_buried_wrong",
                            })
                        else:
                            results.append(_make_unrunnable(
                                move, FLAG_SIM_FAILED, "expose_buried_wrong",
                                detail=sim.get("hint", "simulate_move returned ok=False")
                            ))
                            print(f"  [ENUM DEBUG] SIM_FAILED expose_buried_wrong at {pos} "
                                f"layer {top_layer}: {sim.get('hint', sim.get('error', '?'))}")

                    break  # only report first wrong layer per cell

        

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _norm_pos(pos: str) -> str:
    """Normalise '(0, 0)' or '( 0,0 )' → '(0,0)'."""
    return "(" + ",".join(p.strip() for p in pos.strip("()").split(",")) + ")"


def _normalise_span_keys(spans: Dict) -> Dict:
    """
    Convert JSON span dict with string keys {"0": [...], "1": [...]}
    to integer keys {0: [...], 1: [...]}.
    Each span list entry is kept as-is (list or tuple both work for unpacking).
    """
    return {int(k): v for k, v in spans.items()}


def _find_span_partner(
    pos: str, layer: int, spans: Dict
) -> Tuple[Optional[str], Optional[int]]:
    """
    Look up the span partner for `pos` at `layer` in a spans dict.
    Returns (partner_pos, layer) or (None, None) if not found.
    """
    layer_spans = spans.get(layer, [])
    for entry in layer_spans:
        a, b = entry[0], entry[1]
        a, b = _norm_pos(str(a)), _norm_pos(str(b))
        if a == pos:
            return b, layer
        if b == pos:
            return a, layer
    return None, None


def _make_unrunnable(move: Dict, flag: str, source: str, detail: str = "") -> Dict:
    """Return a record for a candidate that could not be simulated."""
    return {
        "move"               : move,
        "structurePlacement" : None,
        "sidePlacement"      : None,
        "overall_progress"   : None,
        "flag"               : flag,
        "source"             : source,
        "detail"             : detail,
    }