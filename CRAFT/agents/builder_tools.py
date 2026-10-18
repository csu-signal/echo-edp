import copy
import re
import traceback
def simulate_move(game_state, move_json):
    move_json = dict(move_json)
    move_json.setdefault("span_to", None)
    move_json.setdefault("confirmation", "simulation")
    
    gs = copy.deepcopy(game_state)
    try:
        result = gs.execute_move(move_json)
        if not isinstance(result, tuple) or len(result) != 5:
            raise ValueError(f"execute_move returned {type(result)}: {result}")
        # execute_move returns 5-tuple
        success, progress_data, structurePlacement, sidePlacement, overall = result
    except Exception as e:
     
        print(f"  [SIM crash] {traceback.format_exc()}")  # print full traceback
        # return {"ok": False, "hint": f"Exception: {e}. Fix move format."}
        return {"ok": False, "error": str(e), "hint": f"Exception: {e}. Fix move format."}

    if success:
        metrics = progress_data.get("metrics", {}) if isinstance(progress_data, dict) else {}
        return {
            "ok": True,
            "blocks_correct": metrics.get("blocks_placed_correctly", 0),
            "blocks_total": metrics.get("blocks_total_target", 0),
            "overall_progress": metrics.get("overall_progress", 0),
            "correctness": {
                "structurePlacement": bool(structurePlacement),
                "sidePlacement": bool(sidePlacement),
            },
        }
    else:
        # progress_data is {"error": "..."} on failure
        error_msg = progress_data.get("error", "Unknown") if isinstance(progress_data, dict) else str(progress_data)
        expected_match = re.search(r'expected (\(\d,\d\))', error_msg)
        hint = f"FAILED: {error_msg}."
        if expected_match:
            hint += f" → Use span_to={expected_match.group(1)} instead."
        elif "Wrong layer" in error_msg or "expected" in error_msg:
            hint += " → Count stack height from board state for correct layer."
        elif "stack is empty" in error_msg:
            hint += " → Cell is empty, cannot remove."
        elif "span_to" in error_msg and "adjacent" in error_msg:
            hint += " → span_to must be directly adjacent (not diagonal)."
        elif "invisible" in error_msg:
            hint += " → Neither position nor span_to can be (1,1) or (2,1)."
        hint += " Fix ONLY the field causing this error, keep everything else."

         
        return {"ok": False, "error": error_msg, "hint": hint}
