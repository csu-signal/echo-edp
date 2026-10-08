# judge_prompts.py
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import numpy as np
import json
import re
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd

 

#oracle alignment and pragamtic sufficiency judge prompts 


# PS1. Do the director messages collectively identify at least one 
# specific location on the board that needs a block placed or removed?

# PS2. Do the director messages collectively specify the correct block 
# type — both color and size (small vs large) — for the intended action?

# PS3. Would a builder reading only these director messages have 
# sufficient positional and block-type information to execute the 
# intended move without independent spatial reasoning about the 
# target structure?

# PS4. Do the director messages use precise spatial anchors that 
# uniquely identify the target location, or vague relative language 
# that could map to multiple grid positions?

# PS5. Does the builder confirmation indicate it correctly understood 
# the directors' collective intent, regardless of whether the move 
# execution succeeded?

# PS6. If the builder did not execute the intended move correctly, 
# was the failure attributable to director underspecification 
# rather than builder execution mechanics (wrong layer, missing span)?

# Would a rational builder reading only these director messages have sufficient information to select an oracle correct move without independent spatial reasoning?


def ps_judge_prompt(
    director_messages,
    oracle_moves,
    board_state,
    builder_confirmation,
    condition,  # 'C1_followed' or 'C2_not_followed'
):
    oracle_str = "\n".join(
        f"  {m['action'].upper()} {m['block']} at {m['position']} "
        f"layer {m['layer']}"
        f"{f' spanning to {m[\"span_to\"]}' if m.get('span_to') else ''}"
        for m in oracle_moves
    )
    
    msgs_str = "\n".join(
        f"  {did}: {msg}"
        for did, msg in director_messages.items()
        if msg.strip()
    )
    
    condition_note = (
        "NOTE: The builder successfully selected an oracle correct move this turn."
        if condition == 'C1_followed'
        else "NOTE: The builder did NOT select an oracle correct move this turn."
    )
    
    return f"""You are evaluating whether the collective director messages \
in a collaborative 3D construction task were pragmatically sufficient \
to guide a builder agent toward a correct verified move.

Three directors each hold a private 2D projection of a target 3D structure \
and must communicate with a builder through natural language only. \
The builder does not have access to the target structure and must \
infer the correct action from director messages alone.

CURRENT BOARD STATE:
{board_state}

ORACLE CORRECT MOVES THIS TURN (verified by game engine as making \
forward progress toward target):
{oracle_str}

DIRECTOR MESSAGES THIS TURN:
{msgs_str}

BUILDER RESPONSE AND REASONING:
{builder_confirmation}

{condition_note}

EVALUATION — for each question answer "Yes", "No", or "Unclear" \
with a brief one-sentence justification.

PS1. Do the director messages collectively identify at least one \
specific location on the board that needs a block placed or removed?

PS2. Do the director messages collectively specify the correct block \
type — both color AND size (small vs large/domino) — for at least \
one of the oracle correct moves?

PS3. Would a rational builder reading only these director messages \
have sufficient information to select at least one oracle correct \
move without needing to perform independent spatial reasoning \
about the target structure?

PS4. Do the director messages use precise spatial anchors that \
uniquely identify the target location (explicit coordinates, \
unambiguous landmark references, or clear directional anchors), \
rather than vague relative language that could map to multiple \
grid positions?

PS5. Does the builder confirmation indicate it correctly understood \
the directors' collective intent for this turn, regardless of \
whether the move execution ultimately succeeded?

PS6. If the builder did not execute the correct move, was the \
failure primarily attributable to director underspecification \
(missing position, wrong block type, wrong size, ambiguous \
spatial language) rather than builder execution mechanics \
(wrong layer computation, missing span endpoint, stacking \
constraint violation)?
Answer "N/A" if the builder successfully executed an oracle \
correct move this turn.

Return your response as a JSON object with keys PS1 through PS6, \
each containing an "answer" field ("Yes", "No", "Unclear", or "N/A") \
and a "reason" field (one sentence). \
Return only valid JSON with no additional text or markdown."""




def sg_judge_prompt(
    target_view,
    board_state,
    oracle_moves,
    internal_thinking,
    # available but not yet used:
    # wall_cells=None,
    # prev_failed_move=None,
    # public_message=None,
):
    return f"""You are evaluating the spatial grounding quality of a \
director agent in a collaborative construction task.
The director has a private view of one wall of a 3D target structure \
and must reason about what blocks are missing before instructing a builder.

TARGET VIEW (what this director needs the structure to look like):
{target_view}

CURRENT BOARD STATE:
{board_state}

ORACLE CORRECT MOVES THIS TURN:
{oracle_moves}

DIRECTOR INTERNAL REASONING:
{internal_thinking}

EVALUATION:
For each question below answer with "Yes", "No", or "Unclear" and \
provide a brief one-sentence justification.

Questions:
1. Does the internal reasoning correctly identify at least one block \
that is missing from this director's visible wall, based on the target \
view and current board state?
2. Does the internal reasoning avoid describing blocks or positions \
that are already correctly placed on the board?
3. Does the internal reasoning reference the correct layer for the \
missing block, accounting for what is already stacked at that position?
4. Does the internal reasoning identify at least one action that \
matches or closely corresponds to one of the oracle correct moves?
5. Is the physical action implied by the internal reasoning executable \
given the current board state, respecting stacking order?
6. Does the internal reasoning correctly interpret the size of the \
missing block (small versus large) based on the target view?
7. Does the internal reasoning stay within this director's visible \
wall cells rather than describing cells belonging to another director?

Return your response as a JSON object with keys SG1 through SG7, \
each containing an "answer" field ("Yes", "No", or "Unclear") and \
a "reason" field. Return only valid JSON with no additional text."""


def mm_judge_prompt(
    internal_thinking,
    public_message,
    other_messages,
    conv_window,
    # available but not yet used:
    # prev_failed_move=None,
    # progress_delta=None,
    # board_state=None,
    # oracle_moves=None,
):
    other_str = "\n".join(
        f"{d}: {msg}" for d, msg in other_messages.items() if msg
    )
    return f"""You are evaluating the Theory-of-Mind quality of a \
director agent in a collaborative construction task.
The director must produce a public message calibrated to what the \
builder and other directors already know, not just what she can see.

DIRECTOR INTERNAL REASONING (background context only):
{internal_thinking}

DIRECTOR PUBLIC MESSAGE (what was broadcast to all agents):
{public_message}

 OTHER DIRECTORS' MESSAGES THIS TURN:
{other_str}

RECENT CONVERSATION HISTORY (last few turns):
{conv_window}

EVALUATION:
For each question below answer with "Yes", "No", or "Unclear" and \
provide a brief one-sentence justification.

Questions:
1. Does the public message add information not already communicated \
by the other directors in this turn or the immediately preceding turn?
2. Does the public message avoid repeating an instruction already \
given and acted upon in a previous turn?
3. Does the public message reflect awareness of what the builder \
already knows from the conversation history?
4. Does the public message accurately translate the key finding from \
the internal reasoning into natural language without losing critical \
spatial information?
5. Does the public message focus on information uniquely visible from \
this director's wall rather than information another director could \
have provided equally well?
6. Is the public message specific enough for the builder to execute \
without needing further clarification, naming a block type, location, \
and action?
7. If another director gave a conflicting instruction this turn, does \
the public message acknowledge or attempt to resolve the conflict?
8. Does the public message show awareness of the boundary between \
what this director uniquely sees and what other directors can also see?

Return your response as a JSON object with keys MM1 through MM8, \
each containing an "answer" field ("Yes", "No", or "Unclear") and \
a "reason" field. Return only valid JSON with no additional text."""


# run_judges.py
import json
import re
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
# from judge_prompts import sg_judge_prompt, mm_judge_prompt
import time
import random

load_dotenv()


MODEL_NAMES = [
    "qwen-7b",
    "qwen-14b",
    "llama-8b",
    "mistral-7b",
    "gemma-9b",
    "deepseek-v2-lite",
    "qwen-32b",
    "qwen-72b",
]

DIRECTOR_CELLS = {
    "D1": ["(0,0)", "(1,0)", "(2,0)"],
    "D2": ["(0,0)", "(0,1)", "(0,2)"],
    "D3": ["(0,2)", "(1,2)", "(2,2)"],
}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_3"))

# ── Data loading ──────────────────────────────────────────────

def load_oracle_runs(oracle_dir, model_name, builder_model="gpt-4o-mini"):
    model_dir = Path(oracle_dir) / f"{model_name}_{builder_model}"
    files = sorted(model_dir.glob("*.json"))
    games = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        game = d["games"][0]
        game["_file"] = str(f)
        game["_model"] = model_name
        games.append(game)
    print(f"  {model_name}: {len(games)} games loaded")
    return games

def load_all_models(oracle_dir, model_names, builder_model="gpt-4o-mini"):
    all_games = {}
    for m in model_names:
        try:
            all_games[m] = load_oracle_runs(oracle_dir, m, builder_model)
        except Exception as e:
            print(f"  FAILED {m}: {e}")
    return all_games

# ── Field extraction ──────────────────────────────────────────


def extract_ps_judge_inputs(root_dirs, df_cond, 
                             run_filter=3,
                             c1_sample=500,
                             rng_seed=42):
    """
    Extract all C2 turns + sampled C1 turns with fields needed
    for PS judge: director messages, oracle moves, board state,
    builder confirmation, condition.
    """
    import random
    rng = random.Random(rng_seed)
    
    # get C2 turn identifiers
    c2_ids = set(zip(
        df_cond[df_cond['condition']=='C2_not_followed']['model'],
        df_cond[df_cond['condition']=='C2_not_followed']['file'],
        df_cond[df_cond['condition']=='C2_not_followed']['turn']
    ))
    
    # get C1 turn identifiers — sample
    c1_all = list(zip(
        df_cond[df_cond['condition']=='C1_followed']['model'],
        df_cond[df_cond['condition']=='C1_followed']['file'],
        df_cond[df_cond['condition']=='C1_followed']['turn']
    ))
    c1_sample_ids = set(map(tuple, 
        rng.sample(c1_all, min(c1_sample, len(c1_all)))))
    
    records = []
    
    for root_dir in root_dirs:
        for model_dir in sorted(Path(root_dir).iterdir()):
            if not model_dir.is_dir():
                continue
            
            label = clean_model_label_new(model_dir.name)
            files = [f for f in sorted(model_dir.glob("*.json"))
                     if f.name.endswith(f"_{run_filter}.json")]
            
            for fpath in files:
                fname = fpath.name
                
                # check relevance
                relevant_c2 = {(m,f,t) for m,f,t in c2_ids 
                               if m==label and f==fname}
                relevant_c1 = {(m,f,t) for m,f,t in c1_sample_ids 
                               if m==label and f==fname}
                
                if not relevant_c2 and not relevant_c1:
                    continue
                
                with open(fpath) as f:
                    d = json.load(f)
                game = d["games"][0]
                
                for turn in game.get("turns", []):
                    tn  = turn.get("turn_number")
                    key = (label, fname, tn)
                    
                    if key not in relevant_c2 and key not in relevant_c1:
                        continue
                    
                    condition = ('C2_not_followed' 
                                 if key in relevant_c2 
                                 else 'C1_followed')
                    
                    oracle    = turn.get("oracle_moves", [])
                    attempted = turn.get("move_attempted", {})
                    dr        = turn.get("director_responses", {})
                    board     = turn.get("structure_before", {})
                    
                    # director messages
                    dir_msgs = {
                        did: dr.get(did,{}).get("public_message","").strip()
                        for did in ["D1","D2","D3"]
                        if dr.get(did,{}).get("public_message","").strip()
                    }
                    
                    if not dir_msgs:
                        continue
                    
                    # builder confirmation
                    confirm = attempted.get("confirmation", "")
                    action  = attempted.get("action", "")
                    block   = attempted.get("block", "")
                    pos     = attempted.get("position", "")
                    layer   = attempted.get("layer", "")
                    
                    builder_str = (
                        f"Action attempted: {action} {block} "
                        f"at {pos} layer {layer}\n"
                        f"Builder reasoning: {confirm}"
                    )
                    
                    # board state — non-empty only
                    board_str = json.dumps(
                        {k:v for k,v in board.items() if v},
                        indent=2
                    )
                    
                    records.append({
                        'model':              label,
                        'file':               fname,
                        'turn':               tn,
                        'condition':          condition,
                        'complexity':         game.get('complexity'),
                        'director_messages':  dir_msgs,
                        'oracle_moves':       oracle,
                        'board_state':        board_str,
                        'builder_confirmation': builder_str,
                        'failure_type':       classify_failure(turn),
                    })
    
    df_ps = pd.DataFrame(records)
    
    print(f"PS judge inputs extracted:")
    print(f"  C2 turns: {len(df_ps[df_ps['condition']=='C2_not_followed'])}")
    print(f"  C1 turns: {len(df_ps[df_ps['condition']=='C1_followed'])}")
    print(f"  Total:    {len(df_ps)}")
    print(f"\nPer model:")
    print(df_ps.groupby(['model','condition']).size().unstack(fill_value=0).to_string())
    
    return df_ps


 


def extract_sg_fields(game, turn, did):
    """Extract all fields needed for SG-Judge. 
    Only a subset is passed to the prompt currently.
    All fields are extracted here for future use."""
    turn_data        = game["turns"][turn]
    turn_num         = turn_data["turn_number"]
    target_view      = game["target_director_views"].get(did, {})
    board_state      = turn_data.get("structure_before", {})
    oracle_moves     = turn_data.get("oracle_moves", [])
    director_resp    = turn_data.get("director_responses", {}).get(did, {})
    internal_thinking = director_resp.get("internal_thinking", "")
    public_message   = director_resp.get("public_message", "")
    wall_cells       = DIRECTOR_CELLS.get(did, [])

    # available but not yet in prompt
    prev_failed_move = False
    if turn > 0:
        prev_turn = game["turns"][turn - 1]
        prev_failed_move = prev_turn.get("failed_move", False)

    return {
        # ── in prompt ─────────────────────────────────────────
        "target_view"       : json.dumps(target_view, indent=2),
        "board_state"       : json.dumps(board_state, indent=2),
        "oracle_moves"      : json.dumps(oracle_moves, indent=2),
        "internal_thinking" : internal_thinking,
        # ── available, not yet in prompt ──────────────────────
        # "wall_cells"        : wall_cells,
        # "prev_failed_move"  : prev_failed_move,
        # "public_message"    : public_message,
        # ── metadata ──────────────────────────────────────────
        "_did"              : did,
        "_turn_num"         : turn_num,
        "_structure_id"     : game["structure_id"],
        "_model"            : game["_model"],
    }
 
def extract_mm_fields(game, turn, did, conv_window=3):
    turn_data         = game["turns"][turn]
    turn_num          = turn_data["turn_number"]
    director_resp     = turn_data.get("director_responses", {}).get(did, {})
    internal_thinking = director_resp.get("internal_thinking", "")
    public_message    = director_resp.get("public_message", "")

    other_messages = {
        d: turn_data.get("director_responses", {}).get(d, {}).get("public_message", "")
        for d in ["D1", "D2", "D3"] if d != did
    }

    # fix: define conv_snapshot first, then slice
    conv_snapshot    = turn_data.get("conversation_snapshot", [])
    conv_window_data = conv_snapshot[-conv_window * 4:] if conv_snapshot else []

    prev_failed_move = False
    progress_delta   = None
    board_state      = turn_data.get("structure_before", {})
    oracle_moves     = turn_data.get("oracle_moves", []) #this logs is not the same as builder followed oracle move (which needs to be fixed); this is raw moves, should be ok

    if turn > 0:
        prev_turn        = game["turns"][turn - 1]
        prev_failed_move = prev_turn.get("failed_move", False)
        prev_pd          = prev_turn.get("progress_data", {})
        progress_delta   = prev_pd.get("progress_delta") \
                           if isinstance(prev_pd, dict) else None

    return {
        # ── in prompt ─────────────────────────────────────────
        "internal_thinking" : internal_thinking,
        "public_message"    : public_message,
        "other_messages"    : other_messages,
        "conv_window"       : "\n".join(conv_window_data),
        # ── available, not yet in prompt ──────────────────────
        # "prev_failed_move" : prev_failed_move,
        # "progress_delta"   : progress_delta,
        # "board_state"      : json.dumps(board_state, indent=2),
        # "oracle_moves"     : json.dumps(oracle_moves, indent=2),
        # ── metadata ──────────────────────────────────────────
        "_did"              : did,
        "_turn_num"         : turn_num,
        "_structure_id"     : game["structure_id"],
        "_model"            : game["_model"],
        "_oracle_moves"     : oracle_moves,   # stored with _ prefix, not passed to prompt
        "_prev_failed_move" : prev_failed_move,
        "_progress_delta"   : progress_delta,
    }

# ── API call ──────────────────────────────────────────────────

def sanitize_for_api(text: str) -> str:
    """Remove control characters that break JSON serialization."""
    if not isinstance(text, str):
        text = str(text)
    # remove null bytes and other control chars except \n \t
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # replace lone surrogates
    text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    return text

def call_judge(prompt, system="You are a careful evaluator. Return only valid JSON.", max_retries=5):
    prompt = sanitize_for_api(prompt)
    system = sanitize_for_api(system)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1500,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            # 400 invalid JSON body — don't retry, prompt is malformed
            if "could not parse the json body" in err_str or ("400" in err_str and "json" in err_str):
                print(f"[SKIP] Malformed prompt (400 JSON error), skipping: {str(e)[:80]}")
                return None
            if attempt < max_retries - 1:
                delay = 2 ** attempt + random.uniform(0, 1)
                print(f"[Retry {attempt+1}] Error: {e} | sleeping {delay:.2f}s")
                time.sleep(delay)
            else:
                print(f"[FAIL] After {max_retries} attempts: {e}")
                return None
# ── Response parsing ──────────────────────────────────────────

def parse_judge_response(raw, judge_type):
    """Two-stage parser: try direct JSON, then strip fences, then regex fallback."""
    if raw is None:
        return None, True

    # stage 1 — direct parse
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        pass

    # stage 2 — strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned), False
    except json.JSONDecodeError:
        pass

    # stage 3 — regex fallback: extract key: "Yes"/"No"/"Unclear" pairs
    pattern = r'"?(SG|MM)\d+"?\s*:\s*\{[^}]*"answer"\s*:\s*"(Yes|No|Unclear)"'
    matches = re.findall(pattern, raw, re.IGNORECASE)
    if matches:
        result = {}
        for prefix, answer in matches:
            key = f"{prefix}{len(result)+1}"
            result[key] = {"answer": answer, "reason": "parsed via regex fallback"}
        return result, False

    return None, True  # judge_failed

def compute_score(parsed, judge_type):
    """Compute numeric score from parsed response.
    Yes=1, No=0, Unclear=0.5. Computed outside LLM call."""
    if parsed is None:
        return None
    prefix = "SG" if judge_type == "sg" else "MM"
    scores = []
    for k, v in parsed.items():
        if k.startswith(prefix):
            ans = v.get("answer", "Unclear") if isinstance(v, dict) else v
            if ans == "Yes":
                scores.append(1.0)
            elif ans == "No":
                scores.append(0.0)
            else:
                scores.append(0.5)
    return sum(scores) / len(scores) if scores else None


def aggregate_all_models(output_dir, judge_type='sg'):
    """Aggregate per-question metrics across all models."""
    output_dir = Path(output_dir)
    filename   = f"{judge_type}_judge_results.jsonl"
    
    all_rows = []
    for model_dir in sorted(output_dir.iterdir()):
        path = model_dir / filename
        if not path.exists():
            continue
        df, _ = compute_per_question_metrics(path, judge_type)
        all_rows.append(df)
    
    if not all_rows:
        return
    
    combined = pd.concat(all_rows, ignore_index=True)
    
    # overall per-question across all models
    overall = combined.groupby('question')['numeric'].agg(
        mean='mean', std='std', n='count'
    ).round(3)
    
    print(f"\n{'='*60}")
    print(f"OVERALL PER-QUESTION ({judge_type.upper()}) ACROSS ALL MODELS")
    print(f"{'='*60}")
    print(overall.to_string())
    
    # per model per question pivot
    pivot = combined.groupby(
        ['model','question'])['numeric'].mean().round(3).unstack()
    print(f"\nPer-model per-question pivot:")
    print(pivot.to_string())
    
    return combined, pivot

    
def compute_per_question_metrics(results_jsonl_path, judge_type='sg'):
    """
    Load JSONL results and compute per-question metrics.
    Returns a DataFrame with mean score per question per model.
    """
    import numpy as np
    
    rows = []
    with open(results_jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            scores = r.get('scores')
            if not scores:
                continue
            prefix = 'SG' if judge_type == 'sg' else 'MM'
            for k, v in scores.items():
                if not k.startswith(prefix):
                    continue
                ans = v.get('answer', 'Unclear') \
                      if isinstance(v, dict) else v
                numeric = 1.0 if ans == 'Yes' \
                         else 0.0 if ans == 'No' \
                         else 0.5
                rows.append({
                    'model'        : r['model'],
                    'structure_id' : r['structure_id'],
                    'turn'         : r['turn'],
                    'director'     : r['director'],
                    'question'     : k,
                    'answer'       : ans,
                    'numeric'      : numeric,
                    'judge_failed' : r['judge_failed'],
                })
    
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data found")
        return df
    
    # per question mean per model
    summary = df.groupby(['model', 'question'])['numeric'].agg(
        mean='mean', std='std', n='count'
    ).round(3).reset_index()
    
    # pivot for easy reading
    pivot = summary.pivot(index='model', columns='question', values='mean')
    
    print(f"\n{'='*60}")
    print(f"PER-QUESTION MEAN SCORES ({judge_type.upper()} JUDGE)")
    print(f"{'='*60}")
    print(pivot.to_string())
    
    # also print answer distribution per question per model
    print(f"\nAnswer distributions:")
    for q in sorted(df['question'].unique()):
        print(f"\n  {q}:")
        dist = df[df['question']==q].groupby(
            ['model','answer'])['numeric'].count().unstack(fill_value=0)
        print(dist.to_string())
    
    return df, pivot

# ── Main evaluation loop ──────────────────────────────────────



    
def run_judges(all_games):
    OUTPUT_DIR.mkdir(exist_ok=True)

    for model_name, games in all_games.items():
        print(f"\n{'='*55}")
        print(f"  MODEL: {model_name}")
        print(f"{'='*55}")

        out_dir = OUTPUT_DIR / model_name
        out_dir.mkdir(exist_ok=True)

        sg_results = []
        mm_results = []

        games_to_run = games[:TEST_STRUCTURES] if TEST_MODE else games

        for game in tqdm(games_to_run, desc=f"{model_name} games"):
            turns     = game["turns"]
            n_turns   = len(turns)
            directors = TEST_DIRECTORS if TEST_MODE else ["D1", "D2", "D3"]
            turn_range = range(min(TEST_TURNS, n_turns)) if TEST_MODE else range(n_turns)

            for t in turn_range:
                for did in directors:
                    # skip if director did not respond this turn
                    if did not in game["turns"][t].get("director_responses", {}):
                        continue

                    # ── SG-Judge ──────────────────────────────
                    sg_fields  = extract_sg_fields(game, t, did)
                    sg_prompt  = sg_judge_prompt(**{
                        k: v for k, v in sg_fields.items()
                        if not k.startswith("_")
                    })
                    sg_raw     = call_judge(sg_prompt)
                    sg_parsed, sg_failed = parse_judge_response(sg_raw, "sg")
                    sg_score   = compute_score(sg_parsed, "sg")

                    sg_results.append({
                        "model"        : model_name,
                        "structure_id" : sg_fields["_structure_id"],
                        "turn"         : sg_fields["_turn_num"],
                        "director"     : did,
                        "judge"        : "SG",
                        "scores"       : sg_parsed,
                        "sg_score"     : sg_score,
                        "judge_failed" : sg_failed,
                        "raw_response" : sg_raw,
                    })

                    # ── MM-Judge ──────────────────────────────
                    mm_fields  = extract_mm_fields(game, t, did)
                    mm_prompt  = mm_judge_prompt(**{
                        k: v for k, v in mm_fields.items()
                        if not k.startswith("_")
                    })
                    mm_raw     = call_judge(mm_prompt)
                    mm_parsed, mm_failed = parse_judge_response(mm_raw, "mm")
                    mm_score   = compute_score(mm_parsed, "mm")

                    mm_results.append({
                        "model"        : model_name,
                        "structure_id" : mm_fields["_structure_id"],
                        "turn"         : mm_fields["_turn_num"],
                        "director"     : did,
                        "judge"        : "MM",
                        "scores"       : mm_parsed,
                        "mm_score"     : mm_score,
                        "judge_failed" : mm_failed,
                        "raw_response" : mm_raw,
                    })

        # ── Save per model ─────────────────────────────────────
        sg_path = out_dir / "sg_judge_results.jsonl"
        mm_path = out_dir / "mm_judge_results.jsonl"

        with open(sg_path, "w") as f:
            for r in sg_results:
                f.write(json.dumps(r) + "\n")

        with open(mm_path, "w") as f:
            for r in mm_results:
                f.write(json.dumps(r) + "\n")

        # ── Summary stats ──────────────────────────────────────
        sg_scores = [r["sg_score"] for r in sg_results if r["sg_score"] is not None]
        mm_scores = [r["mm_score"] for r in mm_results if r["mm_score"] is not None]
        sg_failed = sum(1 for r in sg_results if r["judge_failed"])
        mm_failed = sum(1 for r in mm_results if r["judge_failed"])

        import numpy as np
        summary = {
            "model"          : model_name,
            "n_sg"           : len(sg_results),
            "n_mm"           : len(mm_results),
            "sg_mean"        : float(np.mean(sg_scores)) if sg_scores else None,
            "sg_std"         : float(np.std(sg_scores))  if sg_scores else None,
            "mm_mean"        : float(np.mean(mm_scores)) if mm_scores else None,
            "mm_std"         : float(np.std(mm_scores))  if mm_scores else None,
            "sg_failed"      : sg_failed,
            "mm_failed"      : mm_failed,
        }

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"  SG mean: {summary['sg_mean']:.3f} ± {summary['sg_std']:.3f}"
              f" | failed: {sg_failed}")
        print(f"  MM mean: {summary['mm_mean']:.3f} ± {summary['mm_std']:.3f}"
              f" | failed: {mm_failed}")
        print(f"  Saved to {out_dir}")

 
def run_judges_parallel(all_games, output_dir, 
                        judge_model="gpt-4o-mini",
                        conv_window=3, max_workers=10):
    """Run judge calls in parallel with thread pool."""
    
    def process_one_turn(args):
        game, t, did, model_name = args
        if did not in game["turns"][t].get("director_responses", {}):
            return None, None
        
        sg_fields = extract_sg_fields(game, t, did)
        sg_prompt = sg_judge_prompt(**{
            k: v for k, v in sg_fields.items()
            if not k.startswith("_")
        })
        sg_raw            = call_judge(sg_prompt)
        sg_parsed, sg_fail = parse_judge_response(sg_raw, "sg")
        sg_score          = compute_score(sg_parsed, "sg")
        sg_row = {
            "model"        : model_name,
            "structure_id" : sg_fields["_structure_id"],
            "turn"         : sg_fields["_turn_num"],
            "director"     : did,
            "scores"       : sg_parsed,
            "sg_score"     : sg_score,
            "judge_failed" : sg_fail,
        }

        mm_fields = extract_mm_fields(game, t, did, conv_window)
        mm_prompt = mm_judge_prompt(**{
            k: v for k, v in mm_fields.items()
            if not k.startswith("_")
        })
        mm_raw            = call_judge(mm_prompt)
        mm_parsed, mm_fail = parse_judge_response(mm_raw, "mm")
        mm_score          = compute_score(mm_parsed, "mm")
        mm_row = {
            "model"        : model_name,
            "structure_id" : mm_fields["_structure_id"],
            "turn"         : mm_fields["_turn_num"],
            "director"     : did,
            "scores"       : mm_parsed,
            "mm_score"     : mm_score,
            "judge_failed" : mm_fail,
        }
        return sg_row, mm_row

    output_dir = Path(output_dir)
    # output_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    master = {}

    for model_name, games in all_games.items():
        print(f"\n{'='*55}")
        print(f"  MODEL: {model_name}  ({len(games)} games)")
        print(f"{'='*55}")

        out_dir = output_dir / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # build all tasks
        tasks = [
            (game, t, did, model_name)
            for game in games
            for t in range(len(game["turns"]))
            for did in ["D1", "D2", "D3"]
        ]

        sg_results = []
        mm_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_one_turn, task): task
                for task in tasks
            }
            for future in tqdm(as_completed(futures),
                               total=len(futures),
                               desc=model_name):
                sg_row, mm_row = future.result()
                if sg_row:
                    sg_results.append(sg_row)
                if mm_row:
                    mm_results.append(mm_row)

        # save
        with open(out_dir / "sg_judge_results.jsonl", "w") as f:
            for r in sg_results:
                f.write(json.dumps(r) + "\n")
        with open(out_dir / "mm_judge_results.jsonl", "w") as f:
            for r in mm_results:
                f.write(json.dumps(r) + "\n")

        master[model_name] = {"sg": sg_results, "mm": mm_results}
        print(f"  sg={len(sg_results)}  mm={len(mm_results)}")

    with open(output_dir / "all_judge_results.pkl", "wb") as f:
        pickle.dump(master, f)
# ── Summary stats — INSIDE the for model_name loop ──
        sg_scores = [r["sg_score"] for r in sg_results if r["sg_score"] is not None]
        mm_scores = [r["mm_score"] for r in mm_results if r["mm_score"] is not None]
        sg_failed = sum(1 for r in sg_results if r["judge_failed"])
        mm_failed = sum(1 for r in mm_results if r["judge_failed"])

        summary = {
            "model"   : model_name,
            "n_sg"    : len(sg_results),
            "n_mm"    : len(mm_results),
            "sg_mean" : float(np.mean(sg_scores)) if sg_scores else None,
            "sg_std"  : float(np.std(sg_scores))  if sg_scores else None,
            "mm_mean" : float(np.mean(mm_scores)) if mm_scores else None,
            "mm_std"  : float(np.std(mm_scores))  if mm_scores else None,
            "sg_failed": sg_failed,
            "mm_failed": mm_failed,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  SG mean: {summary['sg_mean']:.3f} ± {summary['sg_std']:.3f}"
              f" | failed: {sg_failed}")
        print(f"  MM mean: {summary['mm_mean']:.3f} ± {summary['mm_std']:.3f}"
              f" | failed: {mm_failed}")

    # master pickle saved AFTER loop
    with open(output_dir / "all_judge_results.pkl", "wb") as f:
        pickle.dump(master, f)

    return master

def run_judges_full(all_games, output_dir, judge_model="gpt-4o-mini", conv_window=3):
    """
    Full run — all models, all structures, all turns, all directors.
    Saves per-model JSONL + one master pickle with everything.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    master = {}  # model -> {sg: [...], mm: [...]}
    
    for model_name, games in all_games.items():
        print(f"\n{'='*55}")
        print(f"  MODEL: {model_name}  ({len(games)} games)")
        print(f"{'='*55}")
        
        out_dir = output_dir / model_name
        out_dir.mkdir(exist_ok=True)
        
        sg_results = []
        mm_results = []
        
        for game in tqdm(games, desc=f"{model_name}"):
            turns   = game["turns"]
            n_turns = len(turns)
            
            for t in range(n_turns):
                for did in ["D1", "D2", "D3"]:
                    if did not in game["turns"][t].get(
                            "director_responses", {}):
                        continue
                    
                    # ── SG ────────────────────────────────────
                    sg_fields = extract_sg_fields(game, t, did)
                    sg_prompt = sg_judge_prompt(**{
                        k: v for k, v in sg_fields.items()
                        if not k.startswith("_")
                    })
                    sg_raw              = call_judge(sg_prompt)
                    sg_parsed, sg_fail  = parse_judge_response(sg_raw, "sg")
                    sg_score            = compute_score(sg_parsed, "sg")
                    
                    sg_row = {
                        "model"        : model_name,
                        "structure_id" : sg_fields["_structure_id"],
                        "turn"         : sg_fields["_turn_num"],
                        "director"     : did,
                        "scores"       : sg_parsed,
                        "sg_score"     : sg_score,
                        "judge_failed" : sg_fail,
                    }
                    sg_results.append(sg_row)
                    
                    # ── MM ────────────────────────────────────
                    mm_fields = extract_mm_fields(
                        game, t, did, conv_window=conv_window)
                    mm_prompt = mm_judge_prompt(**{
                        k: v for k, v in mm_fields.items()
                        if not k.startswith("_")
                    })
                    mm_raw              = call_judge(mm_prompt)
                    mm_parsed, mm_fail  = parse_judge_response(mm_raw, "mm")
                    mm_score            = compute_score(mm_parsed, "mm")
                    
                    mm_row = {
                        "model"        : model_name,
                        "structure_id" : mm_fields["_structure_id"],
                        "turn"         : mm_fields["_turn_num"],
                        "director"     : did,
                        "scores"       : mm_parsed,
                        "mm_score"     : mm_score,
                        "judge_failed" : mm_fail,
                    }
                    mm_results.append(mm_row)
        
        # save per-model JSONL
        with open(out_dir / "sg_judge_results.jsonl", "w") as f:
            for r in sg_results:
                f.write(json.dumps(r) + "\n")
        with open(out_dir / "mm_judge_results.jsonl", "w") as f:
            for r in mm_results:
                f.write(json.dumps(r) + "\n")
        
        master[model_name] = {"sg": sg_results, "mm": mm_results}
        print(f"  sg={len(sg_results)} rows  mm={len(mm_results)} rows")
    
    # save master pickle
    master_path = output_dir / "all_judge_results.pkl"
    with open(master_path, "wb") as f:
        pickle.dump(master, f)
    print(f"\nMaster pickle saved: {master_path}")
    
    return master


def flatten_to_df(master, judge_type='sg'):
    """
    Flatten master dict to a per-question DataFrame.
    judge_type: 'sg' or 'mm'
    """
    prefix = 'SG' if judge_type == 'sg' else 'MM'
    rows   = []
    
    for model_name, data in master.items():
        for r in data[judge_type]:
            scores = r.get('scores')
            if not scores:
                continue
            for q, v in scores.items():
                if not q.startswith(prefix):
                    continue
                ans     = v.get('answer', 'Unclear') \
                          if isinstance(v, dict) else v
                numeric = 1.0 if ans == 'Yes' \
                         else 0.0 if ans == 'No' \
                         else 0.5
                rows.append({
                    'model'        : model_name,
                    'structure_id' : r['structure_id'],
                    'turn'         : r['turn'],
                    'director'     : r['director'],
                    'question'     : q,
                    'answer'       : ans,
                    'numeric'      : numeric,
                    'judge_failed' : r['judge_failed'],
                })
    
    return pd.DataFrame(rows)


def report_all(master):
    """
    Print all reportable averages:
    1. Per-model per-question mean
    2. Per-model overall SG and MM mean
    3. Per-question across all models
    4. F3 divergence rate
    """
    sg_df = flatten_to_df(master, 'sg')
    mm_df = flatten_to_df(master, 'mm')
    
    # ── 1. Per-model per-question pivot ───────────────────────
    print(f"\n{'='*70}")
    print("PER-MODEL PER-QUESTION MEAN (SG)")
    print(f"{'='*70}")
    sg_pivot = sg_df.groupby(['model','question'])['numeric']\
                    .mean().round(3).unstack()
    sg_pivot['SG_mean'] = sg_pivot.mean(axis=1).round(3)
    print(sg_pivot.to_string())
    
    print(f"\n{'='*70}")
    print("PER-MODEL PER-QUESTION MEAN (MM)")
    print(f"{'='*70}")
    mm_pivot = mm_df.groupby(['model','question'])['numeric']\
                    .mean().round(3).unstack()
    mm_pivot['MM_mean'] = mm_pivot.mean(axis=1).round(3)
    print(mm_pivot.to_string())
    
    # ── 2. Per-model overall means ────────────────────────────
    print(f"\n{'='*70}")
    print("PER-MODEL OVERALL SG / MM / COMBINED MEAN")
    print(f"{'='*70}")
    sg_overall = sg_df.groupby('model')['numeric']\
                      .mean().rename('SG_mean').round(3)
    mm_overall = mm_df.groupby('model')['numeric']\
                      .mean().rename('MM_mean').round(3)
    overall    = pd.concat([sg_overall, mm_overall], axis=1)
    overall['combined'] = overall.mean(axis=1).round(3)
    print(overall.sort_values('combined', ascending=False).to_string())
    
    # ── 3. Per-question across all models ─────────────────────
    print(f"\n{'='*70}")
    print("PER-QUESTION ACROSS ALL MODELS")
    print(f"{'='*70}")
    sq_q = sg_df.groupby('question')['numeric']\
                .agg(mean='mean', std='std', n='count').round(3)
    mm_q = mm_df.groupby('question')['numeric']\
                .agg(mean='mean', std='std', n='count').round(3)
    print("SG:")
    print(sq_q.to_string())
    print("\nMM:")
    print(mm_q.to_string())
    
    # ── 4. F3 divergence ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("F3 DIVERGENCE RATE (SG4=Yes AND MM6=No)")
    print(f"{'='*70}")
    sg4 = sg_df[sg_df['question']=='SG4'][
        ['model','structure_id','turn','director','answer']
    ].rename(columns={'answer':'sg4'})
    mm6 = mm_df[mm_df['question']=='MM6'][
        ['model','structure_id','turn','director','answer']
    ].rename(columns={'answer':'mm6'})
    merged      = sg4.merge(mm6,
                            on=['model','structure_id','turn','director'])
    merged['f3'] = ((merged['sg4']=='Yes') & 
                    (merged['mm6']=='No'))
    f3_rate     = merged.groupby('model')['f3']\
                        .mean().round(3).sort_values(ascending=False)
    print(f3_rate.to_string())
    
    # ── 5. Save summary CSVs ──────────────────────────────────
    sg_pivot.to_csv("sg_per_model_per_question.csv")
    mm_pivot.to_csv("mm_per_model_per_question.csv")
    overall.to_csv("overall_sg_mm_scores.csv")
    merged[['model','f3']].groupby('model').mean()\
          .round(3).to_csv("f3_divergence.csv")
    
    print("\nCSVs saved.")
    return sg_df, mm_df, overall, merged

def report_across_runs(all_run_masters):
    """Average per-question scores across N runs with SEM."""
    
    sg_dfs = []
    mm_dfs = []
    
    for run_idx, master in enumerate(all_run_masters):
        sg = flatten_to_df(master, 'sg')
        mm = flatten_to_df(master, 'mm')
        sg['run'] = run_idx
        mm['run'] = run_idx
        sg_dfs.append(sg)
        mm_dfs.append(mm)
    
    sg_all = pd.concat(sg_dfs, ignore_index=True)
    mm_all = pd.concat(mm_dfs, ignore_index=True)
    
    def summarize(df, label):
        # per model per question: mean and SEM across runs
        run_means = df.groupby(
            ['run','model','question'])['numeric'].mean().reset_index()
        
        agg = run_means.groupby(['model','question'])['numeric'].agg(
            mean='mean',
            sem=lambda x: x.std() / np.sqrt(len(x)),
        ).round(4).reset_index()
        
        pivot_mean = agg.pivot(
            index='model', columns='question', values='mean').round(3)
        pivot_sem  = agg.pivot(
            index='model', columns='question', values='sem').round(3)
        
        # overall per model
        overall = run_means.groupby(['run','model'])['numeric']\
                           .mean().reset_index()
        overall_agg = overall.groupby('model')['numeric'].agg(
            mean='mean',
            sem=lambda x: x.std() / np.sqrt(len(x))
        ).round(3).sort_values('mean', ascending=False)
        
        print(f"\n{'='*70}")
        print(f"{label} — MEAN ± SEM ACROSS {len(all_run_masters)} RUNS")
        print(f"{'='*70}")
        
        # print mean(sem) per question per model
        combined = pivot_mean.copy().astype(str)
        for col in pivot_mean.columns:
            combined[col] = pivot_mean[col].astype(str) + \
                            ' ±' + pivot_sem[col].astype(str)
        print(combined.to_string())
        
        print(f"\nOverall {label} mean ± SEM:")
        print(overall_agg.to_string())
        
        # save
        pivot_mean.to_csv(f"{label.lower()}_mean_across_runs.csv")
        pivot_sem.to_csv(f"{label.lower()}_sem_across_runs.csv")
        overall_agg.to_csv(f"{label.lower()}_overall_across_runs.csv")
        
        return pivot_mean, pivot_sem, overall_agg
    
    sg_mean, sg_sem, sg_overall = summarize(sg_all, 'SG')
    mm_mean, mm_sem, mm_overall = summarize(mm_all, 'MM')
    
    # combined ranking
    print(f"\n{'='*70}")
    print("FINAL COMBINED RANKING (SG + MM mean ± SEM)")
    print(f"{'='*70}")
    combined = pd.DataFrame({
        'SG_mean' : sg_overall['mean'],
        'SG_sem'  : sg_overall['sem'],
        'MM_mean' : mm_overall['mean'],
        'MM_sem'  : mm_overall['sem'],
    })
    combined['combined_mean'] = combined[['SG_mean','MM_mean']]\
                                 .mean(axis=1).round(3)
    combined['combined_sem']  = combined[['SG_sem','MM_sem']]\
                                 .mean(axis=1).round(3)
    print(combined.sort_values('combined_mean', ascending=False).to_string())
    combined.to_csv("final_combined_ranking.csv")
    
    return sg_mean, sg_sem, mm_mean, mm_sem
    
if __name__ == "__main__":
    
    SANITY_MODE   = False
    SANITY_GAMES  = 2
    SANITY_TURNS  = 2
    N_RUNS        = 3
    BUILDER_MODEL = "gpt-4o-mini"
    JUDGE_MODEL   = "gpt-4o-mini"
    ORACLE_DIR_FRONTIER = "craft_oracle_upper_bound_no_tools_gpt-4o-mini_closedmodel_geminifix"
 
    MODEL_NAMES_FRONTIER = [
        "gpt-4o", "gpt-4o-mini", "gpt-4.1-mini",
        "claude-sonnet-4-6", "gemini-2.5-flash", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", 
    ]

     
    all_games = load_all_models(ORACLE_DIR_FRONTIER, MODEL_NAMES_FRONTIER, BUILDER_MODEL)
    
    # BASE_DIR      = Path("judge_results_bps_sanity" if SANITY_MODE
    #                      else "judge_results_bps_full_final_basemodelruns")
    BASE_DIR      = Path("judge_results_bps_sanity_frontier_rerun" if SANITY_MODE
                         else "judge_results_bps_full_final_frontier_modelruns_gemini_flash_3lite_run1all_2")
    CONV_WINDOW   = 3
    MAX_WORKERS   = 10

    # all_games = load_all_models(ORACLE_DIR, MODEL_NAMES, BUILDER_MODEL)

    # prepare games slice once — reused across all runs
    if SANITY_MODE:
        print(f"\nSANITY MODE: {SANITY_GAMES} games, "
              f"{SANITY_TURNS} turns, {N_RUNS} runs")
        all_games_run = {
            model: games[:SANITY_GAMES]
            for model, games in all_games.items()
        }
        for model, games in all_games_run.items():
            for game in games:
                game['_turns_orig'] = game['turns']
                game['turns']       = game['turns'][:SANITY_TURNS]
    else:
        all_games_run = all_games

    
    # ── N runs ────────────────────────────────────────────────
    all_run_masters = []

    for run_idx in range(1, N_RUNS + 1):
        print(f"\n{'#'*55}")
        print(f"  RUN {run_idx} / {N_RUNS}")
        print(f"{'#'*55}")

        run_dir = BASE_DIR / f"run_{run_idx}"

        master = run_judges_parallel(
            all_games_run,
            output_dir  = run_dir,
            judge_model = JUDGE_MODEL,
            conv_window = CONV_WINDOW,
            max_workers = MAX_WORKERS,
        )

        # save run pickle
        with open(run_dir / "master.pkl", "wb") as f:
            pickle.dump(master, f)

        all_run_masters.append(master)

        # per-run report
        print(f"\n--- Run {run_idx} report ---")
        sg_df, mm_df, overall, f3 = report_all(master)

    # restore turns
    if SANITY_MODE:
        for model, games in all_games_run.items():
            for game in games:
                game['turns'] = game.pop('_turns_orig')

    # ── Aggregate across runs ─────────────────────────────────
    print(f"\n{'#'*55}")
    print(f"  AGGREGATE ACROSS {N_RUNS} RUNS")
    print(f"{'#'*55}")

    report_across_runs(all_run_masters)
    
    if SANITY_MODE:
        print(f"\n{'='*55}")
        print(f"SANITY CHECK SUMMARY")
        print(f"{'='*55}")
        print(f"Models evaluated : {len(master)}")
        n_questions_sg = 7
        n_questions_mm = 8
        print(f"Expected SG rows : "
              f"{len(master) * SANITY_GAMES * SANITY_TURNS * 3 * n_questions_sg} "
              f"(models × games × turns × directors × questions, max)")
        print(f"Expected MM rows : "
          f"{len(master) * SANITY_GAMES * SANITY_TURNS * 3 * n_questions_mm} "
          f"(models × games × turns × directors × questions, max)")
        print(f"Expected SG rows : "
              f"{len(master) * SANITY_GAMES * SANITY_TURNS * 3} "
              f"(models × games × turns × directors, max)")
        print(f"Questions found  : "
              f"SG={sorted(sg_df['question'].unique())}  "
              f"MM={sorted(mm_df['question'].unique())}")
        failed_sg = sg_df['judge_failed'].sum() if 'judge_failed' in sg_df.columns else 'N/A'
        failed_mm = mm_df['judge_failed'].sum() if 'judge_failed' in mm_df.columns else 'N/A'
        print(f"Judge failures   : SG={failed_sg}  MM={failed_mm}")
    
    # reload saved runs
    all_run_masters = []
    for run_idx in range(1, N_RUNS + 1):
        with open(BASE_DIR / f"run_{run_idx}" / "master.pkl", "rb") as f:
            all_run_masters.append(pickle.load(f))
    report_across_runs(all_run_masters)   
     


    