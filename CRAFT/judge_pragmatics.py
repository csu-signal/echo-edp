from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import pickle
from tqdm import tqdm
# judge_prompts.py
import json
import ast
import shutil

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
 
from dotenv import load_dotenv
from tqdm import tqdm
# from judge_prompts import sg_judge_prompt, mm_judge_prompt
import time
import random

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY_3"))
# client = OpenAI()
JUDGE_MODEL = "gpt-4o-mini"



def ps_judge_prompt(
    director_messages,
    oracle_moves,
    board_state,
    builder_confirmation,
    condition,  # 'C1_followed' or 'C2_not_followed'
):
    oracle_str = "\n".join(
    "  {} {} at {} layer {}{}".format(
        m['action'].upper(),
        m['block'],
        m['position'],
        m['layer'],
        f" spanning to {m['span_to']}" if m.get('span_to') else ''
    )
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

def call_ps_judge(row):
    """Call PS judge for one turn. Returns (index, scores_dict or None)."""
    try:
        prompt = ps_judge_prompt(
            director_messages    = row['director_messages'],
            oracle_moves         = row['oracle_moves'],
            board_state          = row['board_state'],
            builder_confirmation = row['builder_confirmation'],
            condition            = row['condition'],
        )
        
        response = client.chat.completions.create(
            model       = JUDGE_MODEL,
            max_tokens  = 800,
            temperature = 0,
            messages    = [{"role": "user", "content": prompt}]
        )
        
        raw = response.choices[0].message.content.strip()
        
        # parse JSON
        clean = raw.replace("```json","").replace("```","").strip()
        parsed = json.loads(clean)
        
        scores = {}
        for q in ['PS1','PS2','PS3','PS4','PS5','PS6']:
            if q in parsed:
                ans = parsed[q].get('answer','Unclear')
                scores[q] = (1.0 if ans=='Yes' 
                             else 0.0 if ans=='No'
                             else 0.5 if ans=='Unclear'
                             else None)  # N/A
                scores[f'{q}_reason'] = parsed[q].get('reason','')
        
        return scores
    
    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_ps_judge(df_ps_inputs, output_dir="judge_results_ps",
                 max_workers=20, n_runs=3):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_run_results = []
    
    for run_idx in range(1, n_runs + 1):
        print(f"\n{'#'*55}")
        print(f"  PS JUDGE RUN {run_idx} / {n_runs}")
        print(f"{'#'*55}")
        
        run_dir = output_dir / f"run_{run_idx}"
        run_dir.mkdir(exist_ok=True)
        
        results = []
        rows    = df_ps_inputs.to_dict('records')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(call_ps_judge, row): i
                for i, row in enumerate(rows)
            }
            
            for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"Run {run_idx}"):
                idx    = futures[future]
                scores = future.result()
                row    = rows[idx]
                
                results.append({
                    'model':      row['model'],
                    'file':       row['file'],
                    'turn':       row['turn'],
                    'condition':  row['condition'],
                    'complexity': row['complexity'],
                    'failure_type': row['failure_type'],
                    'scores':     scores,
                    **({k: v for k, v in (scores or {}).items()
                        if not k.endswith('_reason')}),
                })
        
        # save run
        with open(run_dir / "ps_results.pkl", "wb") as f:
            pickle.dump(results, f)
        
        # save jsonl
        with open(run_dir / "ps_results.jsonl", "w") as f:
            for r in results:
                r_out = {k:v for k,v in r.items() if k != 'scores'}
                f.write(json.dumps(r_out) + "\n")
        
        all_run_results.append(results)
        
        # quick per-run report
        df_run = pd.DataFrame([
            {k:v for k,v in r.items() if k != 'scores'}
            for r in results
        ])
        
        print(f"\nRun {run_idx} summary:")
        for q in ['PS1','PS2','PS3','PS4','PS5','PS6']:
            if q in df_run.columns:
                print(f"  {q}: mean={df_run[q].mean():.3f}")
    
    # save master
    with open(output_dir / "ps_all_runs.pkl", "wb") as f:
        pickle.dump(all_run_results, f)
    
    print(f"\nAll runs saved to {output_dir}")
    return all_run_results

# failure taxonomy from existing logs
def classify_failure(turn):
    oracle    = turn.get("oracle_moves", [])
    attempted = turn.get("move_attempted", {})
    executed  = turn.get("move_executed", False)
    failed    = turn.get("failed_move", False)
    pd        = turn.get("progress_data", {})
    
    if not oracle or not attempted:
        return "no_oracle_or_attempt"
    
    a   = attempted.get("action")
    pos = attempted.get("position")
    lay = attempted.get("layer")
    blk = attempted.get("block")
    spn = attempted.get("span_to")
    
    # check game engine failure first
    if isinstance(pd, dict) and "error" in pd:
        error = pd["error"]
        if "layer" in error.lower():
            return "engine_layer_error"
        if "span" in error.lower():
            return "engine_span_error"
        return "engine_other_error"
    
    # check oracle match levels
    full = any(
        a==m["action"] and pos==m["position"] and
        lay==m["layer"] and blk==m["block"] and
        spn==m.get("span_to")
        for m in oracle)
    
    if full and executed and not failed:
        return "correct"
    
    loose = any(
        a==m["action"] and pos==m["position"] and lay==m["layer"]
        for m in oracle)
    
    strict = any(
        a==m["action"] and pos==m["position"] and
        lay==m["layer"] and blk==m["block"]
        for m in oracle)
    
    if not loose:
        return "wrong_position"      # position itself wrong
    
    if loose and not strict:
        return "wrong_block_color"   # right position, wrong block type
    
    if strict and not full:
        return "wrong_span"          # right block, wrong span
    
    if full and failed:
        return "engine_rejected"     # correct move but engine rejected
    
    return "other"

def clean_model_label_new(dirname: str) -> str:
    name   = dirname.split(",,")[0]
    parts  = name.split("_")
    model  = parts[0]
    builder = parts[1] if len(parts) > 1 else ""

    # base models
    model = model.replace("qwen-", "Qwen-")
    model = model.replace("mistral-", "Mistral-")
    model = model.replace("gemma-", "Gemma-")
    model = model.replace("llama-", "Llama-")
    model = model.replace("deepseek-v2-lite", "DeepSeek-Lite")

    # frontier models
    model = model.replace("claude-sonnet-4-6", "Claude-Sonnet-4.6")
    model = model.replace("gemini-2.5-flash", "Gemini-2.5-Flash")
    model = model.replace("gemini-3-flash-preview", "Gemini-3-Flash")
    model = model.replace("gemini-3.1-flash-lite-preview", "Gemini-3.1-Flash-Lite-Preview")
    model = model.replace("gpt-4.1-mini", "GPT-4.1-Mini")
    model = model.replace("gpt-4o-mini", "GPT-4o-Mini")
    model = model.replace("gpt-4o", "GPT-4o")

    builder = builder.replace("gpt-4o-mini", "GPT-4o-Mini")
    
    return f"{model}"  # drop builder suffix for cleaner labels

# run
# df_cond.to_csv("evaluation_conditional_c1_c2.csv")
# df_cond = pd.read_csv("evaluation_conditional_c1_c2.csv")
df_ps_inputs = pd.read_csv("df_ps_inputs_gemini_31.csv")
 
df_gemini31_only = df_ps_inputs[
    df_ps_inputs['model'] == 'Gemini-3.1-Flash-Lite-Preview'
].reset_index(drop=True).copy()

for col in ['director_messages', 'oracle_moves']:
    if isinstance(df_gemini31_only[col].iloc[0], str):
        df_gemini31_only[col] = df_gemini31_only[col].apply(ast.literal_eval)

# clear bad cache before re-running
shutil.rmtree("judge_results_ps_gemini31", ignore_errors=True)

ps_results_gemini31 = run_ps_judge(
    df_gemini31_only,
    output_dir  = "judge_results_ps_gemini31",
    max_workers = 15,
    n_runs      = 2
)
