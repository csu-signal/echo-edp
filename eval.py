"""
CSG evaluation runner.

Runs multi-turn episodes for trained ECHO models, base Qwen models, and
optionally frontier API models. Saves per-target JSON results and prints
aggregated tables.

Usage:
    # evaluate a trained ECHO checkpoint
    python eval.py --model-type trained --checkpoint /path/to/final_model

    # evaluate a base Qwen model
    python eval.py --model-type base --model-id Qwen/Qwen2.5-7B-Instruct

    # evaluate a frontier API model (requires OPENAI_API_KEY or CLAUDE_API_KEY)
    python eval.py --model-type frontier --model-name gpt-4o

    # ReAct baseline -- works for open (local HF) and closed (API) models alike
    python eval.py --model-type base     --model-id Qwen/Qwen2.5-1.5B-Instruct --variant react
    python eval.py --model-type frontier --model-name gpt-4o                   --variant react

    # print results table from existing eval_results/
    python eval.py --print-only
"""

import os
import re
import gc
import json
import random
import pickle
import argparse
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from openai import OpenAI

from environment import (
    filter_candidates, call_clue_holder, build_llm_prompt,
    parse_llm_response, ARM_PROPERTIES, PROPERTY_QUESTIONS, sanitize_hint,
)
from prompts import SYSTEM_PROMPT, REACT_SYSTEM_PROMPT, build_react_prompt

load_dotenv()

# ── constants ─────────────────────────────────────────────────────────────────

N_ARMS       = 5
MAX_TURNS    = 10
DEVICE       = "cuda:0"
OUT_DIR      = Path("eval_results/per_target")
SECRETS_PATH = Path("eval_results/canonical_secrets.json")
EVAL_SEEDS   = [42, 123, 7]
FRONTIER_RUNS= 3

# Generation budget for the default JSON-only variant.
LOCAL_MAX_NEW_TOKENS    = 128
LOCAL_MAX_LENGTH        = 1024
FRONTIER_MAX_TOKENS     = 256

# Generation budget for the ReAct variant. Larger because the ReAct prompt
# replays every past turn's full response, so context grows each turn and the
# 1024-token default would silently left-truncate the history away.
REACT_LOCAL_MAX_NEW_TOKENS = 256
REACT_LOCAL_MAX_LENGTH     = 2048
REACT_FRONTIER_MAX_TOKENS  = 512

# Replaces an unparseable ReAct response in the replayed history. Keeps the
# Thought/Action shape the agent was asked for, but states the action that was
# actually executed rather than the one the malformed text described.
REACT_PARSE_FAILURE_NOTE = (
    "Thought: (my previous response could not be parsed as an Action, "
    "so a default action was taken -- I must reply in the required format)"
)

# category label written into each result file, per model type
REACT_CATEGORIES = {
    "trained":  "trained_react_proper",
    "base":     "base_react_proper",
    "frontier": "frontier_react",
}

OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(SECRETS_PATH) as f:
    canonical = json.load(f)
ALL_SECRETS = canonical["all"]
STRATA      = ["favorable", "hard_in_dist", "ood_range", "structural"]

METRICS = [
    ("episode_resolve_rate", "Resolve%",  True),
    ("efficiency",           "Efficiency",False),
    ("mean_total_reward",    "Reward",    False),
    ("mean_eliminated",      "Elim",      False),
    ("mean_final_cands",     "FinalCands",False),
    ("arm_match_rate",       "ArmMatch",  True),
    ("redundancy_rate",      "Redundancy",True),
    ("diversity_rate",       "Diversity", True),
    ("mean_turns_taken",     "Turns",     False),
    ("parse_failure_rate",   "ParseFail", True),   # ReAct only; N/A elsewhere
]

# ── episode runner ────────────────────────────────────────────────────────────

def run_episode(generate_fn, secret, n_arms=N_ARMS, max_turns=MAX_TURNS):
    """
    Single episode rollout. generate_fn(prompt_text) -> str is the only
    thing that differs across trained / base / frontier models.
    """
    candidates        = list(range(1, max(101, secret + 1)))
    arm_histories     = [[] for _ in range(n_arms)]
    arm_question_sets = [set() for _ in range(n_arms)]
    episode_rewards   = []
    episode_log       = []

    prop_keywords = {
        "parity":       ["odd", "even", "parity"],
        "prime":        ["prime"],
        "div3":         ["divisible by 3", "multiple of 3"],
        "div7":         ["divisible by 7", "multiple of 7"],
        "div5":         ["divisible by 5", "multiple of 5"],
        "gt50":         ["greater than 50", "more than 50"],
        "square":       ["perfect square", "square number"],
        "digitsum":     ["sum of", "digit sum"],
        "lastdigit":    ["last digit", "units digit", "ones digit"],
        "between25_75": ["between 25", "between25"],
    }

    for t in range(max_turns):
        candidates_before = len(candidates)
        prompt_text       = build_llm_prompt(
            candidates=candidates, arm_histories=arm_histories,
            turn=t, max_turns=max_turns, n_arms=n_arms,
        )

        raw_response = generate_fn(prompt_text)

        completion = re.sub(r"```json\s*", "", raw_response)
        completion = re.sub(r"```\s*", "", completion).strip()
        brace_count = 0
        for idx, ch in enumerate(completion):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    completion = completion[:idx + 1]
                    break

        arm, question = parse_llm_response(completion, n_arms)
        q_key         = question.strip().lower()

        if q_key in arm_question_sets[arm]:
            hint            = "You already asked this. No new information."
            eliminated      = 0
            reward          = -0.1
            arm_match       = False
            diversity       = False
            redundant       = True
            candidates_after= candidates_before
        else:
            hint = call_clue_holder(question, secret, candidates)
            if hint is None:
                hint = "I cannot help with that."
            hint = sanitize_hint(hint, secret)

            arm_question_sets[arm].add(q_key)
            candidates_new   = filter_candidates(candidates, hint)
            eliminated       = candidates_before - len(candidates_new)
            candidates       = candidates_new
            candidates_after = len(candidates_new)
            reward           = eliminated / candidates_before if candidates_before > 0 else 0.0

            arm_match = any(
                any(kw in question.lower() for kw in prop_keywords.get(prop, [prop]))
                for prop in ARM_PROPERTIES.get(arm, [])
            )
            all_asked = {q.strip().lower() for h in arm_histories for q, _ in h}
            diversity = q_key not in all_asked
            redundant = False
            arm_histories[arm].append((question, hint))

        resolved     = len(candidates) == 1 and candidates[0] == secret
        turns_saved  = max_turns - t if resolved else 0
        eff_bonus    = 0.1 * turns_saved if resolved else 0.0
        resolve_bonus= 1.0 if resolved else 0.0

        training_reward = reward
        if not redundant:
            if arm_match:  training_reward += 0.05
            if diversity:  training_reward += 0.05
            if resolved:   training_reward += resolve_bonus + eff_bonus

        episode_log.append({
            "turn": t, "secret": secret,
            "raw_response": raw_response, "completion": completion,
            "arm": arm, "question": question, "hint": hint,
            "candidates_before": candidates_before,
            "candidates_after": candidates_after,
            "eliminated": eliminated,
            "normalized_reward": reward,
            "arm_match": arm_match, "diversity": diversity, "redundant": redundant,
            "resolved": resolved, "resolve_bonus": resolve_bonus,
            "efficiency_bonus": eff_bonus, "turns_saved": turns_saved,
            "training_reward": training_reward,
        })
        episode_rewards.append(training_reward)

        if resolved:
            break

    resolved_final   = any(e["resolved"] for e in episode_log)
    turns_taken      = len(episode_log)
    total_reward     = sum(episode_rewards)

    return {
        "secret":           secret,
        "resolved":         resolved_final,
        "turns_taken":      turns_taken,
        "resolve_turn":     next((e["turn"] for e in episode_log if e["resolved"]), None),
        "final_candidates": len(candidates),
        "total_reward":     total_reward,
        "efficiency":       total_reward / max(turns_taken, 1),
        "mean_eliminated":  np.mean([e["eliminated"]    for e in episode_log]),
        "mean_cands_before":np.mean([e["candidates_before"] for e in episode_log]),
        "mean_cands_after": np.mean([e["candidates_after"]  for e in episode_log]),
        "redundancy_rate":  np.mean([e["redundant"]     for e in episode_log]),
        "diversity_rate":   np.mean([e["diversity"]     for e in episode_log]),
        "arm_match_rate":   np.mean([e["arm_match"]     for e in episode_log]),
        "normalized_reward":total_reward / max(turns_taken, 1),
        "log":              episode_log,
    }


def try_parse_react_action(response_text, n_arms):
    """
    Parse a ReAct response -> (arm, question), or None if no valid action
    could be extracted.

    Deliberately independent of environment.parse_llm_response, which is
    shared with training and must keep its current signature. The extraction
    stages below mirror it exactly; the only difference is that failure is
    reported as None instead of being absorbed into a silent random fallback.
    If the stages in environment.py are ever changed, change them here too.
    """
    if not response_text:
        return None

    # stage 1 -- whole response is JSON
    try:
        parsed = json.loads(response_text)
        return int(parsed["arm"]) % n_arms, parsed["question"]
    except Exception:
        pass

    # stage 2 -- strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        return int(parsed["arm"]) % n_arms, parsed["question"]
    except Exception:
        pass

    # stage 3 -- LAST {...} carrying both keys (ReAct puts prose first)
    json_matches = re.findall(r'\{[^{}]*"arm"[^{}]*"question"[^{}]*\}', response_text)
    if json_matches:
        try:
            parsed = json.loads(json_matches[-1])
            return int(parsed["arm"]) % n_arms, parsed["question"]
        except Exception:
            pass

    # stage 4 -- scan lines in reverse for a bare JSON object
    for line in reversed(response_text.strip().split("\n")):
        line = line.strip().rstrip("`").strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if "arm" in parsed and "question" in parsed:
                    return int(parsed["arm"]) % n_arms, parsed["question"]
            except Exception:
                continue

    return None


def run_episode_react(generate_fn, secret, n_arms=N_ARMS, max_turns=MAX_TURNS):
    """
    Single ReAct episode rollout.
    """
    candidates        = list(range(1, max(101, secret + 1)))
    arm_histories     = [[] for _ in range(n_arms)]
    arm_question_sets = [set() for _ in range(n_arms)]
    episode_rewards   = []
    episode_log       = []
    react_history     = []   # (thought_action_text, observation) per turn

    prop_keywords = {
        "parity":       ["odd", "even", "parity"],
        "prime":        ["prime"],
        "div3":         ["divisible by 3", "multiple of 3"],
        "div7":         ["divisible by 7", "multiple of 7"],
        "div5":         ["divisible by 5", "multiple of 5"],
        "gt50":         ["greater than 50", "more than 50"],
        "square":       ["perfect square", "square number"],
        "digitsum":     ["sum of", "digit sum"],
        "lastdigit":    ["last digit", "units digit", "ones digit"],
        "between25_75": ["between 25", "between25"],
    }

    for t in range(max_turns):
        candidates_before = len(candidates)
        prompt_text       = build_react_prompt(
            candidates=candidates, react_history=react_history,
            turn=t, max_turns=max_turns, n_arms=n_arms,
        )

        raw_response = generate_fn(prompt_text)

        # no brace-trimming here -- see docstring
        parsed       = try_parse_react_action(raw_response, n_arms)
        parse_failed = parsed is None
        if parse_failed:
            # same fallback environment.parse_llm_response would have applied
            arm, question = random.randint(0, n_arms - 1), PROPERTY_QUESTIONS["parity"]
        else:
            arm, question = parsed
        q_key = question.strip().lower()

        if q_key in arm_question_sets[arm]:
            hint            = "You already asked this. No new information."
            eliminated      = 0
            reward          = -0.1
            arm_match       = False
            diversity       = False
            redundant       = True
            candidates_after= candidates_before
        else:
            hint = call_clue_holder(question, secret, candidates)
            if hint is None:
                hint = "I cannot help with that."
            hint = sanitize_hint(hint, secret)

            arm_question_sets[arm].add(q_key)
            candidates_new   = filter_candidates(candidates, hint)
            eliminated       = candidates_before - len(candidates_new)
            candidates       = candidates_new
            candidates_after = len(candidates_new)
            reward           = eliminated / candidates_before if candidates_before > 0 else 0.0

            arm_match = any(
                any(kw in question.lower() for kw in prop_keywords.get(prop, [prop]))
                for prop in ARM_PROPERTIES.get(arm, [])
            )
            all_asked = {q.strip().lower() for h in arm_histories for q, _ in h}
            diversity = q_key not in all_asked
            redundant = False
            arm_histories[arm].append((question, hint))

        # recorded on both branches: a redundant turn still becomes an
        # Observation the agent sees next turn.
        #
        # On a parse failure the raw text describes an action that was never
        # executed, while the Observation answers the fallback question.
        # Replaying it verbatim states a false premise that every later turn
        # then reasons from. Replay what actually ran instead, and say so, so
        # the model can correct its formatting.
        if parse_failed:
            replay = (f"{REACT_PARSE_FAILURE_NOTE}\n"
                      f'Action: {{"arm": {arm}, "question": "{question}"}}')
        else:
            replay = raw_response
        react_history.append((replay, hint))

        resolved     = len(candidates) == 1 and candidates[0] == secret
        turns_saved  = max_turns - t if resolved else 0
        eff_bonus    = 0.1 * turns_saved if resolved else 0.0
        resolve_bonus= 1.0 if resolved else 0.0

        training_reward = reward
        if not redundant:
            if arm_match:  training_reward += 0.05
            if diversity:  training_reward += 0.05
            if resolved:   training_reward += resolve_bonus + eff_bonus

        episode_log.append({
            "turn": t, "secret": secret,
            "raw_response": raw_response, "completion": raw_response,
            "arm": arm, "question": question, "hint": hint,
            "candidates_before": candidates_before,
            "candidates_after": candidates_after,
            "eliminated": eliminated,
            "normalized_reward": reward,
            "arm_match": arm_match, "diversity": diversity, "redundant": redundant,
            "parse_failed": parse_failed,
            "resolved": resolved, "resolve_bonus": resolve_bonus,
            "efficiency_bonus": eff_bonus, "turns_saved": turns_saved,
            "training_reward": training_reward,
        })
        episode_rewards.append(training_reward)

        if resolved:
            break

    resolved_final   = any(e["resolved"] for e in episode_log)
    turns_taken      = len(episode_log)
    total_reward     = sum(episode_rewards)

    return {
        "secret":           secret,
        "resolved":         resolved_final,
        "turns_taken":      turns_taken,
        "resolve_turn":     next((e["turn"] for e in episode_log if e["resolved"]), None),
        "final_candidates": len(candidates),
        "total_reward":     total_reward,
        "efficiency":       total_reward / max(turns_taken, 1),
        "mean_eliminated":  np.mean([e["eliminated"]    for e in episode_log]),
        "mean_cands_before":np.mean([e["candidates_before"] for e in episode_log]),
        "mean_cands_after": np.mean([e["candidates_after"]  for e in episode_log]),
        "redundancy_rate":  np.mean([e["redundant"]     for e in episode_log]),
        "diversity_rate":   np.mean([e["diversity"]     for e in episode_log]),
        "arm_match_rate":   np.mean([e["arm_match"]     for e in episode_log]),
        "parse_failure_rate":np.mean([e["parse_failed"] for e in episode_log]),
        "normalized_reward":total_reward / max(turns_taken, 1),
        "log":              episode_log,
    }


def run_all_episodes(generate_fn, secrets, desc, max_workers=1, runner=run_episode):
    results = []
    if max_workers == 1:
        for secret in tqdm(secrets, desc=desc):
            results.append(runner(generate_fn, secret))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(runner, generate_fn, s): s for s in secrets}
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"[skip] {e}")
    return results

# ── aggregate ─────────────────────────────────────────────────────────────────

def aggregate(episodes):
    return {
        "n_episodes":           len(episodes),
        "episode_resolve_rate": float(np.mean([e["resolved"]          for e in episodes])),
        "mean_turns_taken":     float(np.mean([e["turns_taken"]       for e in episodes])),
        "mean_total_reward":    float(np.mean([e["total_reward"]      for e in episodes])),
        "efficiency":           float(np.mean([e["efficiency"]        for e in episodes])),
        "mean_final_cands":     float(np.mean([e["final_candidates"]  for e in episodes])),
        "mean_eliminated":      float(np.mean([e["mean_eliminated"]   for e in episodes])),
        "mean_cands_before":    float(np.mean([e["mean_cands_before"] for e in episodes])),
        "mean_cands_after":     float(np.mean([e["mean_cands_after"]  for e in episodes])),
        "redundancy_rate":      float(np.mean([e["redundancy_rate"]   for e in episodes])),
        "diversity_rate":       float(np.mean([e["diversity_rate"]    for e in episodes])),
        "arm_match_rate":       float(np.mean([e["arm_match_rate"]    for e in episodes])),
        "normalized_reward":    float(np.mean([e["normalized_reward"] for e in episodes])),
        # ReAct only -- run_episode does not track it, so stays absent there
        **({"parse_failure_rate":
            float(np.mean([e["parse_failure_rate"] for e in episodes]))}
           if episodes and "parse_failure_rate" in episodes[0] else {}),
        "stratum_resolve": {
            s: float(np.mean([e["resolved"] for e in episodes if e["secret"] in canonical[s]]))
            if any(e["secret"] in canonical[s] for e in episodes) else None
            for s in STRATA
        },
    }

# ── result I/O ────────────────────────────────────────────────────────────────

def target_path(target_id, run_idx):
    return OUT_DIR / f"{target_id}__run{run_idx}.json"

def already_done(target_id, run_idx):
    return target_path(target_id, run_idx).exists()

def save_result(target_id, run_idx, result):
    p       = target_path(target_id, run_idx)
    pkl_path= p.with_suffix(".pkl")
    episodes= [ep["log"] for ep in result["episodes"]]
    with open(pkl_path, "wb") as f:
        pickle.dump(episodes, f)
    result_json = {k: v for k, v in result.items() if k != "episodes"}
    result_json["episodes_pkl"] = str(pkl_path)
    with open(p, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"  saved -> {p.name}")

# ── model generators ──────────────────────────────────────────────────────────

def make_local_generator(model, tokenizer, device, temperature=1.0,
                         system_prompt=SYSTEM_PROMPT,
                         max_new_tokens=LOCAL_MAX_NEW_TOKENS,
                         max_length=LOCAL_MAX_LENGTH):
    def generate_fn(prompt_text):
        messages  = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt_text},
        ]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(
            formatted, return_tensors="pt",
            truncation=True, max_length=max_length, add_special_tokens=False,
        ).to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens = max_new_tokens,
                do_sample      = True,
                temperature    = temperature,
                top_p          = 0.9,
                pad_token_id   = tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    return generate_fn


def make_frontier_generator(model_name, temperature=1.0,
                            max_tokens=FRONTIER_MAX_TOKENS,
                            system_prompt=SYSTEM_PROMPT):
    """
    Factory for frontier API models (OpenAI and Anthropic).

    Set OPENAI_API_KEY for GPT models, CLAUDE_API_KEY for Claude models.
    """
    def generate_fn(prompt_text):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt_text},
        ]
        if model_name.startswith("claude"):
            import anthropic
            client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
            resp   = client.messages.create(
                model      = model_name,
                max_tokens = max_tokens,
                system     = messages[0]["content"],
                messages   = [{"role": "user", "content": messages[1]["content"]}],
            )
            return resp.content[0].text.strip()
        else:
            client    = OpenAI()
            is_o_model= model_name.startswith("o")
            kwargs    = dict(
                model                 = model_name,
                messages              = messages,
                max_completion_tokens = 1024 if is_o_model else max_tokens,
            )
            if not is_o_model:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
    return generate_fn


def load_trained_model(checkpoint_path, policy_model, device):
    base  = AutoModelForCausalLM.from_pretrained(policy_model, torch_dtype=torch.bfloat16).to(device)
    model = PeftModel.from_pretrained(base, checkpoint_path, torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()
    model.eval()
    return model


def load_base_model(model_id, device):
    model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer

# ── eval orchestrator ─────────────────────────────────────────────────────────

def run_eval(args):
    POLICY_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    # ── variant wiring ────────────────────────────────────────────────────
    is_react   = args.variant == "react"
    suffix     = "__react_proper" if is_react else ""
    runner     = run_episode_react if is_react else run_episode
    category   = (REACT_CATEGORIES[args.model_type] if is_react
                  else args.model_type)
    local_kw   = dict(
        system_prompt  = REACT_SYSTEM_PROMPT,
        max_new_tokens = REACT_LOCAL_MAX_NEW_TOKENS,
        max_length     = REACT_LOCAL_MAX_LENGTH,
    ) if is_react else {}
    frontier_kw= dict(
        system_prompt = REACT_SYSTEM_PROMPT,
        max_tokens    = REACT_FRONTIER_MAX_TOKENS,
    ) if is_react else {}

    if args.model_type == "trained":
        tokenizer              = AutoTokenizer.from_pretrained(POLICY_MODEL)
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.padding_side = "left"

        for eval_seed in EVAL_SEEDS:
            target_id = f"echo__{Path(args.checkpoint).parent.name}{suffix}"
            run_idx   = eval_seed
            if already_done(target_id, run_idx):
                print(f"[skip] {target_id} run{run_idx}")
                continue

            torch.manual_seed(eval_seed)
            np.random.seed(eval_seed)

            model  = load_trained_model(args.checkpoint, POLICY_MODEL, DEVICE)
            gen    = make_local_generator(model, tokenizer, DEVICE, **local_kw)
            eps    = run_all_episodes(gen, ALL_SECRETS, desc=f"echo seed{eval_seed}",
                                      runner=runner)
            result = {"target_id": target_id, "run_idx": run_idx,
                      "category": category, "checkpoint": args.checkpoint,
                      **aggregate(eps), "episodes": eps}
            save_result(target_id, run_idx, result)
            del model
            torch.cuda.empty_cache()
            gc.collect()

    elif args.model_type == "base":
        model_id = args.model_id
        short    = model_id.split("/")[-1].lower()
        for eval_seed in EVAL_SEEDS:
            target_id = f"base__{short}{suffix}"
            run_idx   = eval_seed
            if already_done(target_id, run_idx):
                print(f"[skip] {target_id} run{run_idx}")
                continue

            torch.manual_seed(eval_seed)
            np.random.seed(eval_seed)
            model, tok = load_base_model(model_id, DEVICE)
            gen        = make_local_generator(model, tok, DEVICE, **local_kw)
            eps        = run_all_episodes(gen, ALL_SECRETS, desc=f"{short} seed{eval_seed}",
                                          runner=runner)
            result     = {"target_id": target_id, "run_idx": run_idx,
                          "category": category, "model_id": model_id,
                          **aggregate(eps), "episodes": eps}
            save_result(target_id, run_idx, result)
            del model
            torch.cuda.empty_cache()
            gc.collect()

    elif args.model_type == "frontier":
        model_name = args.model_name
        target_id  = f"frontier__{model_name}{suffix}"
        for run_idx in range(FRONTIER_RUNS):
            if already_done(target_id, run_idx):
                print(f"[skip] {target_id} run{run_idx}")
                continue
            gen    = make_frontier_generator(model_name, **frontier_kw)
            eps    = run_all_episodes(gen, ALL_SECRETS,
                                      desc=f"{model_name} run{run_idx}",
                                      max_workers=8, runner=runner)
            result = {"target_id": target_id, "run_idx": run_idx,
                      "category": category, "model_name": model_name,
                      **aggregate(eps), "episodes": eps}
            save_result(target_id, run_idx, result)

    print("\neval complete")

# ── result display ────────────────────────────────────────────────────────────

def load_all_results():
    grouped = defaultdict(list)
    for p in sorted(OUT_DIR.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        grouped[r["target_id"]].append(r)
    return grouped


def mean_sem(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = float(np.mean(vals))
    s = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, s


VARIANT_LABELS = {"cot": "CoT", "react": "ReAct", "react_proper": "ReAct"}


def display_name(target_id):
    parts = target_id.split("__")
    if parts[0] == "frontier":
        name = f"[F] {parts[1]}"
    elif parts[0] == "base":
        size = parts[1].split("-")[1].upper()
        name = f"[B] Qwen-{size}"
    else:
        name = f"[T] {parts[1]}"
    # keep prompting variants as their own rows -- without this,
    # base__qwen2.5-1.5b-instruct__react_proper and the plain
    # base__qwen2.5-1.5b-instruct collapse to the same key and get averaged
    # together by print_main_table / print_stratum_table / save_aggregated.
    if len(parts) > 2:
        name += f" ({VARIANT_LABELS.get(parts[2], parts[2])})"
    return name


def print_main_table(grouped):
    order      = {"[F]": 0, "[B]": 1, "[T]": 2}
    by_display = defaultdict(list)
    for t, runs in grouped.items():
        for r in runs:
            by_display[display_name(t)].append(r)

    col_w = 13
    print(f"\n{'='*95}\nOVERALL RESULTS  (mean ± SEM)\n{'='*95}")
    print(f"{'Model':<30}", end="")
    for _, label, _ in METRICS:
        print(f"  {label:>{col_w}}", end="")
    print()
    print("-" * (30 + (col_w + 2) * len(METRICS)))

    for name, runs in sorted(by_display.items(), key=lambda x: (order.get(x[0][:3], 9), x[0])):
        print(f"{name:<30}", end="")
        for key, _, is_pct in METRICS:
            vals = [r.get(key) for r in runs if r.get(key) is not None]
            m, s = mean_sem(vals)
            if m is None:
                print(f"  {'N/A':>{col_w}}", end="")
            elif is_pct:
                print(f"  {m:>{col_w-4}.1%}±{s:.2%}", end="")
            else:
                print(f"  {m:>{col_w-4}.3f}±{s:.3f}", end="")
        print()


def print_stratum_table(grouped):
    order      = {"[F]": 0, "[B]": 1, "[T]": 2}
    by_display = defaultdict(list)
    for t, runs in grouped.items():
        for r in runs:
            by_display[display_name(t)].append(r)

    col_w = 16
    print(f"\n{'='*80}\nRESOLVE RATE BY STRATUM  (mean ± SEM)\n{'='*80}")
    print(f"{'Model':<30}", end="")
    for s in STRATA:
        print(f"  {s:>{col_w}}", end="")
    print(f"  {'overall':>{col_w}}")
    print("-" * (30 + (col_w + 2) * (len(STRATA) + 1)))

    for name, runs in sorted(by_display.items(), key=lambda x: (order.get(x[0][:3], 9), x[0])):
        print(f"{name:<30}", end="")
        for stratum in STRATA:
            vals = [r.get("stratum_resolve", {}).get(stratum) for r in runs]
            m, s = mean_sem([v for v in vals if v is not None])
            print(f"  {m:>{col_w-5}.1%}±{s:.2%}" if m is not None else f"  {'N/A':>{col_w}}", end="")
        vals = [r.get("episode_resolve_rate") for r in runs]
        m, s = mean_sem([v for v in vals if v is not None])
        print(f"  {m:>{col_w-5}.1%}±{s:.2%}" if m is not None else f"  {'N/A':>{col_w}}")


def save_aggregated(grouped):
    out        = {}
    by_display = defaultdict(list)
    for t, runs in grouped.items():
        for r in runs:
            by_display[display_name(t)].append(r)

    for name, runs in by_display.items():
        agg = {}
        for key, _, _ in METRICS:
            vals   = [r.get(key) for r in runs if r.get(key) is not None]
            m, s   = mean_sem(vals)
            agg[key] = {"mean": m, "sem": s, "n": len(vals)}
        agg["stratum_resolve"] = {}
        for stratum in STRATA:
            vals   = [r.get("stratum_resolve", {}).get(stratum) for r in runs]
            m, s   = mean_sem([v for v in vals if v is not None])
            agg["stratum_resolve"][stratum] = {"mean": m, "sem": s}
        out[name] = agg

    p = Path("eval_results/aggregated_results.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {p}")

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSG evaluation runner")
    parser.add_argument("--model-type",  choices=["trained", "base", "frontier"],
                        help="Type of model to evaluate")
    parser.add_argument("--checkpoint",  type=str, help="Path to trained LoRA final_model dir")
    parser.add_argument("--model-id",    type=str, help="HuggingFace model ID for base eval")
    parser.add_argument("--model-name",  type=str, help="API model name for frontier eval (e.g. gpt-4o)")
    parser.add_argument("--variant",     choices=["json", "react"], default="json",
                        help="Prompting variant. 'json' (default) asks for a bare JSON "
                             "action each turn from a stateless prompt. 'react' runs the "
                             "ReAct baseline: Thought/Action/Observation history replayed "
                             "every turn, saved under a __react_proper target id.")
    parser.add_argument("--print-only",  action="store_true",
                        help="Skip eval; just print tables from existing results")
    args = parser.parse_args()

    if args.print_only:
        grouped = load_all_results()
        print(f"loaded {sum(len(v) for v in grouped.values())} result files")
        print_main_table(grouped)
        print_stratum_table(grouped)
        save_aggregated(grouped)
    else:
        if args.model_type is None:
            parser.error("--model-type required unless --print-only")
        run_eval(args)
        grouped = load_all_results()
        print_main_table(grouped)
        print_stratum_table(grouped)
        save_aggregated(grouped)
