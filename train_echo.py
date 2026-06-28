"""
ECHO training for the Clue Selector Game (CSG).

ECHO (Epistemic Credit for History-Conditioned Optimization):
  Per-turn cross-generation advantage normalization. Compares G parallel
  rollouts of the same secret at each turn depth, giving a belief-state-aware
  credit signal without explicit value estimation.

Usage:
    python train_echo.py                     # single seed (42)
    python train_echo.py --seeds 42 123 7    # multi-seed
    python train_echo.py --steps 500 --model Qwen/Qwen2.5-1.5B-Instruct
"""

import os
import re
import gc
import json
import math
import time
import random
import pickle
import datetime
import argparse
import numpy as np
from pathlib import Path

import torch
from dotenv import load_dotenv
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import GRPOConfig, GRPOTrainer
from trl.models import unwrap_model_for_generation
import wandb

from environment import (
    filter_candidates, is_prime, call_clue_holder,
    build_llm_prompt, parse_llm_response, ARM_PROPERTIES, sanitize_hint,
)
from advantages import compute_multiturn_advantages

load_dotenv()

os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1800'
os.environ['TORCH_NCCL_ENABLE_MONITORING']      = '0'
os.environ["HF_HUB_OFFLINE"]                    = "1"
os.environ["TRANSFORMERS_OFFLINE"]              = "1"

# ── game constants ────────────────────────────────────────────────────────────

N_ARMS    = 5
MAX_TURNS = 10

ADVANTAGE_MODE = "echo"   # ECHO uses cross-generation per-turn normalization

# ── reward function ───────────────────────────────────────────────────────────

def compute_clue_reward(
    completion_text,
    secret,
    candidates,
    arm_histories,
    turn,
    max_turns      = MAX_TURNS,
    n_arms         = N_ARMS,
    redundancy_penalty = -0.1,
    resolve_bonus      =  1.0,
    efficiency_weight  =  0.1,
    arm_match_bonus    =  0.05,
    diversity_bonus    =  0.05,
):
    """
    Compute reward for one turn.

    Returns:
        training_reward : scalar
        arm             : selected arm index
        question        : question text
        candidates_after: updated candidate list
        reward_info     : dict of all components for logging
    """
    candidates_before = len(candidates)
    arm, question     = parse_llm_response(completion_text, n_arms)

    reward_info = {
        "normalized": 0.0, "raw_eliminated": 0,
        "candidates_before": candidates_before, "candidates_after": candidates_before,
        "eliminated": 0, "arm_selected": arm,
        "arm_match_bonus": 0.0, "diversity_bonus": 0.0, "redundancy_penalty": 0.0,
        "resolved": False, "resolve_bonus": 0.0, "efficiency_bonus": 0.0,
        "turns_saved": 0, "training_reward": 0.0, "total_signal": 0.0,
        "turn": turn, "max_turns": max_turns, "turns_remaining": max_turns - turn,
    }

    already_asked_this_arm = {q.strip().lower() for q, _ in arm_histories[arm]}
    if question is None:
        reward_info["training_reward"] = redundancy_penalty
        return redundancy_penalty, arm if arm is not None else 0, "", candidates, reward_info
    if question.strip().lower() in already_asked_this_arm:
        reward_info["redundancy_penalty"] = redundancy_penalty
        reward_info["training_reward"]    = redundancy_penalty
        return redundancy_penalty, arm, question, candidates, reward_info

    all_asked = {q.strip().lower() for h in arm_histories for q, _ in h}
    if question.strip().lower() not in all_asked:
        reward_info["diversity_bonus"] = diversity_bonus

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
    arm_match = any(
        any(kw in question.lower() for kw in prop_keywords.get(prop, [prop]))
        for prop in ARM_PROPERTIES.get(arm, [])
    )
    if arm_match:
        reward_info["arm_match_bonus"] = arm_match_bonus

    hint = call_clue_holder(question, secret, candidates)
    hint = sanitize_hint(hint, secret)
    if hint is None:
        hint = "I cannot help with that."
    reward_info["hint"] = hint

    candidates_after  = filter_candidates(candidates, hint)
    eliminated        = candidates_before - len(candidates_after)
    normalized_reward = eliminated / candidates_before if candidates_before > 0 else 0.0

    reward_info.update({
        "raw_eliminated": eliminated, "eliminated": eliminated,
        "candidates_after": len(candidates_after), "normalized": normalized_reward,
    })

    resolved = len(candidates_after) == 1 and candidates_after[0] == secret
    reward_info["resolved"] = resolved
    if resolved:
        turns_saved = max_turns - turn
        reward_info.update({
            "turns_saved":      turns_saved,
            "resolve_bonus":    resolve_bonus,
            "efficiency_bonus": efficiency_weight * turns_saved,
        })

    training_reward = (
        normalized_reward
        + reward_info["arm_match_bonus"]
        + reward_info["diversity_bonus"]
        + (reward_info["resolve_bonus"] + reward_info["efficiency_bonus"] if resolved else 0.0)
    )
    reward_info["training_reward"] = training_reward

    return training_reward, arm, question, candidates_after, reward_info


class ClueRewardFunction:
    def __init__(self, n_arms=N_ARMS, max_turns=MAX_TURNS,
                 redundancy_penalty=-0.1, resolve_bonus=1.0,
                 efficiency_weight=0.1, arm_match_bonus=0.05, diversity_bonus=0.05):
        self.n_arms             = n_arms
        self.max_turns          = max_turns
        self.redundancy_penalty = redundancy_penalty
        self.resolve_bonus      = resolve_bonus
        self.efficiency_weight  = efficiency_weight
        self.arm_match_bonus    = arm_match_bonus
        self.diversity_bonus    = diversity_bonus
        self.__name__           = "clue_reward"
        self.last_reward_infos  = []

    def _truncate_to_json(self, text):
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text).strip()
        brace_count = 0
        for idx, ch in enumerate(text):
            if ch == '{':
                brace_count += 1
            elif ch == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[:idx + 1]
        return text

    def __call__(self, prompts, completions, **kwargs):
        secrets            = kwargs["secret"]
        turns              = kwargs["turn"]
        candidates_list    = [
            json.loads(c) if isinstance(c, str) else c
            for c in kwargs["candidates"]
        ]
        arm_histories_list = [
            [[tuple(pair) for pair in arm] for arm in json.loads(ah)]
            if isinstance(ah, str) else ah
            for ah in kwargs["arm_histories"]
        ]

        self.last_reward_infos = []
        rewards                = []

        for i, completion in enumerate(completions):
            text = completion[0].get("content", "") if isinstance(completion, list) else str(completion)
            text = self._truncate_to_json(text)

            reward, arm, question, _, reward_info = compute_clue_reward(
                completion_text    = text,
                secret             = secrets[i],
                candidates         = candidates_list[i],
                arm_histories      = arm_histories_list[i],
                turn               = turns[i],
                max_turns          = self.max_turns,
                n_arms             = self.n_arms,
                redundancy_penalty = self.redundancy_penalty,
                resolve_bonus      = self.resolve_bonus,
                efficiency_weight  = self.efficiency_weight,
                arm_match_bonus    = self.arm_match_bonus,
                diversity_bonus    = self.diversity_bonus,
            )
            reward_info.update({"completion": text, "arm": arm, "question": question})
            self.last_reward_infos.append(reward_info)
            rewards.append(reward)

        return rewards

# ── dataset ───────────────────────────────────────────────────────────────────

def build_multiturn_dataset(secrets, tokenizer):
    """Minimal dataset — each sample is a secret; episodes generated online."""
    samples = [{"prompt": f"Secret game instance {s}", "secret": int(s)} for s in secrets]
    return HFDataset.from_list(samples)

# ── trainer ───────────────────────────────────────────────────────────────────

class ECHOTrainer(GRPOTrainer):
    """
    GRPO trainer implementing ECHO advantage computation.

    Overrides _generate_and_score_completions to run fully online multi-turn
    episodes, then applies cross-generation per-turn advantage normalization.
    """

    CLUE_LOG_KEYS = [
        "normalized", "eliminated", "candidates_before", "candidates_after",
        "resolved", "arm_match_bonus", "diversity_bonus", "redundancy_penalty",
        "resolve_bonus", "efficiency_bonus", "turns_saved", "arm_selected",
        "question", "completion", "turn", "hint",
        "candidates_before_int", "all_arm_histories_snapshot",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._clue_logs = {k: [] for k in self.CLUE_LOG_KEYS}

        self._clue_reward_func = None
        for func in self.reward_funcs:
            if isinstance(func, ClueRewardFunction):
                self._clue_reward_func = func
                break
        if self._clue_reward_func is None:
            raise ValueError("ECHOTrainer requires a ClueRewardFunction in reward_funcs")

        self._last_clue_metrics     = {}
        self._buffered_reward_infos = None
        self._buffered_group_sizes  = None
        self._current_group_sizes   = None

    def _clear_clue_logs(self):
        for k in self.CLUE_LOG_KEYS:
            self._clue_logs[k] = []

    def _read_reward_infos(self):
        if not self._clue_reward_func.last_reward_infos:
            return
        for info in self._clue_reward_func.last_reward_infos:
            for k in self.CLUE_LOG_KEYS:
                if k in info:
                    self._clue_logs[k].append(info[k])

    def _safe_mean(self, vals):
        vals = [v for v in vals if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _safe_rate(self, bools):
        vals = [1.0 if bool(v) else 0.0 for v in bools if v is not None]
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def _prepare_inputs(self, generation_batch):
        mode = "train" if self.model.training else "eval"

        if mode == "train":
            generate_every     = self.args.steps_per_generation * self.num_iterations
            is_generation_step = (self._step % generate_every == 0
                                  or self._buffered_inputs is None)

            if is_generation_step:
                generation_batch      = self._generate_and_score_completions(generation_batch)
                self._buffered_inputs = self._split_by_episodes(
                    generation_batch,
                    self._current_group_sizes,
                    self.args.steps_per_generation,
                )

            step_in_gen = self._step % self.args.steps_per_generation
            if self._buffered_reward_infos is not None:
                self._compute_and_log_clue_metrics(
                    self._buffered_reward_infos[step_in_gen],
                    self._buffered_group_sizes[step_in_gen],
                    mode,
                )

            inputs     = self._buffered_inputs[step_in_gen]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(generation_batch)
            if self._buffered_reward_infos is not None:
                all_infos  = [info for sl in self._buffered_reward_infos for info in sl]
                all_gsizes = [gs   for sl in self._buffered_group_sizes  for gs   in sl]
                self._compute_and_log_clue_metrics(all_infos, all_gsizes, mode)

        return inputs

    def _split_by_episodes(self, output, group_sizes, steps_per_generation):
        ep_per_step    = len(group_sizes) // steps_per_generation
        turn_boundaries = [0]
        for gs in group_sizes:
            turn_boundaries.append(turn_boundaries[-1] + gs)

        slices = []
        for i in range(steps_per_generation):
            start_turn = turn_boundaries[i * ep_per_step]
            end_turn   = turn_boundaries[(i + 1) * ep_per_step]
            sl = {}
            for k, v in output.items():
                if k == "num_items_in_batch":
                    sl[k] = output["completion_mask"][start_turn:end_turn].sum()
                elif isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == output["advantages"].shape[0]:
                    sl[k] = v[start_turn:end_turn]
                else:
                    sl[k] = v
            slices.append(sl)
        return slices

    def _compute_and_log_clue_metrics(self, reward_infos, group_sizes, mode):
        self._clear_clue_logs()
        self._clue_reward_func.last_reward_infos = reward_infos
        self._read_reward_infos()

        prefix = "" if mode == "train" else "eval_"
        L, M   = self._clue_logs, self._metrics[mode]

        M.setdefault(f"{prefix}clue/normalized_reward",  []).append(self._safe_mean(L["normalized"]))
        M.setdefault(f"{prefix}clue/eliminated_mean",    []).append(self._safe_mean(L["eliminated"]))
        M.setdefault(f"{prefix}clue/candidates_before",  []).append(self._safe_mean(L["candidates_before"]))
        M.setdefault(f"{prefix}clue/candidates_after",   []).append(self._safe_mean(L["candidates_after"]))
        M.setdefault(f"{prefix}clue/resolve_rate",       []).append(self._safe_rate(L["resolved"]))
        M.setdefault(f"{prefix}clue/arm_match_rate",     []).append(
            self._safe_rate([v > 0 for v in L["arm_match_bonus"]]))
        M.setdefault(f"{prefix}clue/diversity_rate",     []).append(
            self._safe_rate([v > 0 for v in L["diversity_bonus"]]))
        M.setdefault(f"{prefix}clue/redundancy_rate",    []).append(
            self._safe_rate([v < 0 for v in L["redundancy_penalty"]]))
        M.setdefault(f"{prefix}clue/resolve_bonus_mean", []).append(self._safe_mean(L["resolve_bonus"]))
        M.setdefault(f"{prefix}clue/efficiency_bonus_mean",[]).append(self._safe_mean(L["efficiency_bonus"]))
        M.setdefault(f"{prefix}clue/turns_saved_mean",   []).append(self._safe_mean(L["turns_saved"]))
        M.setdefault(f"{prefix}clue/mean_turn",          []).append(self._safe_mean(L["turn"]))
        M.setdefault(f"{prefix}clue/mean_episode_length",[]).append(float(np.mean(group_sizes)))
        M.setdefault(f"{prefix}clue/episode_resolve_rate",[]).append(
            float(np.mean([
                1.0 if any(info.get("resolved", False) for info in reward_infos[
                    sum(group_sizes[:i]):sum(group_sizes[:i+1])
                ]) else 0.0
                for i in range(len(group_sizes))
            ]))
        )
        slice_rewards = [info.get("training_reward", 0.0) for info in reward_infos]
        M.setdefault("reward",     []).append(float(np.mean(slice_rewards)))
        M.setdefault("reward_std", []).append(float(np.std(slice_rewards)))

    def _generate_and_score_completions(self, inputs):
        device  = self.accelerator.device
        secrets = [int(inp["secret"]) for inp in inputs]

        all_prompt_ids     = []
        all_completion_ids = []
        all_rewards        = []
        all_reward_infos   = []
        group_sizes        = []

        for secret in secrets:
            for _ in range(self.num_generations):
                episode = self._run_single_episode(secret, device)
                all_prompt_ids.extend(episode["prompt_ids"])
                all_completion_ids.extend(episode["completion_ids"])
                all_rewards.extend(episode["rewards"])
                all_reward_infos.extend(episode["reward_infos"])
                group_sizes.append(len(episode["rewards"]))

        if self.state.global_step % self.args.save_steps == 0:
            accum_path = Path(self.args.output_dir) / "episodes_all_steps.pkl"
            accum = []
            if accum_path.exists():
                with open(accum_path, "rb") as f:
                    accum = pickle.load(f)
            accum.append({
                "step": self.state.global_step,
                "secrets": secrets,
                "group_sizes": group_sizes,
                "reward_infos": all_reward_infos,
            })
            with open(accum_path, "wb") as f:
                pickle.dump(accum, f)

        pad_id = self.processing_class.pad_token_id
        max_p  = max(t.shape[0] for t in all_prompt_ids)
        max_c  = max(t.shape[0] for t in all_completion_ids)

        prompt_ids_padded = torch.stack([
            torch.cat([torch.full((max_p - t.shape[0],), pad_id), t])
            for t in all_prompt_ids
        ]).to(device)
        completion_ids_padded = torch.stack([
            torch.cat([t, torch.full((max_c - t.shape[0],), pad_id)])
            for t in all_completion_ids
        ]).to(device)

        completion_mask       = (completion_ids_padded != pad_id).int()
        rewards_tensor        = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        prompt_completion_ids = torch.cat([prompt_ids_padded, completion_ids_padded], dim=1)
        attention_mask_full   = torch.cat([
            (prompt_ids_padded != pad_id).int(), completion_mask
        ], dim=1)

        ref_per_token_logps = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model, prompt_completion_ids, attention_mask_full,
                        max_c, batch_size=self.args.per_device_train_batch_size,
                        compute_entropy=False,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model, prompt_completion_ids, attention_mask_full,
                            max_c, batch_size=self.args.per_device_train_batch_size,
                            compute_entropy=False,
                        )

        with torch.no_grad():
            old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                self.model, prompt_completion_ids, attention_mask_full,
                max_c, batch_size=self.args.per_device_train_batch_size,
                compute_entropy=False,
            )

        B          = len(secrets)
        advantages = compute_multiturn_advantages(
            rewards     = rewards_tensor,
            group_sizes = group_sizes,
            B           = B,
            G           = self.num_generations,
            mode        = ADVANTAGE_MODE,
            device      = device,
        )

        self._current_group_sizes = group_sizes
        ep_per_step     = len(group_sizes) // self.args.steps_per_generation
        turn_boundaries = [0]
        for gs in group_sizes:
            turn_boundaries.append(turn_boundaries[-1] + gs)

        self._buffered_reward_infos = []
        self._buffered_group_sizes  = []
        for i in range(self.args.steps_per_generation):
            start_ep   = i * ep_per_step
            end_ep     = (i + 1) * ep_per_step
            start_turn = turn_boundaries[start_ep]
            end_turn   = turn_boundaries[end_ep]
            self._buffered_reward_infos.append(all_reward_infos[start_turn:end_turn])
            self._buffered_group_sizes.append(group_sizes[start_ep:end_ep])

        output = {
            "prompt_ids":          prompt_ids_padded,
            "prompt_mask":         (prompt_ids_padded != pad_id).int(),
            "completion_ids":      completion_ids_padded,
            "completion_mask":     completion_mask,
            "advantages":          advantages,
            "num_items_in_batch":  completion_mask.sum(),
            "old_per_token_logps": old_per_token_logps,
        }
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def _run_single_episode(self, secret, device):
        candidates        = list(range(1, 101))
        arm_histories     = [[] for _ in range(N_ARMS)]
        arm_question_sets = [set() for _ in range(N_ARMS)]
        prompt_ids_list   = []
        completion_ids_list = []
        rewards_list      = []
        reward_infos_list = []

        for t in range(MAX_TURNS):
            prompt_text = build_llm_prompt(
                candidates    = candidates,
                arm_histories = arm_histories,
                turn          = t,
                max_turns     = MAX_TURNS,
                n_arms        = N_ARMS,
            )
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Follow instructions exactly."},
                {"role": "user",   "content": prompt_text},
            ]
            formatted = self.processing_class.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self.processing_class.encode(
                formatted, return_tensors="pt",
                truncation=True, max_length=1024, add_special_tokens=False,
            ).to(device)

            with torch.no_grad():
                with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped:
                    output = unwrapped.generate(
                        input_ids,
                        max_new_tokens = 128,
                        do_sample      = True,
                        temperature    = self.args.temperature,
                        top_p          = 0.9,
                        pad_token_id   = self.processing_class.eos_token_id,
                    )

            new_tokens   = output[0][input_ids.shape[1]:]
            raw_response = self.processing_class.decode(new_tokens, skip_special_tokens=True).strip()

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

            reward, arm, question, candidates_after, reward_info = compute_clue_reward(
                completion_text    = completion,
                secret             = secret,
                candidates         = candidates,
                arm_histories      = arm_histories,
                turn               = t,
                redundancy_penalty = self._clue_reward_func.redundancy_penalty,
                resolve_bonus      = self._clue_reward_func.resolve_bonus,
                efficiency_weight  = self._clue_reward_func.efficiency_weight,
                arm_match_bonus    = self._clue_reward_func.arm_match_bonus,
                diversity_bonus    = self._clue_reward_func.diversity_bonus,
            )
            reward_info.update({
                "completion":                   completion,
                "arm":                          arm,
                "question":                     question,
                "all_arm_histories_snapshot":   [list(a) for a in arm_histories],
                "candidates_before_int":        len(candidates),
            })

            q_key = (question or "").strip().lower()
            if q_key not in arm_question_sets[arm]:
                arm_question_sets[arm].add(q_key)
                candidates = candidates_after
            hint = reward_info.get("hint", "I cannot help with that.")
            arm_histories[arm].append((question or "", hint))

            prompt_ids_list.append(input_ids[0].cpu())
            completion_ids_list.append(new_tokens.cpu())
            rewards_list.append(reward)
            reward_infos_list.append(reward_info)

            if len(candidates) == 1:
                break

        return {
            "prompt_ids":     prompt_ids_list,
            "completion_ids": completion_ids_list,
            "rewards":        rewards_list,
            "reward_infos":   reward_infos_list,
        }

    def log(self, logs, start_time=None):
        mode    = "train" if self.model.training else "eval"
        metrics = {k: sum(v) / len(v) for k, v in self._metrics[mode].items() if v}
        if mode == "eval":
            metrics = {f"eval_{k}": v for k, v in metrics.items()}
        logs = {**logs, **metrics}

        step   = self.state.global_step
        prefix = "" if mode == "train" else "eval_"

        def _g(key):
            return metrics.get(f"{prefix}clue/{key}", float("nan"))

        print(
            f"\n{'='*60}\n[{mode.upper()}] step={step}\n"
            f"  reward           = {logs.get('reward', float('nan')):.4f}\n"
            f"  normalized       = {_g('normalized_reward'):.4f}\n"
            f"  eliminated       = {_g('eliminated_mean'):.2f}\n"
            f"  resolve_rate     = {_g('resolve_rate'):.3f}\n"
            f"  episode_resolve  = {_g('episode_resolve_rate'):.3f}\n"
            f"  arm_match_rate   = {_g('arm_match_rate'):.3f}\n"
            f"  diversity_rate   = {_g('diversity_rate'):.3f}\n"
            f"  redundancy_rate  = {_g('redundancy_rate'):.3f}\n"
            f"  turns_saved      = {_g('turns_saved_mean'):.2f}\n"
            f"  episode_length   = {_g('mean_episode_length'):.2f}\n"
            f"{'='*60}"
        )

        if self.accelerator.is_main_process:
            if hasattr(self.args, "report_to") and "wandb" in self.args.report_to:
                try:
                    if wandb.run is not None:
                        wandb.log({
                            **{k: v for k, v in logs.items() if isinstance(v, (int, float))},
                            "train/step": step,
                        })
                except ImportError:
                    pass

        super().log(logs, start_time)

# ── config ────────────────────────────────────────────────────────────────────

class Fixed_GRPOConfig(GRPOConfig):
    """Fixes TRL's mutual-exclusion bug for generation_batch_size / steps_per_generation."""

    def __post_init__(self):
        import transformers
        transformers.TrainingArguments.__post_init__(self)

        num_processes = self.world_size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation  = self.gradient_accumulation_steps
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes * self.steps_per_generation
            )
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            global_batch_size = self.per_device_train_batch_size * num_processes
            if self.generation_batch_size % global_batch_size != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be "
                    f"divisible by global batch size ({global_batch_size})."
                )
            self.steps_per_generation = self.generation_batch_size // global_batch_size
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes * self.steps_per_generation
            )
        else:
            self.generation_batch_size = (
                self.per_device_train_batch_size * num_processes * self.steps_per_generation
            )

        if self.do_eval and self.eval_strategy != "no":
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError("Global eval batch size must be divisible by num_generations.")

        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError("generation_batch_size must be divisible by num_generations.")

        if self.num_generations < 2:
            raise ValueError("GRPO requires at least 2 generations per prompt.")


def build_grpo_config(run_name, output_dir, policy_model="Qwen/Qwen2.5-1.5B-Instruct",
                      per_device_train_batch_size=4, num_generations=4,
                      max_prompt_length=768, max_completion_length=64,
                      max_steps=500, learning_rate=5e-6, kl_coef=0.0,
                      logging_steps=2, save_steps=100, report_to="wandb",
                      seed=42):
    return Fixed_GRPOConfig(
        output_dir                   = output_dir,
        run_name                     = run_name,
        per_device_train_batch_size  = per_device_train_batch_size,
        num_generations              = num_generations,
        generation_batch_size        = per_device_train_batch_size * num_generations,
        steps_per_generation         = num_generations,
        max_prompt_length            = max_prompt_length,
        max_completion_length        = max_completion_length,
        max_steps                    = max_steps,
        learning_rate                = learning_rate,
        beta                         = kl_coef,
        logging_steps                = logging_steps,
        eval_strategy                = "no",
        save_strategy                = "steps",
        save_steps                   = save_steps,
        report_to                    = report_to,
        seed                         = seed,
        temperature                  = 1.0,
        top_p                        = 0.9,
        top_k                        = 50,
        max_grad_norm                = 0.5,
        bf16                         = True,
        remove_unused_columns        = False,
    )

# ── training entry point ──────────────────────────────────────────────────────

def train(
    seeds          = (42,),
    max_steps      = 350,
    save_every     = 100,
    policy_model   = "Qwen/Qwen2.5-1.5B-Instruct",
    report_to      = "none",
    log_dir        = None,
):
    log_dir = Path(log_dir or "clue_grpo_echo")
    log_dir.mkdir(parents=True, exist_ok=True)

    all_run_dirs = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        set_seed(seed)

        now      = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        run_name = f"echo_{policy_model.split('/')[-1]}_seed{seed}_{now}"
        run_dir  = log_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        tokenizer                = AutoTokenizer.from_pretrained(policy_model)
        tokenizer.pad_token      = tokenizer.eos_token
        tokenizer.padding_side   = "left"

        # training secrets: 1-100 minus held-out eval set
        np.random.seed(0)
        all_secrets = list(range(1, 101))
        eval_secrets= set(np.random.choice(all_secrets, size=5, replace=False))
        train_secrets = [s for s in all_secrets if s not in eval_secrets]

        train_dataset = build_multiturn_dataset(train_secrets, tokenizer)
        print(f"seed={seed} | train={len(train_dataset)} secrets | eval={sorted(eval_secrets)}")

        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            policy_model, torch_dtype=torch.bfloat16
        ).to("cuda:0")
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()

        clue_reward = ClueRewardFunction()
        grpo_config = build_grpo_config(
            run_name  = run_name,
            output_dir= str(run_dir),
            max_steps = max_steps,
            save_steps= save_every,
            report_to = report_to,
            seed      = seed,
        )

        trainer = ECHOTrainer(
            model            = model,
            reward_funcs     = [clue_reward],
            args             = grpo_config,
            train_dataset    = train_dataset,
            processing_class = tokenizer,
        )

        try:
            trainer.train()
            trainer.save_model(str(run_dir / "final_model"))
            trainer.state.save_to_json(str(run_dir / "trainer_state.json"))
            print(f"seed={seed} complete -> {run_dir}")
            all_run_dirs.append(str(run_dir))
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"seed={seed} failed — continuing")
        finally:
            del trainer, model, clue_reward
            torch.cuda.empty_cache()
            gc.collect()

    manifest = {
        "seeds":        [int(s) for s in seeds],
        "policy_model": policy_model,
        "max_steps":    max_steps,
        "advantage_mode": ADVANTAGE_MODE,
        "run_dirs":     all_run_dirs,
    }
    with open(log_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"all seeds complete | manifest -> {log_dir / 'manifest.json'}")
    return all_run_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECHO training for CSG")
    parser.add_argument("--seeds",    nargs="+", type=int, default=[42])
    parser.add_argument("--steps",    type=int,  default=350)
    parser.add_argument("--model",    type=str,  default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--log-dir",  type=str,  default="clue_grpo_echo")
    parser.add_argument("--report-to",type=str,  default="none")
    parser.add_argument("--save-every",type=int, default=100)
    args = parser.parse_args()

    train(
        seeds      = args.seeds,
        max_steps  = args.steps,
        policy_model = args.model,
        log_dir    = args.log_dir,
        report_to  = args.report_to,
        save_every = args.save_every,
    )
