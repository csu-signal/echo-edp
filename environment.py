"""
Clue Selector Game (CSG) — environment core.

Covers:
  - GPT-4o-mini oracle (call_clue_holder)
  - Deterministic oracle fallback (simulate_hint)
  - Belief update operator (filter_candidates)
  - Arm property definitions (ARM_PROPERTIES, PROPERTY_QUESTIONS)
  - Prompt builder and response parser
"""

import re
import os
import json
import time
import random

from openai import OpenAI
from dotenv import load_dotenv

from prompts import ORACLE_SYSTEM_PROMPT, ORACLE_PROMPT_TEMPLATE, build_selector_prompt

load_dotenv()
client    = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CLUE_MODEL = "gpt-4o-mini"

# ── arm / property definitions ────────────────────────────────────────────────

PROPERTY_QUESTIONS = {
    "parity":       "Is the number odd or even?",
    "prime":        "Is the number prime?",
    "div3":         "Is the number divisible by 3?",
    "div7":         "Is the number divisible by 7?",
    "div5":         "Is the number divisible by 5?",
    "gt50":         "Is the number greater than 50?",
    "square":       "Is the number a perfect square?",
    "digitsum":     "What is the sum of the digits?",
    "lastdigit":    "What is the last digit?",
    "between25_75": "Is the number between 25 and 75?",
}

ARM_PROPERTIES = {
    0: ["parity",    "gt50"],
    1: ["prime",     "square"],
    2: ["div3",      "div7",   "div5"],
    3: ["digitsum",  "lastdigit"],
    4: ["between25_75"],
}

# ── helpers ───────────────────────────────────────────────────────────────────

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def sanitize_for_api(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    return text


def sanitize_hint(hint, secret):
    """Remove direct mentions of the secret number from oracle hints."""
    if hint is None:
        return hint
    hint = re.sub(
        rf'(?<!\bby\s)\b{secret}\b(?!\s*(?:is|was|are))',
        "the number", hint
    )
    return hint.strip()

# ── oracle ────────────────────────────────────────────────────────────────────

def call_clue_holder(question, secret, candidates, max_retries=5):
    """Call GPT-4o-mini oracle. Returns one-sentence hint or None on failure."""
    prompt = sanitize_for_api(
        ORACLE_PROMPT_TEMPLATE.format(secret=secret, question=question)
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model    = CLUE_MODEL,
                messages = [
                    {"role": "system", "content": ORACLE_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature = 0.0,
                max_tokens  = 60,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e).lower()
            if "400" in err_str and "json" in err_str:
                return None
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt + random.uniform(0, 1))
    return None


def simulate_hint(question, secret):
    """Deterministic fallback for known property questions (avoids API calls)."""
    q = question.lower()
    if "odd or even" in q:
        return "The number is odd." if secret % 2 != 0 else "The number is even."
    elif "prime" in q and "perfect" not in q:
        return "The number is prime." if is_prime(secret) else "The number is not prime."
    elif "divisible by 3" in q:
        return "The number is divisible by 3." if secret % 3 == 0 else "The number is not divisible by 3."
    elif "divisible by 7" in q:
        return "The number is divisible by 7." if secret % 7 == 0 else "The number is not divisible by 7."
    elif "divisible by 5" in q:
        return "The number is divisible by 5." if secret % 5 == 0 else "The number is not divisible by 5."
    elif "greater than 50" in q:
        return "The number is greater than 50." if secret > 50 else "The number is less than 50."
    elif "perfect square" in q:
        return "The number is a perfect square." if int(secret**0.5)**2 == secret else "The number is not a perfect square."
    elif "sum of the digits" in q:
        return f"The sum of the digits is {sum(int(d) for d in str(secret))}."
    elif "last digit" in q:
        return f"The last digit is {secret % 10}."
    elif "between 25 and 75" in q:
        return "The number is between 25 and 75." if 25 <= secret <= 75 else "The number is not between 25 and 75."
    return None


def discrimination_power(question, candidates, secret):
    hint = simulate_hint(question, secret)
    if hint is None:
        return 0
    return len(candidates) - len(filter_candidates(candidates, hint))

# ── belief update ─────────────────────────────────────────────────────────────

def filter_candidates(candidates, hint):
    """Update candidate set given an oracle hint (belief update operator τ)."""
    if hint is None:
        return candidates
    hint_lower = hint.lower()
    if "cannot help" in hint_lower or "already asked" in hint_lower:
        return candidates

    filtered = candidates[:]

    if any(p in hint_lower for p in ["is odd", "number is odd", "not even", "not an even"]):
        filtered = [c for c in filtered if c % 2 != 0]
    elif any(p in hint_lower for p in ["is even", "number is even", "not odd", "not an odd"]):
        filtered = [c for c in filtered if c % 2 == 0]

    if any(p in hint_lower for p in ["not a prime", "not prime", "is not prime", "not a prime number"]):
        filtered = [c for c in filtered if not is_prime(c)]
    elif any(p in hint_lower for p in ["is a prime", "is prime", "a prime number"]):
        filtered = [c for c in filtered if is_prime(c)]

    if any(p in hint_lower for p in ["not a perfect square", "not perfect square", "not a square number"]):
        filtered = [c for c in filtered if int(c**0.5)**2 != c]
    elif any(p in hint_lower for p in ["perfect square", "square number", "is a square"]):
        filtered = [c for c in filtered if int(c**0.5)**2 == c]

    not_gt_match = re.search(r"not greater than (\d+)", hint_lower)
    gt_match     = re.search(r"greater than (\d+)", hint_lower)
    if not_gt_match and "digit" not in hint_lower:
        filtered = [c for c in filtered if c <= int(not_gt_match.group(1))]
    elif gt_match and "digit" not in hint_lower:
        filtered = [c for c in filtered if c > int(gt_match.group(1))]

    not_lt_match = re.search(r"(?:not less than|not smaller than)\s+(\d+)", hint_lower)
    lt_match     = re.search(r"(?:less than|smaller than)\s+(\d+)", hint_lower)
    if not_lt_match:
        filtered = [c for c in filtered if c >= int(not_lt_match.group(1))]
    elif lt_match and "digit" not in hint_lower:
        filtered = [c for c in filtered if c < int(lt_match.group(1))]

    not_lastdigit_match = re.search(r"last digit.{0,15}not\s+(\d+)", hint_lower)
    lastdigit_lt_match  = re.search(r"last digit.{0,20}less than\s+(\d+)", hint_lower)
    lastdigit_gt_match  = re.search(r"last digit.{0,20}greater than\s+(\d+)", hint_lower)
    lastdigit_match     = re.search(r"last digit(?:\s+of\s+the\s+number)?\s+is\s+(\d+)", hint_lower)
    onesdigit_match     = re.search(r"ones digit is (\d+)", hint_lower)
    ends_match          = re.search(r"ends (?:in|with) (\d+)", hint_lower)

    if not_lastdigit_match:
        filtered = [c for c in filtered if c % 10 != int(not_lastdigit_match.group(1))]
    elif lastdigit_lt_match:
        filtered = [c for c in filtered if c % 10 < int(lastdigit_lt_match.group(1))]
    elif lastdigit_gt_match:
        filtered = [c for c in filtered if c % 10 > int(lastdigit_gt_match.group(1))]
    elif lastdigit_match or onesdigit_match or ends_match:
        digit_match = lastdigit_match or onesdigit_match or ends_match
        filtered = [c for c in filtered if c % 10 == int(digit_match.group(1))]

    atmost_match  = re.search(r"at most\s*(\d+)", hint_lower)
    atleast_match = re.search(r"at least\s*(\d+)", hint_lower)
    if atmost_match:
        filtered = [c for c in filtered if c <= int(atmost_match.group(1))]
    if atleast_match:
        filtered = [c for c in filtered if c >= int(atleast_match.group(1))]

    not_between_match = re.search(r"not between (\d+)(?:\s+and\s+|\s+to\s+)(\d+)", hint_lower)
    between_match     = re.search(r"between (\d+)(?:\s+and\s+|\s+to\s+)(\d+)", hint_lower)
    if not_between_match:
        lo, hi = int(not_between_match.group(1)), int(not_between_match.group(2))
        filtered = [c for c in filtered if not (lo <= c <= hi)]
    elif between_match:
        lo, hi = int(between_match.group(1)), int(between_match.group(2))
        filtered = [c for c in filtered if lo <= c <= hi]

    div_match = re.search(r"\b(not\s+)?divisible\s+by\s+(\d+)", hint_lower)
    if div_match:
        d = int(div_match.group(2))
        if div_match.group(1):
            filtered = [c for c in filtered if c % d != 0]
        else:
            filtered = [c for c in filtered if c % d == 0]

    candiv_match = re.search(r"can be divided by\s*(\d+)", hint_lower)
    if candiv_match:
        filtered = [c for c in filtered if c % int(candiv_match.group(1)) == 0]

    not_mult_match = re.search(r"not (?:a )?multiple of (\d+)", hint_lower)
    mult_match     = re.search(r"(?:a )?multiple of (\d+)", hint_lower)
    if not_mult_match:
        filtered = [c for c in filtered if c % int(not_mult_match.group(1)) != 0]
    elif mult_match:
        filtered = [c for c in filtered if c % int(mult_match.group(1)) == 0]

    range_match = re.search(r"(?:in the range|range of)\s+(\d+)\s+(?:to|and)\s+(\d+)", hint_lower)
    if range_match and not between_match and not not_between_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        filtered = [c for c in filtered if lo <= c <= hi]

    exceeds_match = re.search(r"(?:exceeds|is above|above)\s+(\d+)", hint_lower)
    if exceeds_match:
        filtered = [c for c in filtered if c > int(exceeds_match.group(1))]

    not_exceeds_match = re.search(r"does not exceed\s+(\d+)", hint_lower)
    if not_exceeds_match:
        filtered = [c for c in filtered if c <= int(not_exceeds_match.group(1))]

    dsum_match      = re.search(r"sum of.{0,40}digits.{0,40}(?:is|=)\s*(\d+)", hint_lower)
    dsum_alt_match  = re.search(r"digit sum.{0,30}(?:is|=)\s*(\d+)", hint_lower)
    dsum_addup_match= re.search(r"digits.{0,20}(?:sum|add)\s+(?:up\s+)?to\s+(\d+)", hint_lower)
    dsum_gt_match   = re.search(r"sum of.{0,20}digits.{0,20}greater than\s+(\d+)", hint_lower)
    dsum_lt_match   = re.search(r"sum of.{0,20}digits.{0,20}less than\s+(\d+)", hint_lower)
    dsum_result     = dsum_match or dsum_alt_match or dsum_addup_match

    if dsum_gt_match:
        filtered = [c for c in filtered if sum(int(d) for d in str(c)) > int(dsum_gt_match.group(1))]
    elif dsum_lt_match:
        filtered = [c for c in filtered if sum(int(d) for d in str(c)) < int(dsum_lt_match.group(1))]
    elif dsum_result:
        s = int(dsum_result.group(1))
        filtered = [c for c in filtered if sum(int(d) for d in str(c)) == s]

    if "not composite" in hint_lower or "not a composite" in hint_lower:
        filtered = [c for c in filtered if is_prime(c) or c == 1]
    elif "is composite" in hint_lower or "a composite number" in hint_lower:
        filtered = [c for c in filtered if not is_prime(c) and c > 1]

    if "not a power of 2" in hint_lower or "not a power of two" in hint_lower:
        filtered = [c for c in filtered if (c & (c - 1)) != 0 or c == 0]
    elif "power of 2" in hint_lower or "power of two" in hint_lower:
        filtered = [c for c in filtered if c > 0 and (c & (c - 1)) == 0]

    palindromes = {n for n in range(1, 201) if str(n) == str(n)[::-1]}
    if "not a palindrome" in hint_lower or "not palindrome" in hint_lower:
        filtered = [c for c in filtered if c not in palindromes]
    elif "is a palindrome" in hint_lower or "palindrome" in hint_lower:
        filtered = [c for c in filtered if c in palindromes]

    fibs = {1, 2, 3, 5, 8, 13, 21, 34, 55, 89}
    if "not a fibonacci" in hint_lower:
        filtered = [c for c in filtered if c not in fibs]
    elif "fibonacci" in hint_lower:
        filtered = [c for c in filtered if c in fibs]

    triangulars = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91}
    if "not a triangular" in hint_lower:
        filtered = [c for c in filtered if c not in triangulars]
    elif "triangular" in hint_lower:
        filtered = [c for c in filtered if c in triangulars]

    return filtered if filtered else candidates

# ── prompt builder / response parser ─────────────────────────────────────────

def build_llm_prompt(candidates, arm_histories, turn, max_turns, n_arms, use_cot=False):
    """Alias for build_selector_prompt; use_cot kept for backward compatibility."""
    if use_cot:
        cot_prefix = (
            "Think step by step:\n"
            "1. Which arms have already been queried and what did you learn?\n"
            "2. Which arm has the most remaining information?\n"
            "3. What property of the number would eliminate the most candidates?\n"
            "Then output your decision.\n\n"
        )
        base = build_selector_prompt(candidates, arm_histories, turn, max_turns, n_arms)
        return cot_prefix + base
    return build_selector_prompt(candidates, arm_histories, turn, max_turns, n_arms)


def parse_llm_response(response_text, n_arms):
    """Parse model output -> (arm_idx, question). Falls back to parity on failure."""
    if not response_text:
        return random.randint(0, n_arms - 1), PROPERTY_QUESTIONS["parity"]

    try:
        parsed = json.loads(response_text)
        return int(parsed["arm"]) % n_arms, parsed["question"]
    except Exception:
        pass

    cleaned = re.sub(r"```(?:json)?", "", response_text).strip().rstrip("`").strip()
    try:
        parsed = json.loads(cleaned)
        return int(parsed["arm"]) % n_arms, parsed["question"]
    except Exception:
        pass

    json_matches = re.findall(r'\{[^{}]*"arm"[^{}]*"question"[^{}]*\}', response_text)
    if json_matches:
        try:
            parsed = json.loads(json_matches[-1])
            return int(parsed["arm"]) % n_arms, parsed["question"]
        except Exception:
            pass

    for line in reversed(response_text.strip().split("\n")):
        line = line.strip().rstrip("`").strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                parsed = json.loads(line)
                if "arm" in parsed and "question" in parsed:
                    return int(parsed["arm"]) % n_arms, parsed["question"]
            except Exception:
                continue

    return random.randint(0, n_arms - 1), PROPERTY_QUESTIONS["parity"]
