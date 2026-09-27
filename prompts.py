"""
Prompt templates for the Clue Selector Game (CSG).
"""

SYSTEM_PROMPT = "You are a helpful assistant. Follow instructions exactly."

REACT_SYSTEM_PROMPT = (
    "You are solving a number guessing game using ReAct "
    "(Reasoning + Acting).\n"
    "At each turn: write a Thought about what you learned "
    "from past observations and what to do next, "
    "then write your Action as JSON.\n\n"
    "Format:\n"
    "Thought: <reasoning>\n"
    "Action: {\"arm\": <0-4>, \"question\": \"<question>\"}"
)

ORACLE_SYSTEM_PROMPT = "You are a clue holder in a number guessing game. Return one sentence only."

ORACLE_PROMPT_TEMPLATE = """You know the secret number is {secret}.

A player has asked you: "{question}"

Rules:
- Answer ONLY the specific property asked about
- Do NOT volunteer extra information beyond what was asked.
- If the question is vague or not about a specific property, say: "Please ask about a specific property like parity, divisibility, or range."
- Do not reveal the number directly

One sentence only."""


def build_selector_prompt(candidates, arm_histories, turn, max_turns, n_arms):
    """Build the prompt sent to the selector model each turn."""
    histories_text = ""
    for i, hist in enumerate(arm_histories):
        if hist:
            histories_text += f"  Arm {i} (queried {len(hist)}x):\n"
            for q, h in hist:
                histories_text += f"    Q: {q}\n"
                histories_text += f"    A: {h}\n"
        else:
            histories_text += f"  Arm {i}: not yet queried\n"

    return f"""You are solving a number guessing game.
A secret number between 1 and 100 has been chosen.
You have {n_arms} clue holders, each knowing one fact about the number.

Turn {turn + 1} of {max_turns}
Remaining candidates ({len(candidates)}): {candidates}

Arm histories:
{histories_text}
Good questions ask about specific properties that split the candidates:
- parity (odd/even)
- primality (is it prime?)
- divisibility (divisible by 3, 7, etc.)
- digit properties (sum of digits, last digit)
- range (greater/less than a specific value)
- perfect square or cube

Do NOT repeat a question already asked to any arm.
Pick the arm with the most remaining information and ask a precise property question.

Respond with JSON only: {{"arm": <0 to {n_arms-1}>, "question": "<precise property question>"}}"""


def build_react_prompt(candidates, react_history, turn, max_turns, n_arms):
    """
    Build the prompt sent to the ReAct selector each turn.
    react_history is a list of (raw_response, observation) tuples.
    """
    base_prompt = (
        f"Turn {turn+1} of {max_turns}\n"
        f"Remaining candidates ({len(candidates)}): {candidates}\n\n"
        f"You have {n_arms} clue holders (arms 0-{n_arms-1}), "
        f"each knowing specific facts about the number.\n\n"
        "Good questions ask about:\n"
        "- parity (odd/even)\n"
        "- range (greater/less than a value)\n"
        "- divisibility (by 3, 5, 7, etc.)\n"
        "- primality, perfect square\n"
        "- digit properties (sum of digits, last digit)\n\n"
        "Do NOT repeat a question already asked to any arm.\n"
    )

    if react_history:
        history_text = "Past interactions:\n"
        for i, (thought_action, observation) in enumerate(react_history):
            history_text += f"\n--- Turn {i+1} ---\n"
            history_text += f"{thought_action}\n"
            history_text += f"Observation: {observation}\n"
        full_prompt = history_text + "\n" + base_prompt
    else:
        full_prompt = base_prompt

    full_prompt += (
        f"\nRespond with:\n"
        f"Thought: <your reasoning>\n"
        f"Action: {{\"arm\": <0-{n_arms-1}>, "
        f"\"question\": \"<question>\"}}"
    )
    return full_prompt
