"""
ECHO advantage computation and baselines.

ECHO ("echo" mode): per-turn cross-generation normalization.
  Normalizes each turn's reward across G parallel rollouts of the same secret,
  giving a turn-level advantage that compares completions at the same belief state.

Baselines included:
  episode_return  — REINFORCE: sum rewards per episode, same value broadcast to all turns
  rloo_per_turn   — leave-one-out per turn
  igpo            — IGPO: group-pooled turn normalization + turn-level reward-to-go
  echo_rtg        — ECHO's per-turn normalization applied to raw reward-to-go
"""

import torch

MAX_TURNS = 10


def _echo_advantages(rewards, group_sizes, B, G, device):
    """
    ECHO: normalize within (secret, turn_position) group across G generations.

    rewards:     (total_turns,)
    padded:      (B*G, MAX_TURNS)  — pad each episode to MAX_TURNS
    reshaped:    (B, G, MAX_TURNS) — B secrets × G gens × T turns
    normalized across G dim per turn position, then unpadded back to (total_turns,)
    """
    padded   = torch.zeros(B * G, MAX_TURNS, device=device)
    flat_idx = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        padded[ep_idx, :ep_size] = rewards[flat_idx:flat_idx + ep_size]
        flat_idx += ep_size

    padded   = padded.view(B, G, MAX_TURNS)
    mean_r   = padded.mean(dim=1, keepdim=True)
    std_r    = padded.std(dim=1, keepdim=True).clamp_min(1e-4)
    padded   = ((padded - mean_r) / std_r).view(B * G, MAX_TURNS)

    advantages = torch.zeros_like(rewards)
    flat_idx   = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        advantages[flat_idx:flat_idx + ep_size] = padded[ep_idx, :ep_size]
        flat_idx += ep_size

    return advantages


def _episode_return_advantages(rewards, group_sizes, B, G, device):
    """
    REINFORCE: sum rewards per episode, normalize across G episodes per secret.
    All turns in an episode share the same advantage value.

    rewards:     (total_turns,)
    ep_returns:  (B*G,)  — one scalar per episode
    ep_returns:  (B, G)  — normalize across G
    broadcast back to (total_turns,)
    """
    ep_returns = torch.zeros(B * G, device=device)
    flat_idx   = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        ep_returns[ep_idx] = rewards[flat_idx:flat_idx + ep_size].sum()
        flat_idx += ep_size

    ep_returns = ep_returns.view(B, G)
    mean_r     = ep_returns.mean(dim=1, keepdim=True)
    std_r      = ep_returns.std(dim=1, keepdim=True).clamp_min(1e-4)
    ep_adv     = ((ep_returns - mean_r) / std_r).view(B * G)

    advantages = torch.zeros_like(rewards)
    flat_idx   = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        advantages[flat_idx:flat_idx + ep_size] = ep_adv[ep_idx]
        flat_idx += ep_size

    return advantages


def _rloo_per_turn_advantages(rewards, group_sizes, B, G, device):
    """
    RLOO leave-one-out per turn position.

    For each (secret, turn_position), the baseline for generation g is the
    mean of all other G-1 generations at that same position.

    rewards:    (total_turns,)
    padded:     (B*G, MAX_TURNS)
    reshaped:   (B, G, MAX_TURNS)
    loo_mean:   (B, G, MAX_TURNS) — mean excluding self per (b, t)
    """
    padded   = torch.zeros(B * G, MAX_TURNS, device=device)
    flat_idx = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        padded[ep_idx, :ep_size] = rewards[flat_idx:flat_idx + ep_size]
        flat_idx += ep_size

    padded   = padded.view(B, G, MAX_TURNS)
    total    = padded.sum(dim=1, keepdim=True)
    loo_mean = (total - padded) / max(G - 1, 1)
    std_r    = padded.std(dim=1, keepdim=True).clamp_min(1e-4)
    padded   = ((padded - loo_mean) / std_r).view(B * G, MAX_TURNS)

    advantages = torch.zeros_like(rewards)
    flat_idx   = 0
    for ep_idx in range(B * G):
        ep_size = group_sizes[ep_idx]
        advantages[flat_idx:flat_idx + ep_size] = padded[ep_idx, :ep_size]
        flat_idx += ep_size

    return advantages


def _igpo_advantages(rewards, group_sizes, B, G, device, outcomes=None, gamma=1.0):
    """
    IGPO (Wang et al., 2025), "separate" normalization.

    Dense turn rewards are z-scored over all turns of a secret's G episodes, outcomes are
    z-scored across the G episodes and added on each episode's final turn, and each turn's
    advantage is the discounted reward-to-go of the normalized rewards.
    """
    starts = [0]
    for gs in group_sizes:
        starts.append(starts[-1] + gs)

    norm = torch.zeros_like(rewards)
    for b in range(B):
        lo, hi = starts[b * G], starts[(b + 1) * G]
        r = rewards[lo:hi]
        norm[lo:hi] = (r - r.mean()) / r.std(unbiased=False).clamp_min(1e-4)

    if outcomes is not None:
        o = outcomes.to(device).view(B, G)
        o = ((o - o.mean(dim=1, keepdim=True)) / o.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)).view(-1)
        for ep_idx in range(B * G):
            norm[starts[ep_idx + 1] - 1] += o[ep_idx]

    advantages = torch.zeros_like(rewards)
    for ep_idx in range(B * G):
        running = 0.0
        for i in range(starts[ep_idx + 1] - 1, starts[ep_idx] - 1, -1):
            running = norm[i] + gamma * running
            advantages[i] = running

    return advantages


def _echo_rtg_advantages(rewards, group_sizes, B, G, device, gamma=1.0):
    """Discounted reward-to-go per turn, then normalized exactly as in ECHO."""
    rtg = torch.zeros_like(rewards)
    flat_idx = 0
    for ep_size in group_sizes:
        running = 0.0
        for i in range(flat_idx + ep_size - 1, flat_idx - 1, -1):
            running = rewards[i] + gamma * running
            rtg[i] = running
        flat_idx += ep_size

    return _echo_advantages(rtg, group_sizes, B, G, device)


def compute_multiturn_advantages(rewards, group_sizes, B, G, mode, device, outcomes=None, gamma=1.0):
    """
    Dispatcher for all advantage modes.

    Args:
        rewards     : (total_turns,) flat reward tensor
        group_sizes : list of ints — turns per episode, len = B*G
        B           : number of secrets in this batch
        G           : number of generations per secret
        mode        : "echo" | "episode_return" | "rloo_per_turn" | "igpo" | "echo_rtg"
        device      : torch device string
        outcomes    : optional (B*G,) episode outcome rewards; igpo normalizes them separately,
                      other modes add them to each episode's final turn reward
        gamma       : turn-level discount for igpo and echo_rtg

    Returns:
        advantages  : (total_turns,) tensor
    """
    if mode == "igpo":
        return _igpo_advantages(rewards, group_sizes, B, G, device, outcomes, gamma)
    if outcomes is not None:
        rewards = rewards.clone()
        ends    = torch.tensor(group_sizes, device=rewards.device).cumsum(0) - 1
        rewards[ends] += outcomes.to(rewards.device)

    if mode == "echo":
        return _echo_advantages(rewards, group_sizes, B, G, device)
    elif mode == "episode_return":
        return _episode_return_advantages(rewards, group_sizes, B, G, device)
    elif mode == "rloo_per_turn":
        return _rloo_per_turn_advantages(rewards, group_sizes, B, G, device)
    elif mode == "echo_rtg":
        return _echo_rtg_advantages(rewards, group_sizes, B, G, device, gamma)
    else:
        raise ValueError(f"Unknown advantage mode: {mode!r}. "
                         f"Choose from: echo, episode_return, rloo_per_turn, igpo, echo_rtg")
