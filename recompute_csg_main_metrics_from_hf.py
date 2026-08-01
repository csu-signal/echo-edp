#!/usr/bin/env python3
"""Recompute the main CSG evaluation metrics from a Hugging Face dataset.

This script rebuilds
all metrics from raw turn/episode fields in the published dataset .
Expected dataset layout
-----------------------
A Hugging Face DatasetDict containing a ``turns`` split. Each row should
represent one turn and contain, either directly or through ``turn_json``:

Required identifiers
    model_key or model
    run_id
    episode_id
    turn_index or turn

Required raw game fields
    candidates_before
    candidates_after
    eliminated
    redundant (optional if the hint text reliably marks redundancy)
    hint (optional but used as a redundancy fallback)

Required for Reason%
    raw_response or completion

Required for Resolve
    One of:
      * an ``episodes`` split containing ``episode_id`` and ``resolved``;
      * ``episode_resolved`` repeated on turn rows; or
      * the final turn's ``candidates_after`` value.

Main metrics
------------
Resolve
    Fraction of episodes that identify the secret within the turn budget.

Zero
    Fraction of NON-REDUNDANT turns that eliminate zero candidates.

Qual
    Turn-level elimination quality:
        min(eliminated / max(candidates_before / 2, 1), 1)
    It is averaged over all valid turns within a run.

Ground
    Fraction of all turns that are non-redundant and have Qual >= 0.5.

LateCalib
    Redundancy rate among turns at position >= 5 with <= 10 candidates before
    the action. Lower is better.

ReasonPct
    Fraction of turns with visible text outside the required structured action
    JSON. This is a diagnostic heuristic.

ZeroEvents
    Number of non-redundant zero-elimination turns that have a following turn.
    This is the denominator for recovery metrics.

Rec1
    Fraction of ZeroEvents whose immediate next turn is non-redundant and
    eliminates at least one candidate.

GRecover
    Fraction of ZeroEvents whose immediate next turn is grounded.

RecQ
    Mean Qual of the immediate next turn after a ZeroEvent.

NextZero
    Fraction of ZeroEvents immediately followed by another non-redundant
    zero-elimination turn. Lower is better.

NextBad
    Fraction of ZeroEvents immediately followed by either another Zero turn or
    a redundant turn. Lower is better.

TTR
    Number of turn transitions from a ZeroEvent to the next grounded turn,
    averaged only over events for which grounded recovery occurs. Lower is
    better. Immediate grounded recovery has TTR = 1.

TTRsucc
    Fraction of ZeroEvents followed by a grounded turn at any later point in
    the same episode.

ResAfterZero
    Fraction of episodes containing at least one ZeroEvent that still resolve.

Aggregation
-----------
Metrics are first computed independently for each run. Model-level values are
then the equal-weight mean and standard error across runs. Event metrics are
pooled across events within each run before run-level averaging.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "This script requires Hugging Face datasets. Install it with: "
        "pip install datasets pyarrow pandas numpy"
    ) from exc


# =============================================================================
# 1. Metric constants
# =============================================================================

MAX_TURNS = 10
GROUND_THRESHOLD = 0.5
LATE_TURN_MIN = 5          # one-based position in the episode
NARROW_CANDIDATE_MAX = 10

# Reason% reproduces the original diagnostic idea while making the thresholds
# explicit and configurable in one place.
REASON_OUTSIDE_JSON_MIN_CHARS = 10
REASON_NO_ACTION_JSON_MIN_CHARS = 50

PUBLIC_METRICS = [
    "Resolve",
    "Zero",
    "Qual",
    "Ground",
    "LateCalib",
    "ReasonPct",
    "ZeroEvents",
    "Rec1",
    "GRecover",
    "RecQ",
    "NextZero",
    "NextBad",
    "TTR",
    "TTRsucc",
    "ResAfterZero",
]


# =============================================================================
# 2. Dataset loading and raw-field normalization
# =============================================================================


def load_hf_dataset(source: str, revision: Optional[str] = None) -> DatasetDict:
    """Load a DatasetDict from a local directory or a Hugging Face Hub ID."""
    path = Path(source).expanduser()

    if path.exists():
        loaded = load_from_disk(str(path))
    else:
        loaded = load_dataset(source, revision=revision)

    if isinstance(loaded, Dataset):
        return DatasetDict({"turns": loaded})
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Unsupported dataset object: {type(loaded)!r}")
    return loaded


def _json_object(value: Any) -> dict[str, Any]:
    """Parse a JSON dictionary if possible; otherwise return an empty dict."""
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        obj = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return dict(obj) if isinstance(obj, Mapping) else {}


def coalesce_raw_field(row: pd.Series, names: Iterable[str]) -> Any:
    """Read a raw field directly, falling back to the optional turn_json blob."""
    for name in names:
        if name in row.index and not _is_missing(row[name]):
            return row[name]

    raw_turn = _json_object(row.get("turn_json"))
    for name in names:
        if name in raw_turn and not _is_missing(raw_turn[name]):
            return raw_turn[name]

    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def to_float(value: Any, field: str, context: str) -> float:
    if _is_missing(value):
        raise ValueError(f"{context}: missing required field '{field}'")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: invalid {field}={value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"{context}: non-finite {field}={value!r}")
    return out


def to_bool(value: Any) -> bool:
    """Convert common stored boolean representations without bool('False') bugs."""
    if _is_missing(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    raise ValueError(f"Cannot interpret boolean value: {value!r}")


def normalize_turns(turns: pd.DataFrame) -> pd.DataFrame:
    """Create a canonical raw-turn table and discard precomputed metrics.

    Only identifiers, raw game fields, and raw model text are retained. Derived
    metric columns from the source dataset are intentionally ignored.
    """
    if turns.empty:
        raise ValueError("The turns split is empty.")

    canonical_rows: list[dict[str, Any]] = []

    for source_row_index, row in turns.iterrows():
        model_key = coalesce_raw_field(row, ["model_key", "model"])
        run_id = coalesce_raw_field(row, ["run_id"])
        episode_id = coalesce_raw_field(row, ["episode_id"])
        raw_turn_index = coalesce_raw_field(row, ["turn_index", "turn"])

        context = (
            f"source row {source_row_index}, model={model_key!r}, "
            f"run={run_id!r}, episode={episode_id!r}"
        )

        if _is_missing(model_key):
            raise ValueError(f"{context}: missing model_key/model")
        if _is_missing(run_id):
            raise ValueError(f"{context}: missing run_id")
        if _is_missing(episode_id):
            raise ValueError(f"{context}: missing episode_id")
        if _is_missing(raw_turn_index):
            raise ValueError(f"{context}: missing turn_index/turn")

        try:
            turn_index = int(raw_turn_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}: invalid turn index {raw_turn_index!r}") from exc

        before = to_float(
            coalesce_raw_field(row, ["candidates_before"]),
            "candidates_before",
            context,
        )
        after = to_float(
            coalesce_raw_field(row, ["candidates_after"]),
            "candidates_after",
            context,
        )

        raw_eliminated = coalesce_raw_field(row, ["eliminated"])
        if _is_missing(raw_eliminated):
            eliminated = before - after
        else:
            eliminated = to_float(raw_eliminated, "eliminated", context)

        if before <= 0:
            raise ValueError(f"{context}: candidates_before must be positive")
        if after < 0:
            raise ValueError(f"{context}: candidates_after cannot be negative")
        if eliminated < 0:
            raise ValueError(f"{context}: eliminated cannot be negative")

        # Check, rather than silently fix, inconsistent raw logs.
        expected_eliminated = before - after
        if not math.isclose(eliminated, expected_eliminated, abs_tol=1e-9):
            raise ValueError(
                f"{context}: eliminated={eliminated} but "
                f"candidates_before-candidates_after={expected_eliminated}"
            )

        hint = coalesce_raw_field(row, ["hint"])
        redundant_raw = coalesce_raw_field(row, ["redundant"])
        raw_response = coalesce_raw_field(row, ["raw_response"])
        completion = coalesce_raw_field(row, ["completion"])
        episode_resolved = coalesce_raw_field(row, ["episode_resolved", "resolved"])

        canonical_rows.append(
            {
                "model_key": str(model_key),
                "model": str(coalesce_raw_field(row, ["model"]) or model_key),
                "group": coalesce_raw_field(row, ["group"]),
                "run_id": str(run_id),
                "run_index": coalesce_raw_field(row, ["run_index"]),
                "episode_id": str(episode_id),
                "episode_index": coalesce_raw_field(row, ["episode_index"]),
                "turn_index": turn_index,
                "candidates_before": before,
                "candidates_after": after,
                "eliminated": eliminated,
                "redundant_raw": redundant_raw,
                "hint": "" if _is_missing(hint) else str(hint),
                "raw_response": None if _is_missing(raw_response) else str(raw_response),
                "completion": None if _is_missing(completion) else str(completion),
                "episode_resolved_raw": episode_resolved,
            }
        )

    out = pd.DataFrame(canonical_rows)

    sort_cols = ["model_key", "run_id", "episode_id", "turn_index"]
    out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    duplicate_mask = out.duplicated(
        ["model_key", "run_id", "episode_id", "turn_index"], keep=False
    )
    if duplicate_mask.any():
        duplicates = out.loc[
            duplicate_mask,
            ["model_key", "run_id", "episode_id", "turn_index"],
        ]
        raise ValueError(
            "Duplicate turn indices found within episodes:\n"
            + duplicates.to_string(index=False)
        )

    episode_sizes = out.groupby(
        ["model_key", "run_id", "episode_id"], sort=False
    ).size()
    too_long = episode_sizes[episode_sizes > MAX_TURNS]
    if not too_long.empty:
        raise ValueError(
            f"Found episodes longer than MAX_TURNS={MAX_TURNS}:\n"
            + too_long.to_string()
        )

    return out


def normalize_episodes(episodes: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Keep only raw episode outcome fields, ignoring episode metric columns."""
    if episodes is None or episodes.empty:
        return pd.DataFrame(columns=["model_key", "run_id", "episode_id", "resolved"])

    rows: list[dict[str, Any]] = []
    for source_row_index, row in episodes.iterrows():
        episode_id = coalesce_raw_field(row, ["episode_id"])
        resolved = coalesce_raw_field(row, ["resolved", "episode_resolved"])
        if _is_missing(episode_id) or _is_missing(resolved):
            continue
        rows.append(
            {
                "model_key": coalesce_raw_field(row, ["model_key", "model"]),
                "run_id": coalesce_raw_field(row, ["run_id"]),
                "episode_id": str(episode_id),
                "resolved": to_bool(resolved),
                "_source_row": source_row_index,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["model_key", "run_id", "episode_id", "resolved"])

    out = pd.DataFrame(rows)
    out["model_key"] = out["model_key"].where(out["model_key"].notna(), None)
    out["run_id"] = out["run_id"].where(out["run_id"].notna(), None)
    return out.drop(columns="_source_row")


# =============================================================================
# 3. Turn-level metric primitives
# =============================================================================


def is_redundant_turn(row: pd.Series) -> bool:
    """Redundancy from the raw flag, with original hint-text fallbacks."""
    raw_flag = row.get("redundant_raw")
    redundant_flag = False if _is_missing(raw_flag) else to_bool(raw_flag)
    hint = str(row.get("hint", "")).lower()
    return redundant_flag or "already asked" in hint or "no new information" in hint


def turn_quality(candidates_before: float, eliminated: float) -> float:
    """Candidate-elimination quality relative to an ideal binary split."""
    ideal_elimination = max(candidates_before / 2.0, 1.0)
    return min(eliminated / ideal_elimination, 1.0)


def _find_action_json_span(text: str) -> Optional[tuple[int, int]]:
    """Find a decodable JSON object containing an ``arm`` field."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            obj, consumed = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, Mapping) and "arm" in obj:
            return start, start + consumed
    return None


def reasoning_presence(raw_response: Optional[str], completion: Optional[str]) -> int:
    """Detect visible rationale outside the required structured action format."""
    raw = raw_response if raw_response is not None else completion
    if raw is None:
        raise ValueError(
            "ReasonPct cannot be computed because both raw_response and completion "
            "are missing for a turn."
        )

    text = str(raw).strip()
    if not text:
        return 0

    cleaned = text.replace("```json", "").replace("```", "").strip()
    span = _find_action_json_span(cleaned)

    if span is not None:
        start, end = span
        outside = (cleaned[:start].strip() + " " + cleaned[end:].strip()).strip()
        return int(len(outside) > REASON_OUTSIDE_JSON_MIN_CHARS)

    return int(len(cleaned) > REASON_NO_ACTION_JSON_MIN_CHARS)


def add_turn_derivatives(turns: pd.DataFrame) -> pd.DataFrame:
    """Add only the primitive fields needed to aggregate the main metrics."""
    out = turns.copy()

    out["turn_position"] = (
        out.groupby(["model_key", "run_id", "episode_id"], sort=False).cumcount() + 1
    )

    out["redundant"] = out.apply(is_redundant_turn, axis=1)
    out["Qual_turn"] = [
        turn_quality(before, eliminated)
        for before, eliminated in zip(out["candidates_before"], out["eliminated"])
    ]
    out["zero_nonredundant"] = (~out["redundant"]) & (out["eliminated"] == 0)
    out["grounded"] = (~out["redundant"]) & (
        out["Qual_turn"] >= GROUND_THRESHOLD
    )
    out["late_narrow"] = (
        (out["turn_position"] >= LATE_TURN_MIN)
        & (out["candidates_before"] <= NARROW_CANDIDATE_MAX)
    )
    out["reasoning_present"] = [
        reasoning_presence(raw, completion)
        for raw, completion in zip(out["raw_response"], out["completion"])
    ]

    return out


# =============================================================================
# 4. Episode outcomes and ZeroEvent recovery records
# =============================================================================


def build_episode_table(
    turns: pd.DataFrame,
    episode_source: pd.DataFrame,
) -> pd.DataFrame:
    """Resolve one raw outcome per episode without using metric columns."""
    episode_rows: list[dict[str, Any]] = []

    episode_lookup_exact: dict[tuple[str, str, str], bool] = {}
    episode_lookup_id: dict[str, bool] = {}

    for _, row in episode_source.iterrows():
        episode_id = str(row["episode_id"])
        resolved = bool(row["resolved"])
        episode_lookup_id[episode_id] = resolved
        if not _is_missing(row.get("model_key")) and not _is_missing(row.get("run_id")):
            episode_lookup_exact[
                (str(row["model_key"]), str(row["run_id"]), episode_id)
            ] = resolved

    keys = ["model_key", "run_id", "episode_id"]
    for (model_key, run_id, episode_id), ep in turns.groupby(keys, sort=False):
        ep = ep.sort_values("turn_position", kind="stable")
        exact_key = (str(model_key), str(run_id), str(episode_id))

        if exact_key in episode_lookup_exact:
            resolved = episode_lookup_exact[exact_key]
        elif str(episode_id) in episode_lookup_id:
            resolved = episode_lookup_id[str(episode_id)]
        else:
            repeated = ep["episode_resolved_raw"].dropna()
            if not repeated.empty:
                bool_values = repeated.map(to_bool).unique()
                if len(bool_values) != 1:
                    raise ValueError(
                        f"Conflicting episode_resolved values for {exact_key}: "
                        f"{bool_values.tolist()}"
                    )
                resolved = bool(bool_values[0])
            else:
                resolved = bool(ep.iloc[-1]["candidates_after"] == 1)

        episode_rows.append(
            {
                "model_key": model_key,
                "model": ep.iloc[0]["model"],
                "group": ep.iloc[0]["group"],
                "run_id": run_id,
                "run_index": ep.iloc[0]["run_index"],
                "episode_id": episode_id,
                "Resolve_episode": resolved,
                "turns_taken": len(ep),
            }
        )

    return pd.DataFrame(episode_rows)


def build_recovery_events(turns: pd.DataFrame) -> pd.DataFrame:
    """Create one record for every recoverable ZeroEvent.

    A ZeroEvent is a non-redundant zero-elimination turn with a following turn.
    """
    event_rows: list[dict[str, Any]] = []
    keys = ["model_key", "run_id", "episode_id"]

    for (model_key, run_id, episode_id), ep in turns.groupby(keys, sort=False):
        ep = ep.sort_values("turn_position", kind="stable").reset_index(drop=True)

        for i in range(len(ep) - 1):
            current = ep.iloc[i]
            if not bool(current["zero_nonredundant"]):
                continue

            nxt = ep.iloc[i + 1]
            immediate_nonzero = (not bool(nxt["redundant"])) and (nxt["eliminated"] > 0)
            immediate_grounded = bool(nxt["grounded"])
            next_zero = bool(nxt["zero_nonredundant"])
            next_bad = next_zero or bool(nxt["redundant"])

            later_grounded_positions = [
                j for j in range(i + 1, len(ep)) if bool(ep.iloc[j]["grounded"])
            ]
            ttr_success = len(later_grounded_positions) > 0
            ttr = (
                later_grounded_positions[0] - i
                if ttr_success
                else np.nan
            )

            event_rows.append(
                {
                    "model_key": model_key,
                    "model": current["model"],
                    "group": current["group"],
                    "run_id": run_id,
                    "run_index": current["run_index"],
                    "episode_id": episode_id,
                    "zero_turn_position": int(current["turn_position"]),
                    "Rec1_event": bool(immediate_nonzero),
                    "GRecover_event": bool(immediate_grounded),
                    "RecQ_event": float(nxt["Qual_turn"]),
                    "NextZero_event": bool(next_zero),
                    "NextBad_event": bool(next_bad),
                    "TTR_event": float(ttr) if ttr_success else np.nan,
                    "TTRsucc_event": bool(ttr_success),
                }
            )

    columns = [
        "model_key",
        "model",
        "group",
        "run_id",
        "run_index",
        "episode_id",
        "zero_turn_position",
        "Rec1_event",
        "GRecover_event",
        "RecQ_event",
        "NextZero_event",
        "NextBad_event",
        "TTR_event",
        "TTRsucc_event",
    ]
    return pd.DataFrame(event_rows, columns=columns)


# =============================================================================
# 5. Run-level and model-level aggregation
# =============================================================================


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if not values.empty else np.nan


def sem(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) <= 1:
        return np.nan
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def build_run_metrics(
    turns: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the publication metrics independently within each run."""
    rows: list[dict[str, Any]] = []

    for (model_key, run_id), tg in turns.groupby(["model_key", "run_id"], sort=False):
        eg = episodes[
            (episodes["model_key"] == model_key) & (episodes["run_id"] == run_id)
        ]
        vg = events[
            (events["model_key"] == model_key) & (events["run_id"] == run_id)
        ]

        if eg.empty:
            raise ValueError(f"No episode rows found for model={model_key}, run={run_id}")

        nonredundant = tg.loc[~tg["redundant"]]
        late_eligible = tg.loc[tg["late_narrow"]]

        event_episode_ids = set(vg["episode_id"].astype(str))
        episodes_after_zero = eg[eg["episode_id"].astype(str).isin(event_episode_ids)]

        row = {
            "model_key": model_key,
            "model": tg.iloc[0]["model"],
            "group": tg.iloc[0]["group"],
            "run_id": run_id,
            "run_index": tg.iloc[0]["run_index"],
            "n_episodes": int(len(eg)),
            "n_turns": int(len(tg)),

            "Resolve": safe_mean(eg["Resolve_episode"]),
            # Zero's denominator is explicitly NON-REDUNDANT turns.
            "Zero": safe_mean(nonredundant["zero_nonredundant"]),
            "Qual": safe_mean(tg["Qual_turn"]),
            # Ground's denominator is all turns.
            "Ground": safe_mean(tg["grounded"]),
            "LateCalib": safe_mean(late_eligible["redundant"]),
            "ReasonPct": safe_mean(tg["reasoning_present"]),
            "ZeroEvents": int(len(vg)),
            "Rec1": safe_mean(vg["Rec1_event"]),
            "GRecover": safe_mean(vg["GRecover_event"]),
            "RecQ": safe_mean(vg["RecQ_event"]),
            "NextZero": safe_mean(vg["NextZero_event"]),
            "NextBad": safe_mean(vg["NextBad_event"]),
            # TTR is conditional on a successful later grounded recovery.
            "TTR": safe_mean(vg.loc[vg["TTRsucc_event"], "TTR_event"]),
            "TTRsucc": safe_mean(vg["TTRsucc_event"]),
            "ResAfterZero": safe_mean(episodes_after_zero["Resolve_episode"]),
        }
        rows.append(row)

    columns = [
        "model_key",
        "model",
        "group",
        "run_id",
        "run_index",
        "n_episodes",
        "n_turns",
        *PUBLIC_METRICS,
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["group", "model_key", "run_id"], kind="stable", na_position="last"
    ).reset_index(drop=True)


def build_model_metrics(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight model mean and SEM across independent runs."""
    rows: list[dict[str, Any]] = []

    for model_key, g in run_metrics.groupby("model_key", sort=False):
        row: dict[str, Any] = {
            "model_key": model_key,
            "model": g.iloc[0]["model"],
            "group": g.iloc[0]["group"],
            "n_runs": int(g["run_id"].nunique()),
            "n_episodes": int(g["n_episodes"].sum()),
            "n_turns": int(g["n_turns"].sum()),
        }

        for metric in PUBLIC_METRICS:
            row[f"{metric}_mean"] = safe_mean(g[metric])
            row[f"{metric}_sem"] = sem(g[metric])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["group", "model_key"], kind="stable", na_position="last"
    ).reset_index(drop=True)


# =============================================================================
# 6. Validation and export
# =============================================================================


def validate_outputs(
    turns: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    run_metrics: pd.DataFrame,
) -> None:
    """Fail loudly on denominator or range errors."""
    bounded_metrics = [
        "Resolve",
        "Zero",
        "Qual",
        "Ground",
        "LateCalib",
        "ReasonPct",
        "Rec1",
        "GRecover",
        "RecQ",
        "NextZero",
        "NextBad",
        "TTRsucc",
        "ResAfterZero",
    ]

    for metric in bounded_metrics:
        values = pd.to_numeric(run_metrics[metric], errors="coerce").dropna()
        invalid = values[(values < 0) | (values > 1)]
        if not invalid.empty:
            raise AssertionError(f"{metric} has values outside [0, 1]: {invalid.tolist()}")

    if (pd.to_numeric(run_metrics["ZeroEvents"], errors="raise") < 0).any():
        raise AssertionError("ZeroEvents cannot be negative.")

    ttr_values = pd.to_numeric(run_metrics["TTR"], errors="coerce").dropna()
    if (ttr_values < 1).any():
        raise AssertionError("TTR must be at least 1 when defined.")

    # Every recovery event must correspond to a turn that has a next turn.
    if not events.empty:
        event_counts = events.groupby(["model_key", "run_id"]).size()
        run_counts = run_metrics.set_index(["model_key", "run_id"])["ZeroEvents"]
        aligned = run_counts.reindex(event_counts.index)
        if not np.array_equal(aligned.to_numpy(), event_counts.to_numpy()):
            raise AssertionError("ZeroEvents does not match the recovery-event table.")

    if episodes["episode_id"].duplicated().any():
        # Episode IDs are expected to be globally unique in the supplied builder.
        # Use a compound-key check instead so repeated local IDs across runs remain valid.
        compound_dup = episodes.duplicated(["model_key", "run_id", "episode_id"])
        if compound_dup.any():
            raise AssertionError("Duplicate episode rows found.")

    if len(turns) == 0 or len(episodes) == 0:
        raise AssertionError("No turns or episodes were produced.")


def save_outputs(
    output_dir: Path,
    turns: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
    run_metrics: pd.DataFrame,
    model_metrics: pd.DataFrame,
    save_audits: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_audits:
        # Optional audit tables contain only raw identifiers and the primitive/event
        # fields needed to verify the main metrics. No auxiliary paper metrics are
        # computed or exported.
        turn_audit_columns = [
            "model_key",
            "model",
            "group",
            "run_id",
            "run_index",
            "episode_id",
            "turn_index",
            "turn_position",
            "candidates_before",
            "candidates_after",
            "eliminated",
            "redundant",
            "zero_nonredundant",
            "Qual_turn",
            "grounded",
            "late_narrow",
            "reasoning_present",
        ]
        episode_audit_columns = [
            "model_key",
            "model",
            "group",
            "run_id",
            "run_index",
            "episode_id",
            "Resolve_episode",
            "turns_taken",
        ]

        turns[turn_audit_columns].to_parquet(
            output_dir / "turn_metric_audit.parquet", index=False
        )
        episodes[episode_audit_columns].to_parquet(
            output_dir / "episode_metric_audit.parquet", index=False
        )
        events.to_parquet(output_dir / "recovery_event_audit.parquet", index=False)

    run_metrics.to_parquet(output_dir / "run_metrics.parquet", index=False)
    model_metrics.to_parquet(output_dir / "model_metrics.parquet", index=False)

    run_metrics.to_csv(output_dir / "run_metrics.csv", index=False)
    model_metrics.to_csv(output_dir / "model_metrics.csv", index=False)

    definitions = {
        "constants": {
            "MAX_TURNS": MAX_TURNS,
            "GROUND_THRESHOLD": GROUND_THRESHOLD,
            "LATE_TURN_MIN": LATE_TURN_MIN,
            "NARROW_CANDIDATE_MAX": NARROW_CANDIDATE_MAX,
            "REASON_OUTSIDE_JSON_MIN_CHARS": REASON_OUTSIDE_JSON_MIN_CHARS,
            "REASON_NO_ACTION_JSON_MIN_CHARS": REASON_NO_ACTION_JSON_MIN_CHARS,
        },
        "metrics": PUBLIC_METRICS,
        "aggregation": (
            "Metrics are computed per run; model means and SEMs are then "
            "computed with equal weight per run."
        ),
    }
    with open(output_dir / "metric_config.json", "w", encoding="utf-8") as handle:
        json.dump(definitions, handle, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the main CSG evaluation metrics from raw Hugging Face "
            "dataset fields, without pickle files or precomputed metric columns."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Local load_from_disk directory or Hugging Face Hub dataset ID.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face Hub revision/commit/tag.",
    )
    parser.add_argument(
        "--turns-split",
        default="turns",
        help="Name of the raw per-turn split. Default: turns",
    )
    parser.add_argument(
        "--episodes-split",
        default="episodes",
        help="Optional raw episode split used for resolved outcomes. Default: episodes",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for run/model metric summaries.",
    )
    parser.add_argument(
        "--save-audits",
        action="store_true",
        help=(
            "Also save turn-, episode-, and recovery-event audit tables. "
            "The default output contains only the main run/model metrics."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_hf_dataset(args.dataset, revision=args.revision)

    if args.turns_split not in dataset:
        raise KeyError(
            f"Dataset does not contain the required '{args.turns_split}' split. "
            f"Available splits: {list(dataset.keys())}"
        )

    raw_turns = dataset[args.turns_split].to_pandas()
    raw_episodes = (
        dataset[args.episodes_split].to_pandas()
        if args.episodes_split in dataset
        else None
    )

    turns = add_turn_derivatives(normalize_turns(raw_turns))
    episode_source = normalize_episodes(raw_episodes)
    episodes = build_episode_table(turns, episode_source)
    events = build_recovery_events(turns)
    run_metrics = build_run_metrics(turns, episodes, events)
    model_metrics = build_model_metrics(run_metrics)

    validate_outputs(turns, episodes, events, run_metrics)
    output_dir = Path(args.output_dir).expanduser()
    save_outputs(
        output_dir,
        turns,
        episodes,
        events,
        run_metrics,
        model_metrics,
        save_audits=args.save_audits,
    )

    print("Recomputed main CSG metrics from raw Hugging Face fields.")
    print(f"Turns:          {len(turns):,}")
    print(f"Episodes:       {len(episodes):,}")
    print(f"ZeroEvents:     {len(events):,}")
    print(f"Runs:           {len(run_metrics):,}")
    print(f"Models:         {len(model_metrics):,}")
    print(f"Outputs:        {output_dir}")
    print("\nModel-level metrics:")
    print(model_metrics.to_string(index=False))


if __name__ == "__main__":
    main()
