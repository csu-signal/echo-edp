#!/bin/bash
set -e

echo "=== CRAFT Sample Runs ==="

# ── API models ────────────────────────────────────────────
echo "--- API: gpt-4o-mini oracle+no_tools ---"
python run_craft.py --mode api --director gpt-4o-mini --oracle --oracle_n 5 --no_tools --structures 0 --turns 5

echo "--- API: gpt-4o-mini no_oracle+no_tools ---"
python run_craft.py --mode api --director gpt-4o-mini --no_tools --structures 0 --turns 5

echo "--- API: gpt-4o-mini no_oracle+tools ---"
python run_craft.py --mode api --director gpt-4o-mini --structures 0 --turns 5

# ── Local models ──────────────────────────────────────────
echo "--- Local: qwen-7b oracle+no_tools ---"
python run_craft.py --mode local --director qwen-7b --oracle --oracle_n 5 --no_tools --structures 0 --turns 5

echo "--- Local: qwen-7b no_oracle+no_tools ---"
python run_craft.py --mode local --director qwen-7b --no_tools --structures 0 --turns 5

echo "=== Done ==="
