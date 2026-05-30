#!/usr/bin/env bash
# MIN_COUNT trade-off (personalization side) for todos/rag-diagnosis-plan.md Phase 2.
# Simulate a user who JUST crossed each gate threshold: history depth == MIN_COUNT,
# gate OPEN. Compare personalization across depths 3/5/8 (anchors: control=0,
# treatment=9 already in reports/). Focused on the personalization-responsive
# categories to keep cost bounded.
set -u
cd /home/user/wellness-copilot
PY=/home/user/miniconda3/envs/hga/bin/python

# personalization-responsive subset (25): profile_personalization + multi_turn +
# progress_review + nutrition + training.
SUBSET="personalization_001,personalization_002,personalization_003,personalization_004,\
multi_turn_001,multi_turn_002,multi_turn_003,multi_turn_004,multi_turn_005,\
analyst_001,analyst_002,\
nutrition_001,nutrition_002,nutrition_003,nutrition_004,nutrition_005,nutrition_006,nutrition_007,\
training_001,training_002,training_003,training_004,training_005,training_006,training_007"

export LLM_DISABLE_KEEPALIVE=1
export LLM_REQUEST_TIMEOUT_SEC=60
export EVAL_SAMPLE_TIMEOUT_SEC=240
export EVAL_SEED_EPISODES=1

for N in 3 5 8; do
  echo "############ DEPTH=$N  (gate MIN_COUNT=$N, just-qualified user) ############"
  rm -f /tmp/mincount_${N}.partial.jsonl
  EVAL_SEED_EPISODE_LIMIT=$N \
  EPISODE_SEMANTIC_MIN_COUNT=$N \
  EVAL_PARTIAL_PATH=/tmp/mincount_${N}.partial.jsonl \
  EPISODE_STORE_PATH=/tmp/ep_mc${N}.json \
  EPISODE_INDEX_DIR=/tmp/ep_mc${N}_idx \
    $PY scripts/evaluate_output.py --samples "$SUBSET" --out reports/output_eval_DEPTH${N}.json \
    > /tmp/mincount_${N}.log 2>&1
  echo "depth=$N exit=$?  done=$(wc -l < /tmp/mincount_${N}.partial.jsonl 2>/dev/null || echo 0)/25"
done
echo "MINCOUNT SWEEP DONE"
