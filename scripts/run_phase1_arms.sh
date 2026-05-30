#!/usr/bin/env bash
# Phase 1 of todos/rag-diagnosis-plan.md: paired two-arm output eval.
#   CONTROL  : no episodes seeded (status quo) -> episode_context EMPTY
#   TREATMENT: EVAL_SEED_EPISODES=1            -> episodic RAG fires
# Same 53 samples, same judge, back-to-back to control judge nondeterminism.
set -u
cd /home/user/wellness-copilot
PY=/home/user/miniconda3/envs/hga/bin/python
TS=$(date +%Y%m%d-%H%M%S)

# Resilience against the flaky upstream proxy:
#   - disable HTTP keep-alive so a stale pooled socket can't wedge the next call
#   - per-request timeout so a single call fails fast and retries
#   - per-sample wall-clock watchdog as a backstop
export LLM_DISABLE_KEEPALIVE=1
export LLM_REQUEST_TIMEOUT_SEC=60
export EVAL_SAMPLE_TIMEOUT_SEC=240

echo "############ CONTROL arm (no episodes) ############"
rm -f /tmp/phase1_control.partial.jsonl
EVAL_PARTIAL_PATH=/tmp/phase1_control.partial.jsonl \
EPISODE_STORE_PATH=/tmp/ep_ctrl_${TS}.json \
EPISODE_INDEX_DIR=/tmp/ep_ctrl_idx_${TS} \
  $PY scripts/evaluate_output.py --out reports/output_eval_CONTROL.json \
  > /tmp/phase1_control.log 2>&1
echo "control exit=$?"

echo "############ TREATMENT arm (seeded episodes) ############"
rm -f /tmp/phase1_treatment.partial.jsonl
EVAL_SEED_EPISODES=1 \
EVAL_PARTIAL_PATH=/tmp/phase1_treatment.partial.jsonl \
EPISODE_STORE_PATH=/tmp/ep_treat_${TS}.json \
EPISODE_INDEX_DIR=/tmp/ep_treat_idx_${TS} \
  $PY scripts/evaluate_output.py --out reports/output_eval_SEEDED.json \
  > /tmp/phase1_treatment.log 2>&1
echo "treatment exit=$?"

echo "ALL ARMS DONE"
echo "control  report: $(grep 'Report →' /tmp/phase1_control.log | tail -1)"
echo "treatment report: $(grep 'Report →' /tmp/phase1_treatment.log | tail -1)"
