#!/bin/zsh
# Resume of run_overnight.sh after the 2026-11-20 morning stall (LoRA step 750, mps memory growth).
# Sweeps, CLIP scoring, settings selection/confirmation and sweep grids were finished by hand
# beforehand. Same log as overnight. Nothing here commits or pushes.
#   nohup caffeinate -i -s zsh scripts/run_resume.sh >/dev/null 2>&1 &
set -u
cd "${0:A:h}/.."
PY=.venv/bin/python
LOG=results/overnight.log
STEPS=4000
RUN=rank8
export SSL_CERT_FILE="$($PY -m certifi)"
note() { echo "=== $(date '+%F %T')  $*" >> "$LOG"; }
run() { local name=$1; shift; note "START $name: $*"; "$@" >> "$LOG" 2>&1; local rc=$?; note "END   $name (exit $rc)"; return $rc; }

note "Resume pipeline started (pid $$); training resumes from the last checkpoint"
run train $PY -m src.train_lora --run $RUN --steps $STEPS \
  || run train_retry_gc $PY -m src.train_lora --run $RUN --steps $STEPS --grad-checkpointing \
  || run train_retry_b2 $PY -m src.train_lora --run $RUN --steps $STEPS --grad-checkpointing --batch 2
LORA=$(ls -d results/lora/$RUN/checkpoints/step* 2>/dev/null | tail -1)
note "LoRA checkpoint for evaluation: ${LORA:-NONE}"
run gen_stock $PY -m src.evaluate generate stock
[[ -n "$LORA" ]] && run gen_lora $PY -m src.evaluate generate lora --lora "$LORA"
run metrics $PY -m src.evaluate metrics
[[ -n "$LORA" ]] && run grids_eval $PY -m src.evaluate grids --lora "$LORA"
note "ALL DONE"
