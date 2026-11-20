#!/bin/zsh
# Unattended overnight pipeline (2026-11-20). Runs every remaining stage in order and writes a
# timestamped record to results/overnight.log. Nothing here commits or pushes.
#
# Launch (keeps the Mac awake while it runs; needs AC power and the lid open):
#   nohup caffeinate -i -s zsh scripts/run_overnight.sh >/dev/null 2>&1 &
set -u
cd "${0:A:h}/.."
PY=.venv/bin/python
LOG=results/overnight.log
STEPS=4000
RUN=rank8
mkdir -p results
# python.org Python on macOS lacks system CA certs; use certifi's bundle for any download
export SSL_CERT_FILE="$($PY -m certifi)"

note() { echo "=== $(date '+%F %T')  $*" >> "$LOG"; }
run() {
  local name=$1; shift
  note "START $name: $*"
  "$@" >> "$LOG" 2>&1
  local rc=$?
  note "END   $name (exit $rc)"
  return $rc
}

note "Overnight pipeline started (pid $$)"

# 1. let the guidance / sampler / steps sweeps finish
while pgrep -f "src.sweep all" >/dev/null; do sleep 60; done
note "Sweeps finished"

# 2. score, choose best settings by the agreed rule, confirm the combination
run clip_score     $PY -m src.clip_score run_log
run select         $PY -m src.select_settings
run best_combo     $PY -m src.sweep best
run clip_score2    $PY -m src.clip_score run_log
run confirm        $PY -m src.select_settings confirm
for s in baseline guidance sampler steps; do run "grid_$s" $PY -m src.grids $s; done

# 3. LoRA training (resumes from the last checkpoint on retry; fallbacks save memory)
run train $PY -m src.train_lora --run $RUN --steps $STEPS \
  || run train_retry_gc $PY -m src.train_lora --run $RUN --steps $STEPS --grad-checkpointing \
  || run train_retry_b2 $PY -m src.train_lora --run $RUN --steps $STEPS --grad-checkpointing --batch 2
LORA=$(ls -d results/lora/$RUN/checkpoints/step* 2>/dev/null | tail -1)
note "LoRA checkpoint for evaluation: ${LORA:-NONE}"

# 4. 1,000 held-out images per condition at the best settings, then metrics and grids
run gen_stock $PY -m src.evaluate generate stock
if [[ -n "$LORA" ]]; then
  run gen_lora $PY -m src.evaluate generate lora --lora "$LORA"
fi
run metrics $PY -m src.evaluate metrics
if [[ -n "$LORA" ]]; then
  run grids_eval $PY -m src.evaluate grids --lora "$LORA"
fi

note "ALL DONE"
