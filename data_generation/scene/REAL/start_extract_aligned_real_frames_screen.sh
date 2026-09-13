#!/usr/bin/env bash
set -euo pipefail

SESSION="real_align_extract"
ROOT="/path/to/workspace"
LOG_DIR="/path/to/workspace/SCENEOUTPUT/REAL/aligned_frames_logs"
RUNNER="/path/to/workspace/SCENE/REAL/run_extract_aligned_real_frames_batch.py"

mkdir -p "$LOG_DIR"
chmod +x "$RUNNER"

if screen -list | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "screen session ${SESSION} already exists"
  exit 0
fi

screen -dmS "$SESSION" bash -lc "cd '$ROOT' && python3 '$RUNNER' 2>&1 | tee -a '$LOG_DIR/screen_run.log'"
echo "started screen session: $SESSION"
echo "attach with: screen -r $SESSION"
echo "log file: $LOG_DIR/screen_run.log"
