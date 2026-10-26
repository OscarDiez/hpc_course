#!/bin/bash
# Usage: bash setup_assignment.sh STUDENT_ID [NEW_WORK_DIRECTORY]
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then echo 'Usage: bash setup_assignment.sh STUDENT_ID [NEW_WORK_DIRECTORY]' >&2; exit 2; fi
DEST=${2:-"$HOME/m4s5_individual"}
if [ -e "$DEST" ]; then echo "Directory already exists: $DEST. Choose a NEW directory; nothing overwritten." >&2; exit 2; fi
mkdir -p "$DEST"
cd "$DEST"
BASE='https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/assignments/practice4_distributed_ml'
for FILE in prepare_assignment.py run_experiments.py student.py; do
    curl -fL "$BASE/$FILE" -o "$FILE"
done
curl -fL 'https://raw.githubusercontent.com/OscarDiez/hpc_course/616f5fbe63e2e0d63fbecbcae0232defd44d72ce/sessions/M4S5_async_distributed_ml/M4S5_async_distributed_ml.ipynb' -o guided_source.ipynb
python3 prepare_assignment.py "$1"
echo "Work directory: $PWD"
