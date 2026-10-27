#!/bin/bash -l
#SBATCH --job-name=m5s1-paraview
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:06:00
#SBATCH --output=paraview-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
out="results/paraview-$SLURM_JOB_ID";mkdir -p "$out"
# Make Environment Modules visible in this non-interactive shell.
if ! type module >/dev/null 2>&1; then
 for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash;do
  if [[ -r "$init" ]];then source "$init";fi
  if type module >/dev/null 2>&1;then break;fi
 done
fi
{
 date -u;hostname;scontrol show job "$SLURM_JOB_ID"
 if type module >/dev/null 2>&1;then module list;fi
} >"$out/environment.txt" 2>&1
if type module >/dev/null 2>&1; then
 # Inspect the site catalogue instead of inventing a ParaView module version.
 module -t avail ParaView paraview >"$out/available-modules.txt" 2>&1 || true
 selected=${PARAVIEW_MODULE:-}
 if [[ -z "$selected" ]];then
  selected=$(python3 - "$out/available-modules.txt" <<'PY'
import re,sys
text=open(sys.argv[1]).read()
items=re.findall(r'(?<!\S)([^\s:]*[Pp][Aa][Rr][Aa][Vv][Ii][Ee][Ww][^\s:()]*)',text)
items=[s for s in items if '/' in s and not s.startswith('/')]
print(sorted(set(items))[-1] if items else '')
PY
)
 fi
 if [[ -n "$selected" ]];then
  echo "STEP module load $selected"
  module load "$selected"
  echo "PARAVIEW_MODULE=$selected" >>"$out/environment.txt"
 fi
 module list >>"$out/environment.txt" 2>&1
fi
# A missing application is SKIPPED; an attempted but failed render is FAIL.
if ! command -v pvpython >/dev/null;then
 echo 'SKIPPED_PARAVIEW: pvpython unavailable; see available-modules.txt and environment.txt'
 exit 0
fi
command -v pvpython | tee -a "$out/environment.txt"
pvpython --version >>"$out/environment.txt" 2>&1
# Runs only after a validated CPU field exists. Both paths are on shared user storage.
python3 scripts/export_vtk.py "$HEAT_INPUT" "$N" "$out/temperature.vtk"
pvpython --force-offscreen-rendering scripts/paraview_heat.py "$out/temperature.vtk" "$N" "$out"
echo "PARAVIEW_DONE $out"
