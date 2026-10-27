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
renderer=$(command -v pvbatch || command -v pvpython || true)
if [[ -z "$renderer" ]];then
 echo 'SKIPPED_PARAVIEW: pvbatch/pvpython unavailable; see available-modules.txt and environment.txt'
 exit 0
fi
# pvbatch does not need an interactive window or a notebook display connection.
echo "PARAVIEW_EXECUTABLE=$renderer" | tee -a "$out/environment.txt"
"$renderer" --version >>"$out/environment.txt" 2>&1
# This is a CPU allocation. Use Mesa's CPU renderer instead of opening /dev/dri.
# These settings apply only to this visualisation job, not the CUDA experiment.
export LIBGL_ALWAYS_SOFTWARE=true
export GALLIUM_DRIVER=llvmpipe
export EGL_PLATFORM=surfaceless
export LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
unset DISPLAY WAYLAND_DISPLAY PV_DEBUG_REMOTE_RENDERING
ulimit -c 0
printf 'LIBGL_ALWAYS_SOFTWARE=%s\nGALLIUM_DRIVER=%s\nEGL_PLATFORM=%s\nLP_NUM_THREADS=%s\n' \
 "$LIBGL_ALWAYS_SOFTWARE" "$GALLIUM_DRIVER" "$EGL_PLATFORM" "$LP_NUM_THREADS" >>"$out/environment.txt"
# Runs only after a validated CPU field exists. Both paths are on shared user storage.
python3 scripts/export_vtk.py "$HEAT_INPUT" "$N" "$out/temperature.vtk"
echo 'STEP render ParaView image with Mesa software rendering'
if "$renderer" --force-offscreen-rendering scripts/paraview_heat.py "$out/temperature.vtk" "$N" "$out";then
 echo "PARAVIEW_DONE $out"
else
 code=$?
 echo "PARAVIEW_RENDER_FAILED exit=$code. Inspect environment.txt and this log."
 echo 'The loaded ParaView/Mesa build must support software headless rendering.'
 exit "$code"
fi
