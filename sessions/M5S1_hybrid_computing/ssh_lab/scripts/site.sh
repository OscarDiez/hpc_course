# Source this file in SSH and batch shells. The login shell is important on SciTech.
# These module names are documented in the course; check module avail on your login.
# Initialise the module command if this batch shell did not inherit it.
if ! type module >/dev/null 2>&1; then
 for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
  if [[ -r "$init" ]]; then source "$init"; fi
  if type module >/dev/null 2>&1; then break; fi
 done
fi
# CPU and GPU jobs select their own supported software stack.
if [[ ${LAB_DEVICE:-cpu} == gpu ]]; then
 if type module >/dev/null 2>&1; then module load nvhpc/25.7; fi
 command -v nvcc >/dev/null || { echo 'nvcc unavailable: check the site GPU module' >&2; return 2; }
else
 if [[ ${LAB_NEEDS_MPI:-1} == 1 ]]; then
  if type module >/dev/null 2>&1; then module load OpenMPI/4.1.5-GCC-12.3.0; fi
  command -v mpicc >/dev/null || { echo 'mpicc unavailable: check the site MPI module' >&2; return 2; }
 fi
fi
# Keep thread counts fixed and bind threads to nearby allocated cores.
export OMP_DYNAMIC=FALSE OMP_PLACES=cores OMP_PROC_BIND=close
# No hidden fallback to another compiler stack after a failed module load.
