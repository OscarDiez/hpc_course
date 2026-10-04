#!/bin/bash
#SBATCH --job-name=m3s3_profile
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:12:00
#SBATCH --output=m3s3_profile-%j.out
set -e -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
PYTHON_BIN=$(command -v python3)
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
# Core compiler availability first; profiling modules are optional.
EESSI_INIT=/cvmfs/software.eessi.io/versions/2023.06/init/lmod/bash
if [ -r "$EESSI_INIT" ]; then
    source "$EESSI_INIT"
    module load foss/2023b || true
    module load Score-P/8.4-gompi-2023b || true
    module load CubeLib/4.8.2-GCCcore-13.2.0 || true
fi
printf 'M3S3_JOB job=%s host=%s cpus=%s\n' "${SLURM_JOB_ID:-none}" "$(hostname)" "${SLURM_CPUS_PER_TASK:-unknown}"
lscpu
# The Python runner and all child tools execute inside the same job step.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --cpu-bind=cores "$PYTHON_BIN" m3s3_run_lab.py settings.json
