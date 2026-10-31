# Session 26 alternative-paradigm CPU demonstrations

The canonical notebook is [M6S1](../../sessions/M6S1_quantum_neuromorphic_alternative/). It embeds these exact files and generates `settings.json`. No checkout is required for notebook use.

- `m6s1_lab.c`: event updates, discrete neuron, matrix/quantization and one-qubit interference.
- `run_suite.py`: compile, execute, check against independent Python references and save explicit status/provenance.
- `m6s1_cpu.sbatch`: separate one-CPU, 1 GB, two-minute job on the course CPU partition.

For a manual run from a writable copy of this directory:

```bash
python3 -c 'import json; from run_suite import DEFAULTS; print(json.dumps(DEFAULTS))' > settings.json
python3 run_suite.py
```

For SciTech, use the notebook submission cell so inherited conflicting Slurm memory variables are removed before sbatch. A manual submission from JupyterHub must do the same environment cleanup. Use `module load foss/2023b` when the site exposes that course module. The job inspects the inherited GCC when modules are absent. No compiler installation occurs.

These files execute CPU simulations and illustrative models. They do not benchmark alternative computing hardware. Refer to the session validation report before teaching.
