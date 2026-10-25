#!/bin/bash
# Instructor preflight: extracts and runs the exact student-notebook assets.
set -euo pipefail
TASK_DIR=${M4S5_VALIDATION_DIR:-"$HOME/m4s5-validation"}
mkdir -p "$TASK_DIR"
cd "$TASK_DIR"
curl -fL 'https://raw.githubusercontent.com/OscarDiez/hpc_course/teaching/m4s5-distributed-ml-2026/sessions/M4S5_async_distributed_ml/M4S5_async_distributed_ml.ipynb' -o M4S5_async_distributed_ml.ipynb
python3 - <<'PY'
import hashlib,json,os,pathlib,subprocess
p=pathlib.Path('M4S5_async_distributed_ml.ipynb')
n=json.loads(p.read_text());namespace={}
for c in n['cells']:
    if c['cell_type']=='code' and set(c.get('metadata',{}).get('tags',[])) & {'setup','assets','write-files'}:
        exec(compile(''.join(c['source']),'<notebook asset cell>','exec'),namespace)
r=namespace['RUN_DIR'];registry=r/'submission.json'
if registry.exists():
    job=json.loads(registry.read_text())['job_id'];print('Reusing recorded job',job)
else:
    env=os.environ.copy()
    for k in ('SLURM_MEM_PER_CPU','SLURM_MEM_PER_GPU','SLURM_MEM_PER_NODE'): env.pop(k,None)
    out=subprocess.run(['sbatch','--parsable','run.sbatch'],cwd=r,env=env,text=True,capture_output=True,check=True)
    job=out.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unexpected sbatch response '+out.stdout)
    registry.write_text(json.dumps({'job_id':job,'fingerprint':namespace['fingerprint']},indent=2))
print('Notebook SHA-256:',hashlib.sha256(p.read_bytes()).hexdigest())
print('Submitted/recorded job:',job,'(submission is not PASS)')
print('After the job finishes, run:')
print('sacct -j '+job+' --format=JobID,State,ExitCode,Elapsed')
print('cat '+str(r/'SUMMARY.txt'))
print('Log: '+str(r/('slurm-'+job+'.out')))
print('Report: '+str(r/'report.json'))
PY
