"""Check the notebook wrapper's child environment, without a Slurm server."""
import os,json,sys,shlex
from pathlib import Path
root=Path(__file__).resolve().parent
if not (root/'notebook.ipynb').exists():root=root.parents[1]
path=root/'notebook.ipynb'
if not path.exists():path=root/'M5S1_hybrid_computing.ipynb'
nb=json.loads(path.read_text());helper=next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith('import subprocess\nimport selectors'))
namespace={'os':os};exec(helper,namespace)
injected={'SLURM_JOB_ID':'25204','SLURM_JOBID':'25204','SLURM_STEP_ID':'0','SLURM_CPUS_PER_TASK':'1','SLURM_TRES_PER_TASK':'cpu:1','SLURM_MEM_PER_NODE':'1024','SLURM_CONF':'/tmp/slurm-test-config','SLURM_ACCOUNT':'faculty','SLURM_QOS':'normal'}
before=os.environ.copy()
try:
 os.environ.update(injected)
 check="""import os
for key in ['SLURM_JOB_ID','SLURM_JOBID','SLURM_STEP_ID','SLURM_CPUS_PER_TASK','SLURM_TRES_PER_TASK','SLURM_MEM_PER_NODE']:
 assert key not in os.environ,key
assert os.environ['SLURM_CONF']=='/tmp/slurm-test-config'
assert os.environ['SLURM_ACCOUNT']=='faculty'
assert os.environ['SLURM_QOS']=='normal'
print('PASS child requests a fresh allocation and preserves Slurm config/account/QOS')
"""
 namespace['run'](shlex.quote(sys.executable)+' -c '+shlex.quote(check),show_command=False)
 assert all(os.environ.get(key)==value for key,value in injected.items())
 print('PASS notebook parent allocation environment is unchanged')
finally:
 os.environ.clear();os.environ.update(before)
