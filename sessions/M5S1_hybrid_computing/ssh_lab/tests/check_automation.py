"""Run the notebook from a clean namespace using a local scheduler harness.

Actual C CPU/MPI/OpenMP code executes locally. The Slurm interface is simulated;
GPU work is explicitly not executed. This is not SciTech acceptance evidence.
Requires IPython, a C compiler, mpicc and mpirun; no NumPy/Matplotlib needed.
"""
from pathlib import Path
import contextlib,io,json,os,shutil,subprocess,tempfile,sys,unittest.mock as mock
root=Path(__file__).resolve().parents[1]
notebook=root.parent/'M5S1_hybrid_computing.ipynb'
real_mpicc=shutil.which('mpicc');real_mpirun=shutil.which('mpirun')
if not real_mpicc or not real_mpirun:raise SystemExit('MPI compiler/launcher required for this local test')
with tempfile.TemporaryDirectory(prefix='m5s1-auto-test-') as tmp:
 tmp=Path(tmp);shim=tmp/'shim';shim.mkdir();counter=tmp/'counter'
 scripts={
 'mpicc':'''#!/bin/bash
if [[ ${1:-} == --showme ]];then shift;exec "$REAL_MPICC" -show "$@";fi
exec "$REAL_MPICC" "$@"
''',
 'scontrol':'#!/bin/bash\necho "LOCAL TEST scheduler interface"\n',
 'srun':'''#!/bin/bash
ranks=1
while [[ $1 == -* ]];do
 case "$1" in --ntasks=*)ranks=${1#*=};;esac
 shift
done
case "$1" in *heat_mpi|*heat_hybrid)exec "$REAL_MPIRUN" -np "$ranks" "$@";; *)exec "$@";;esac
''',
 'sbatch':'''#!/usr/bin/env python3
from pathlib import Path
import os,subprocess
counter=Path(os.environ['LOCAL_TEST_COUNTER'])
job=int(counter.read_text())+1 if counter.exists() else 1000
counter.write_text(str(job))
script=Path(__import__('sys').argv[-1]);device=script.stem
env=os.environ.copy();env.update(SLURM_SUBMIT_DIR=str(Path.cwd()),SLURM_JOB_ID=str(job))
with open(f'{device}-{job}.out','w') as out:
 if device=='gpu':
  out.write('LOCAL TEST: GPU not executed\\n');rc=1
 else:
  rc=subprocess.run(['bash',str(script)],stdout=out,stderr=subprocess.STDOUT,env=env).returncode
Path(os.environ['LOCAL_TEST_STATE_DIR'],str(job)).write_text('FAILED|1:0' if rc else 'COMPLETED|0:0')
print(job)
''',
 'squeue':'#!/bin/bash\nexit 0\n',
 'sacct':'''#!/usr/bin/env python3
from pathlib import Path
import os,sys
job=sys.argv[sys.argv.index('-j')+1]
print(Path(os.environ['LOCAL_TEST_STATE_DIR'],job).read_text())
'''}
 for name,s in scripts.items():p=shim/name;p.write_text(s);p.chmod(0o755)
 states=tmp/'states';states.mkdir()
 env=dict(os.environ,PATH=str(shim)+':'+os.environ['PATH'],REAL_MPICC=real_mpicc,REAL_MPIRUN=real_mpirun,
          LOCAL_TEST_COUNTER=str(counter),LOCAL_TEST_STATE_DIR=str(states))
 import IPython.display
 ns={};output=io.StringIO();displays=[]
 j=json.loads(notebook.read_text())
 with mock.patch.dict(os.environ,env),mock.patch.object(Path,'home',return_value=tmp/'home'),mock.patch.object(IPython.display,'display',side_effect=displays.append),contextlib.redirect_stdout(output):
  for c in j['cells']:
   if c['cell_type']!='code':continue
   source=''.join(c['source'])
   if 'EXPERIMENT_N = 128' in source:
    source=source.replace('EXPERIMENT_N = 128','EXPERIMENT_N = 11').replace('EXPERIMENT_STEPS = 150','EXPERIMENT_STEPS = 31').replace('EXPERIMENT_REPEATS = 2','EXPERIMENT_REPEATS = 1')
   exec(compile(source,'notebook-cell','exec'),ns)
  report=ns['GUIDE_REPORT']
  Path('/tmp/m5s1-runall-test.log').write_text(output.getvalue())
  if report[0]['status']!='PASS':
   for f in list(Path(report[0]['run_dir']).rglob('*.log'))+list(Path(report[0]['run_dir']).rglob('environment.txt')):
    Path('/tmp/m5s1-runall-test.log').open('a').write('\n'+str(f)+'\n'+f.read_text())
  assert [r['status'] for r in report]==['PASS','SKIPPED','FAIL','PASS'],report
  assert 'CHECKPOINT PASS hybrid' in output.getvalue()
  # A repeated call must reuse the job, without a second sbatch or source mutation.
  initial=counter.read_text()
  again=ns['automate'].execute(ns['GUIDE_ROOT'],'cpu',ns['GUIDE_SETTINGS'],emit=lambda x:None)
  assert again['status']=='PASS' and counter.read_text()==initial
  assert all(Path(p).exists() for p in report[0]['plots'])
  # Controllers must not misclassify active, timed-out or failed jobs as a pass.
  settings=dict(ns['GUIDE_SETTINGS'],wait_seconds=0,poll_seconds=.001)
  module=ns['automate']
  with mock.patch.object(module,'state',return_value=('PENDING','Resources',None)):
   pending=module.execute(ns['GUIDE_ROOT'],'cpu',settings,emit=lambda x:None)
   assert pending['status']=='WAIT LIMIT'
  with mock.patch.object(module,'state',side_effect=[('PENDING','Resources',None),('COMPLETED','accounting','0:0')]):
   resumed=module.execute(ns['GUIDE_ROOT'],'cpu',dict(settings,wait_seconds=1),emit=lambda x:None)
   assert resumed['status']=='PASS'
  with mock.patch.object(module,'state',return_value=('FAILED','accounting','1:0')):
   failed=module.execute(ns['GUIDE_ROOT'],'cpu',settings,emit=lambda x:None)
   assert failed['status']=='FAIL'
  unknown_root=tmp/'uncertain';unknown_root.mkdir()
  (unknown_root/'automation-cpu.json').write_text(json.dumps({'device':'cpu','job_id':None,'run_dir':'uncertain'}))
  uncertain=module.execute(unknown_root,'cpu',settings,emit=lambda x:None)
  assert uncertain['status']=='UNKNOWN' and counter.read_text()==initial
 # Parse every generated chart as XML, independent of a browser renderer.
 import xml.etree.ElementTree as ET
 for path in report[0]['plots']:ET.fromstring(Path(path).read_text())
 Path('/tmp/m5s1-runall-test.log').write_text(output.getvalue())
 shutil.copy(report[0]['plots'][0],'/tmp/m5s1-actual-heat.svg')
 print('PASS clean Run All: actual CPU, MPI, OpenMP, libraries and gprof; GPU explicitly failed in harness')
 print('PASS automatic collection, SVG generation and persistent report')
 print('PASS duplicate prevention, queue wait/resume, failed accounting and ambiguous submission')
 print('LOCAL SCHEDULER HARNESS ONLY; SciTech and GPU are not tested')
