"""Check notebook execution with real payloads and explicit local CLI test doubles.

This does not validate Lmod, Slurm, Apptainer isolation or a real SIF.
Run with nbformat, IPython and CPU PyTorch available to the test interpreter.
"""
from pathlib import Path
import os,sys,tempfile,subprocess,json,ast,shutil
import nbformat
root=Path(__file__).resolve().parents[2]
notebook=root/'M5S3_containers_portability_reproducibility.ipynb'
real_mpicc=shutil.which('mpicc')
assert real_mpicc,'This author test also requires an MPI C compiler'
nb=nbformat.read(notebook,4);nbformat.validate(nb)
markdown='\n'.join(c.source for c in nb.cells if c.cell_type=='markdown')
for c in nb.cells:
    if c.cell_type=='code' and c.source.startswith('run('):
        command=ast.literal_eval(ast.parse(c.source).body[0].value.args[0])
        assert command in markdown
        result=subprocess.run(['bash','-n'],input=command,text=True,capture_output=True)
        assert result.returncode==0,result.stderr
    elif c.cell_type=='code' and not c.source.startswith('%%'):
        ast.parse(c.source)

for mode in ['available','missing_image','incompatible']:
    with tempfile.TemporaryDirectory(prefix='m5s3-basics-test-') as tmp:
        base=Path(tmp);bin=base/'bin';bin.mkdir();(base/'home').mkdir()
        image=base/'course.sif'
        if mode!='missing_image':image.write_text('Local CLI test double, not a SIF image')
        init=base/'init.sh'
        init.write_text('''module() {
 case "$1" in
  save) printf '%s\\n' "$2" > "$QA_ROOT/collection" ;;
  restore) test "$(cat "$QA_ROOT/collection")" = "$2" ;;
  *) echo "Local test double: module $*" >&2 ;;
 esac
}
''')
        commands={
        'bash':'''#!/bin/bash
if [[ "$1" == -lc ]];then exec /bin/bash -c "$2";else exec /bin/bash "$@";fi
''',
        'h5pcc':'#!/bin/bash\necho "local wrapper test double"\n',
        'mpicc':'''#!/usr/bin/env python3
import os,sys
compiler=os.environ['QA_REAL_MPICC']
os.execv(compiler,[compiler]+sys.argv[1:])
''',
        'srun':'''#!/usr/bin/env python3
import os,sys,subprocess,json
args=sys.argv[1:]
two_nodes='-N2' in args
assert 'SLURM_JOB_ID' not in os.environ,'Inherited notebook job leaked into allocation'
with open(os.environ['QA_ROOT']+'/allocations.jsonl','a') as f:f.write(json.dumps(args)+'\\n')
while args and args[0].startswith('-'):
 a=args.pop(0)
 if a in ['-p','-N','-n','-c','-t','--mem','--time']:args.pop(0)
if two_nodes:
 for rank in range(2):
  env=os.environ.copy();env['QA_TWO_NODE_TASK']=str(rank)
  p=subprocess.run(args,env=env,capture_output=True,text=True)
  if p.returncode:print(p.stderr,file=sys.stderr);raise SystemExit(p.returncode)
  print(str(rank)+': '+p.stdout.strip())
 raise SystemExit(0)
raise SystemExit(subprocess.run(args).returncode)
''',
        'salloc':'''#!/usr/bin/env python3
import sys,subprocess
args=sys.argv[1:]
while args and args[0].startswith('-'):
 a=args.pop(0)
 if a in ['-p','-N','-n','-c','-t','--mem','--time']:args.pop(0)
raise SystemExit(subprocess.run(args).returncode)
''',
        'mpirun':'''#!/usr/bin/env python3
import os,sys,subprocess
args=sys.argv[1:]
while args and args[0].startswith('-'):
 a=args.pop(0)
 if a in ['-np','--map-by']:args.pop(0)
for rank in range(2):
 env=os.environ.copy();env['QA_TWO_NODE_TASK']=str(rank)
 p=subprocess.run(args,env=env)
 if p.returncode:raise SystemExit(p.returncode)
''',
        'apptainer':'''#!/usr/bin/env python3
import os,sys,subprocess,json
args=sys.argv[1:]
with open(os.environ['QA_ROOT']+'/container-calls.jsonl','a') as f:f.write(json.dumps(args)+'\\n')
if args[0]=='--version':print('local CLI test double');raise SystemExit(0)
if args[0]=='inspect':print('{}');raise SystemExit(0)
assert args.pop(0)=='exec'
assert '--cleanenv' in args or '/work/mpi_demo' in args
assert '--nv' not in args
host_bind=None
while args[0].startswith('--'):
 a=args.pop(0)
 if a=='--bind':
  binding=args.pop(0)
  if ':/work' in binding:host_bind=binding.split(':')[0]
 elif a=='--env':args.pop(0)
 elif a=='--pwd':assert args.pop(0)=='/work'
args.pop(0)  # the fake image
program=args.pop(0)
if program=='/work/mpi_demo':
 rank=int(os.environ['QA_TWO_NODE_TASK'])
 print(f'MPI rank {rank}/2 on local-test-node-{rank}: sum = 3')
 if rank==0:print('MPI_SUM_PASS: 3')
 raise SystemExit(0)
assert program=='/opt/conda/bin/python3'
if host_bind:os.chdir(host_bind)
if 'QA_TWO_NODE_TASK' in os.environ:
 print('Container on local-test-node-'+os.environ['QA_TWO_NODE_TASK']);raise SystemExit(0)
if os.environ['QA_MODE']=='incompatible':
 print('Deliberately incompatible runtime (local test)',file=sys.stderr);raise SystemExit(3)
raise SystemExit(subprocess.run([sys.executable]+args).returncode)
'''}
        for name,source in commands.items():
            path=bin/name;path.write_text(source);path.chmod(0o755)
        driver=base/'driver.py';driver.write_text('''import os,json
from pathlib import Path
from IPython.core.interactiveshell import InteractiveShell
nb=json.load(open(os.environ['QA_NOTEBOOK']))
shell=InteractiveShell.instance()
failed=False
for c in nb['cells']:
 if c['cell_type']=='code':
  source=c['source'];source=''.join(source) if isinstance(source,list) else source
  source=source.replace('/data/software/containers/pytorch-2.8.0-cuda12.8.sif',os.environ['QA_IMAGE'])
  result=shell.run_cell(source)
  if not result.success:
   assert os.environ['QA_MODE']!='available',(result.error_before_exec,result.error_in_exec)
   failed=True
   break
folder=Path.home()/'hpc-labs/m5s3-basics'
assert 'Sum 1..100000: 5000050000' in (folder/'module_result.txt').read_text()
assert (folder/'compiler-before.txt').read_bytes()==(folder/'compiler-after.txt').read_bytes()
if os.environ['QA_MODE']=='available':
 assert not failed
 a=json.loads((folder/'run_A.json').read_text());b=json.loads((folder/'run_B.json').read_text());c=json.loads((folder/'run_C.json').read_text())
 assert a==b and a!=c
 assert abs(a['learned_slope']-3)<.1 and abs(c['learned_slope']-4)<.1
 assert (folder/'from_container.txt').read_text()=='Hello from inside the container.\\n'
 assert 'status.txt' not in Path(os.environ['QA_NOTEBOOK']).read_text()
else:
 assert failed,'Direct route should report a real error when its prerequisites fail'
 assert not (folder/'run_A.json').exists()
assert os.environ['SLURM_JOB_ID']=='25204'
print('PASS full local notebook scenario:',os.environ['QA_MODE'])
''')
        env=os.environ.copy();env.update(HOME=str(base/'home'),PATH=str(bin)+':'+env['PATH'],BASH_ENV=str(init),QA_ROOT=str(base),QA_MODE=mode,QA_IMAGE=str(image),QA_NOTEBOOK=str(notebook),QA_REAL_MPICC=real_mpicc,SLURM_JOB_ID='25204',SLURM_CPUS_PER_TASK='1')
        result=subprocess.run([sys.executable,str(driver)],env=env,capture_output=True,text=True,timeout=150)
        if result.returncode:
            print(result.stdout[-5000:]);print(result.stderr[-2500:]);raise SystemExit(result.returncode)
        assert 'PASS full local notebook scenario: '+mode in result.stdout
        allocations=[json.loads(line) for line in (base/'allocations.jsonl').read_text().splitlines()]
        assert allocations and all('gpu' not in args for args in allocations)
        print('PASS',mode,'using real C and, when available, real host CPU PyTorch; mocked modules/Slurm/container CLI')
print('PASS syntax and matching SSH cells. No actual Lmod/Slurm/Apptainer/SIF execution claimed.')
