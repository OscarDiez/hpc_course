"""Real local execution; never claims Slurm/SciTech/GPU validation."""
from pathlib import Path
import array, json, os, shutil, subprocess, tempfile, platform, sys
root=Path(__file__).resolve().parents[1]
checks=[]
def run(cmd,env=None,ok=True):
 p=subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,timeout=45)
 if ok and p.returncode:raise RuntimeError(f'{cmd}\n{p.stdout}\n{p.stderr}')
 return p

def field(path):
 a=array.array('d');a.frombytes(Path(path).read_bytes());return list(a)

def reference(n,steps):
 a=[0.]*(n*n);a[:n]=[100.]*n
 for s in range(steps):
  b=a.copy()
  for y in range(1,n-1):
   for x in range(1,n-1):
    k=y*n+x;b[k]=.25*(a[k-n]+a[k+n]+a[k-1]+a[k+1])
  a=b
 return a

run(['bash','scripts/build.sh','serial']);run(['bash','scripts/build.sh','libraries'])
has_mpi=bool(shutil.which('mpicc') and shutil.which('mpirun'))
if has_mpi:run(['bash','scripts/build.sh','mpi'])
with tempfile.TemporaryDirectory(prefix='m5s1-check-') as tmp:
 tmp=Path(tmp)
 for n,steps in [(8,1),(11,2),(11,31),(17,32),(32,80)]:
  expected=reference(n,steps)
  configs=[('naive',1,1),('serial_O0',1,1),('serial',1,1),('omp',1,1),('omp',1,4)]
  if has_mpi:configs += [('mpi',1,1),('mpi',2,1),('mpi',3,1),('hybrid',2,2),('hybrid',3,2)]
  for mode,p,t in configs:
   out=tmp/f'{n}-{steps}-{mode}-{p}-{t}.bin';env=os.environ.copy()
   env.update(OMP_NUM_THREADS=str(t),OMP_DYNAMIC='FALSE')
   exe=str(root/'bin'/('heat_'+mode))
   cmd=[exe,str(n),str(steps),str(out)]
   if mode in {'mpi','hybrid'}:
    env['LAB_EXPECT_RANKS']=str(p);cmd=['mpirun','-np',str(p)]+cmd
   response=run(cmd,env)
   got=field(out);assert len(got)==n*n
   error=max(abs(a-b) for a,b in zip(got,expected));assert error<=1e-10,(mode,error)
   assert response.stdout.count('RESULT ')==1
   if mode in {'mpi','hybrid'}:
    assert response.stdout.count('MAP rank=')==p
    assert all(f'threads={t}' in line for line in response.stdout.splitlines() if line.startswith('MAP '))
   checks.append(f'PASS N={n} steps={steps} {mode} ranks={p} threads={t} full-field-error={error:g}')
 # Reject invalid arguments and the singleton-launch failure mode.
 assert run([str(root/'bin/heat_serial'),'7','10',str(tmp/'bad.bin')],ok=False).returncode
 checks.append('PASS invalid N rejected')
 if has_mpi:
  env=os.environ.copy();env['LAB_EXPECT_RANKS']='2'
  assert run(['mpirun','-np','1',str(root/'bin/heat_mpi'),'11','31',str(tmp/'bad.bin')],env,ok=False).returncode
  checks.append('PASS unexpected MPI communicator size rejected')
 n=11;steps=31;expected=reference(n,steps)
 ref=tmp/'reference-r0.bin';a=array.array('d',expected);ref.write_bytes(a.tobytes())
 for v in ('v1','v2','fixed'):
  out=tmp/f'library_{v}-r1.bin';env=os.environ.copy();env['LD_LIBRARY_PATH']=str(root/'lib'/v)
  resp=run([str(root/'bin/heat_library'),str(n),str(steps),str(out)],env)
  (tmp/f'library_{v}-r1.log').write_text(resp.stdout)
  cmd=[sys.executable,'scripts/validate.py',str(ref),str(out),str(n)]
  if v=='v2':
   assert run(cmd,ok=False).returncode
   cmd+=['--expect-different']
   got=field(out);assert max(abs(b-3.7315*a) for a,b in zip(expected,got))<1e-10
  run(cmd);checks.append('PASS library '+v+(' expected mismatch and rejection' if v=='v2' else ' agrees with reference'))
 run([sys.executable,'scripts/summarize.py',str(tmp)])
 assert 'EXPECTED_MISMATCH' in (tmp/'measurements.csv').read_text()
 checks.append('PASS CSV revalidation, labels and warmup exclusion')
 # Finiteness check catches corrupted output instead of silently accepting a NaN.
 corrupt=tmp/'nan.bin';a=array.array('d',expected);a[n+1]=float('nan');corrupt.write_bytes(a.tobytes())
 assert run([sys.executable,'scripts/validate.py',str(ref),str(corrupt),str(n)],ok=False).returncode
 checks.append('PASS NaN corruption rejected')
record={'environment':'Local development container; NOT SciTech; NOT Slurm','platform':platform.platform(),
 'gcc':run(['gcc','--version']).stdout.splitlines()[0],
 'mpi':run(['mpichversion']).stdout.splitlines()[0] if has_mpi and shutil.which('mpichversion') else ('available' if has_mpi else 'NOT RUN'),
 'mpi_transport': {k:os.environ.get(k,'site default') for k in ('MPIR_CVAR_CH4_NETMOD','FI_PROVIDER')},
 'gpu':'NOT RUN: no CUDA compiler/device available','slurm':'NOT RUN: no SciTech allocation',
 'checks':checks,'passed':len(checks)}
(root/'tests/local-validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
