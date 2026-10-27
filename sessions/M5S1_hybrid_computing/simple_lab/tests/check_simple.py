"""Local numerical checks; does not claim Slurm, NVIDIA or ParaView execution."""
from pathlib import Path
from array import array
import os,subprocess,tempfile,math,json
root=Path(__file__).resolve().parent
# In the repository this test lives in simple_lab/tests.
if not (root/'heat.c').exists(): root=root.parent

def run(args,env):
    result = subprocess.run(list(map(str,args)),cwd=work,env=env,capture_output=True,text=True,timeout=45)
    if result.returncode: raise RuntimeError(str(args)+'\n'+result.stdout+'\n'+result.stderr)
    return result
def read(path):
    a=array('d');a.frombytes(path.read_bytes());return a
checks=0
with tempfile.TemporaryDirectory(prefix='m5s1-simple-check-') as tmp:
    work=Path(tmp);env=os.environ.copy()
    builds={'naive':['gcc','-O0','-DNAIVE','heat.c'],
            'serial_O0':['gcc','-O0','-DSERIAL_O0','heat.c'],
            'serial':['gcc','-O2','heat.c'],
            'openmp':['gcc','-O2','-fopenmp','heat.c'],
            'mpi':['mpicc','-O2','heat_mpi.c'],
            'hybrid':['mpicc','-O2','-fopenmp','heat_mpi.c']}
    for mode,cmd in builds.items():
        cmd[-1]=str(root/cmd[-1]);run(cmd+['-o',str(work/mode)],env)
    for n,steps in [(3,1),(4,2),(8,7),(12,10),(128,150),(512,3)]:
        expected=[0.]*(n*n);expected[:n]=[100.]*n
        for _ in range(steps):
            new=expected[:]
            for y in range(1,n-1):
                for x in range(1,n-1):
                    k=y*n+x;new[k]=.25*(expected[k-n]+expected[k+n]+expected[k-1]+expected[k+1])
            expected=new
        for mode in builds:
            rank_options=[r for r in (1,2,3) if (n-2)%r==0] if mode in ('mpi','hybrid') else [1]
            for ranks in rank_options:
                path=work/'field.bin';env['OMP_NUM_THREADS']='2'
                args=[work/mode,n,steps,path]
                if mode in ('mpi','hybrid'):args=['mpiexec','-n',ranks]+args
                output=run(args,env);actual=read(path)
                assert len(actual)==len(expected)
                assert all(math.isfinite(v) for v in actual)
                error=max(abs(a-b) for a,b in zip(actual,expected))
                assert error <=1e-10,(mode,n,steps,ranks,error)
                fields=output.stdout.strip().split(',')
                assert fields[0]==mode and int(fields[3])==ranks
                assert int(fields[4])==(2 if mode in ('openmp','hybrid') else 1)
                checks+=1
    bad=subprocess.run(['mpiexec','-n','3',str(work/'mpi'),'9','2','bad.bin'],cwd=work,env=env,capture_output=True,timeout=45)
    assert bad.returncode!=0
    checks+=1
print(f'PASS {checks} numerical/ownership/invalid-layout checks against an independent Python reference.')
print('Actual local C/OpenMP/MPI processes. No Slurm, GPU or ParaView execution claimed.')
