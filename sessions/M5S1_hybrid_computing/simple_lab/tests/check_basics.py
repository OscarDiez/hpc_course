"""Run the C programs against an independent reference; no cluster claim."""
from pathlib import Path
import subprocess,os,tempfile,math
root=Path(__file__).resolve().parent
if not (root/'heat_serial.c').exists():root=root.parent
checks=0
with tempfile.TemporaryDirectory(prefix='m5s1-basics-check-') as tmp:
    work=Path(tmp);env=os.environ.copy();env['OMP_NUM_THREADS']='2';env['OMP_DYNAMIC']='FALSE'
    def run(args):
        p=subprocess.run(list(map(str,args)),cwd=work,env=env,capture_output=True,text=True,timeout=45)
        assert p.returncode==0,(args,p.stdout,p.stderr)
        return p
    for mode in ['serial','openmp','mpi','hybrid']:
        compiler='mpicc' if mode in ('mpi','hybrid') else 'gcc'
        flags=['-fopenmp'] if mode in ('openmp','hybrid') else []
        run([compiler,'-O2','-Wall','-Wextra']+flags+[root/f'heat_{mode}.c','-o',work/mode])
    # Syntax only. The source refuses CPU fallback; no GPU execution is asserted.
    run(['gcc','-O2','-fopenacc',root/'heat_openacc.c','-o',work/'acc-syntax'])
    for n,steps in [(3,1),(4,2),(8,7),(12,10),(128,150),(512,3)]:
        expected=[0.]*(n*n);expected[:n]=[100.]*n
        for _ in range(steps):
            new=expected[:]
            for y in range(1,n-1):
                for x in range(1,n-1):
                    k=y*n+x;new[k]=.25*(expected[k-n]+expected[k+n]+expected[k-1]+expected[k+1])
            expected=new
        for mode in ['serial','openmp','mpi','hybrid']:
            ranks_to_try=[r for r in [1,2,3] if (n-2)%r==0] if mode in ('mpi','hybrid') else [1]
            for ranks in ranks_to_try:
                path=work/'field.vtk';args=[work/mode,n,steps,path]
                if mode in ('mpi','hybrid'):args=['mpiexec','-n',ranks]+args
                output=run(args)
                lines=path.read_text().splitlines();actual=list(map(float,lines[10:]))
                assert lines[4]==f'DIMENSIONS {n} {n} 1'
                assert len(actual)==len(expected)
                assert all(math.isfinite(x) for x in actual)
                assert max(abs(a-b) for a,b in zip(expected,actual))<=1e-10,(n,steps,mode,ranks)
                if mode in ('mpi','hybrid'):assert f'/{ranks}:' in output.stderr
                checks+=1
    bad=subprocess.run(['mpiexec','-n','3',str(work/'mpi'),'9','2','bad.vtk'],cwd=work,env=env,capture_output=True,timeout=45)
    assert bad.returncode!=0;checks+=1
print(f'PASS {checks} real local serial/OpenMP/MPI/hybrid and invalid-layout checks.')
print('OpenACC syntax compiled; NVIDIA execution, Slurm and ParaView are unverified.')
