"""Bounded guided experiments. Significant work requires a Slurm batch allocation.
--local is for instructor validation on a local workstation, never a login node.
"""
from pathlib import Path
import argparse, hashlib, json, math, os, shutil, socket, subprocess, time, traceback

ROOT = Path(__file__).resolve().parent

def command(args, env=None, timeout=180):
    p = subprocess.run([str(a) for a in args], text=True, capture_output=True,
                       timeout=timeout, env=env)
    if p.returncode:
        raise RuntimeError(f'{args!r}\nexit={p.returncode}\n{p.stdout}\n{p.stderr}')
    return p.stdout

def equivalent(actual, reference):
    assert (actual['n'], actual['d'], actual['queries']) == (reference['n'], reference['d'], reference['queries'])
    assert len(actual['hits']) == len(reference['hits'])
    for a, b in zip(actual['hits'], reference['hits']):
        assert a['id'] == b['id'], (a, b)
        assert math.isfinite(a['score']) and math.isclose(a['score'], b['score'], rel_tol=1e-10, abs_tol=1e-10)
    assert math.isfinite(actual['seconds']) and actual['seconds'] > 0

def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['cpu', 'hybrid', 'container'])
    p.add_argument('--local', action='store_true')
    p.add_argument('--n', type=int, default=65536)
    p.add_argument('--d', type=int, default=128)
    p.add_argument('--queries', type=int, default=16)
    a = p.parse_args()
    if not a.local and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('NOT RUN: submit the supplied batch script; do not benchmark on a login/Jupyter node.')
    assert 1 <= a.n <= 1000000 and 1 <= a.d <= 512 and 1 <= a.queries <= 128
    assert a.n*a.d*4 <= 256*1024*1024, 'Keep vector storage at or below 256 MiB for the 1 GB allocation'
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    label = f'{a.mode}-{os.environ.get("SLURM_JOB_ID", "local")}-{time.time_ns()}'
    records = []
    meta = dict(mode=a.mode, execution='LOCAL' if a.local else 'SLURM', job_id=os.environ.get('SLURM_JOB_ID'),
                host=socket.gethostname(), source_sha256=hashlib.sha256((ROOT/'search.c').read_bytes()).hexdigest(),
                configuration=dict(n=a.n, d=a.d, queries=a.queries),
                allocation={k: os.environ.get(k) for k in ['SLURM_JOB_NUM_NODES','SLURM_NTASKS','SLURM_CPUS_PER_TASK']})
    try:
        cc = shutil.which('gcc')
        if not cc: raise RuntimeError('gcc missing: load the site GCC module in site_env.sh')
        meta['compiler'] = command([cc, '--version']).splitlines()[0]
        meta['flags'] = '-O3 -std=c11; OpenMP adds -fopenmp; MPI adds -DUSE_MPI'
        command([cc, '-O3','-std=c11','search.c','-lm','-o','search_serial'])
        size = [str(a.n), str(a.d), str(a.queries)]
        def run(exe, ranks=1, threads=1, prefix=None):
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC='FALSE', OMP_PLACES='cores', OMP_PROC_BIND='close')
            cmd = (prefix or []) + [str(exe)] + size
            if not a.local:
                cmd = ['srun','--exact','--exclusive','--nodes=1',f'--ntasks={ranks}',f'--cpus-per-task={threads}','--cpu-bind=cores'] + cmd
            elif ranks > 1:
                launcher = shutil.which('mpiexec')
                if not launcher: raise RuntimeError('local MPI launcher missing')
                cmd = [launcher, '-n', str(ranks)] + cmd
            result = json.loads(command(cmd, env))
            assert result['ranks']==ranks and result['threads']==threads, 'Runtime did not use the requested team size'
            return result, cmd
        reference, _ = run(ROOT/'search_serial')
        if a.mode == 'cpu':
            command([cc,'-O3','-std=c11','-fopenmp','search.c','-lm','-o','search_omp'])
            configurations = [('serial',ROOT/'search_serial',1,1,None)] + [(f'omp-{t}',ROOT/'search_omp',1,t,None) for t in [1,2,4]]
        elif a.mode == 'hybrid':
            mpi = shutil.which('mpicc')
            if not mpi:
                meta.update(status='SKIPPED',reason='mpicc unavailable: load a site-compatible MPI module; CPU measurements remain usable')
                return
            meta['mpi_wrapper'] = command([mpi,'--version']).splitlines()[0]
            command([mpi,'-O3','-std=c11','-fopenmp','-DUSE_MPI','search.c','-lm','-o','search_mpi'])
            configurations = [('mpi-2x2',ROOT/'search_mpi',2,2,None)]
        else:
            image = Path(os.environ.get('M5S5_IMAGE','/nonexistent')).expanduser()
            app = shutil.which('apptainer')
            if not app or not image.is_file():
                meta.update(status='SKIPPED',reason='Set M5S5_IMAGE to an existing trusted C/OpenMP-capable SIF and load Apptainer')
                return
            meta['image_path'] = str(image.resolve())
            h = hashlib.sha256()
            with image.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
            meta['image_sha256'] = h.hexdigest()
            prefix = [app,'exec','--cleanenv','--bind',str(ROOT)+':/work','--pwd','/work',str(image.resolve())]
            # Compile inside the image, rather than assuming host libgomp/glibc compatibility.
            build = prefix + ['gcc','-O3','-std=c11','-fopenmp','/work/search.c','-lm','-o','/work/search_container']
            if not a.local: build = ['srun','--exact','-N1','-n1','-c1'] + build
            probe = subprocess.run(build,text=True,capture_output=True,timeout=60)
            if probe.returncode:
                meta.update(status='SKIPPED',reason='Image lacks a compatible gcc/OpenMP build environment',diagnostic=probe.stderr[-3000:])
                return
            os.environ['APPTAINERENV_OMP_NUM_THREADS']='4'
            os.environ['APPTAINERENV_OMP_DYNAMIC']='FALSE'
            os.environ['APPTAINERENV_OMP_PLACES']='cores'
            os.environ['APPTAINERENV_OMP_PROC_BIND']='close'
            configurations = [('container-omp-4','/work/search_container',1,4,prefix)]
        for name, exe, ranks, threads, prefix in configurations:
            for repeat in range(3):
                result, cmd = run(exe,ranks,threads,prefix)
                equivalent(result,reference)
                record = dict(configuration=name,repeat=repeat+1,status='PASS',command=cmd,**result)
                records.append(record)
                print(json.dumps(record),flush=True)
        meta['status']='PASS'
    except Exception as e:
        meta.update(status='FAIL',reason=str(e),traceback=traceback.format_exc())
        raise
    finally:
        meta['records']=records
        (out/(label+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
        print('RESULT',meta['status'],out/(label+'.json'),meta.get('reason',''),flush=True)

if __name__=='__main__':main()
