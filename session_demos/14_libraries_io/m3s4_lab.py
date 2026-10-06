"""M3S4 measured exercises. All files stay inside a unique, user-owned run directory."""
from pathlib import Path
import argparse, ctypes as ct, ctypes.util, hashlib, json, math, os, platform
import shlex, shutil, statistics, subprocess, sys, tempfile, time, uuid
import numpy as np
BUILD = 'M3S4-2026-10-06-v8'
DEFAULTS = dict(sizes=[128,256,512,1024], repeats=5, fft_batch=500, seed=2026, input_seed=12345,
                files=200, bytes_per_file=4096, fsync=False, heat_points=128,
                heat_steps=100, checkpoint_interval=10, failure_step=57,
                noise_sigma=0.01, input_elements=50000, mpi_ranks=4, mpi_block=1024,
                weather_shape=[4,32,32], weather_slice=2, compression=None, blas_sizes=[128,256,512], blas_threads=1, lapack_points=128,
                rod_left=80.0, rod_right=20.0, rod_source=10.0)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def save(path,data):
    tmp=Path(str(path)+'.tmp'); tmp.write_text(json.dumps(data,indent=2)); tmp.replace(path)

def checked(argv,cwd,timeout=120,env=None):
    p=subprocess.run([str(x) for x in argv],cwd=cwd,text=True,capture_output=True,timeout=timeout,env=env)
    if p.returncode: raise RuntimeError(f'{argv}: rc={p.returncode}\n{p.stdout}\n{p.stderr}')
    return p.stdout

def validate(s):
    if not s['blas_sizes'] or any(type(n)is not int or not 16<=n<=1024 for n in s['blas_sizes']): raise ValueError('blas_sizes: integers 16..1024')
    if type(s['blas_threads']) is not int or not 1<=s['blas_threads']<=8: raise ValueError('blas_threads: integer 1..8')
    if type(s['lapack_points']) is not int or not 2<=s['lapack_points']<=512: raise ValueError('lapack_points: integer 2..512')
    if any(not math.isfinite(s[k]) for k in ['rod_left','rod_right','rod_source']): raise ValueError('rod parameters must be finite')
    for k in ['repeats','fft_batch','files','bytes_per_file','heat_points','heat_steps',
              'checkpoint_interval','input_elements','mpi_ranks','mpi_block']:
        if type(s[k]) is not int or s[k]<1: raise ValueError(f'{k} must be a positive integer')
    if not s['sizes'] or any(type(n)is not int or not 16<=n<=4096 for n in s['sizes']): raise ValueError('sizes: integers 16..4096')
    if s['heat_points']<3 or not 1<=s['failure_step']<s['heat_steps']: raise ValueError('invalid heat grid/failure step')
    if s['mpi_ranks']>64 or s['mpi_block']>1048576: raise ValueError('MPI classroom limits exceeded')
    if len(s['weather_shape'])!=3 or any(type(n)is not int or n<1 for n in s['weather_shape']): raise ValueError('weather_shape must have three positive dimensions')
    if not 0<=s['weather_slice']<s['weather_shape'][0]: raise ValueError('weather_slice outside time dimension')
    if s['compression'] not in [None,'gzip']: raise ValueError('compression must be None or gzip')
    if s['noise_sigma']<0 or not math.isfinite(s['noise_sigma']): raise ValueError('invalid noise_sigma')
    for k in ['seed','input_seed']:
        if type(s[k])is not int or s[k]<0: raise ValueError(k+' must be nonnegative integer')
    if s['files']>5000 or s['repeats']>20 or s['heat_steps']>10000 or s['fft_batch']>100000 or s['input_elements']>2_000_000: raise ValueError('classroom work limit exceeded')
    if s['files']*s['bytes_per_file']>64*1024*1024 or np.prod(s['weather_shape'])>2_000_000: raise ValueError('classroom data limit exceeded')

def summary(samples):
    return dict(samples_s=samples,median_s=statistics.median(samples),min_s=min(samples),max_s=max(samples))

def fft_lab(s,root):
    # Both direct DFT and FFTW execute compiled code through ctypes. Planning is separate.
    compiler=shutil.which('gcc'); libname=ctypes.util.find_library('fftw3') or ('libfftw3.so' if os.getenv('EBROOTFFTW') else None)
    if not compiler or not libname: return dict(status='SKIPPED',reason='gcc or FFTW3 shared library unavailable')
    source=Path(__file__).with_name('m3s4_dft.c'); binary=root/'direct_dft.so'
    command=[compiler,'-O3','-Wall','-Wextra','-Werror','-fPIC','-shared',str(source),'-lm','-o',str(binary)]
    checked(command,root)
    direct=ct.CDLL(str(binary)); direct.direct_dft.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p]; direct.direct_dft.restype=None
    fftw=ct.CDLL(libname)
    fftw.fftw_plan_dft_r2c_1d.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_uint]; fftw.fftw_plan_dft_r2c_1d.restype=ct.c_void_p
    fftw.fftw_execute.argtypes=[ct.c_void_p]; fftw.fftw_execute.restype=None
    fftw.fftw_destroy_plan.argtypes=[ct.c_void_p]; fftw.fftw_destroy_plan.restype=None
    rows=[]
    for n in s['sizes']:
        j=np.arange(n); x=np.ascontiguousarray(np.sin(2*np.pi*3*j/n)+0.5*np.sin(2*np.pi*7*j/n),dtype=np.float64)
        ref=np.empty(n,dtype=np.complex128); spectrum=np.empty(n//2+1,dtype=np.complex128)
        t=time.perf_counter(); plan=fftw.fftw_plan_dft_r2c_1d(n,x.ctypes.data,spectrum.ctypes.data,64) # FFTW_ESTIMATE
        planning=time.perf_counter()-t
        if not plan: raise RuntimeError('FFTW planning failed')
        try:
            def dft(): direct.direct_dft(n,x.ctypes.data,ref.ctypes.data)
            def fft():
                for _ in range(s['fft_batch']): fftw.fftw_execute(plan)
            dft(); fft() # warmup, excluded
            assert np.allclose(ref[:n//2+1],spectrum,rtol=1e-9,atol=1e-8)
            assert np.allclose(spectrum,np.fft.rfft(x),rtol=1e-9,atol=1e-8)
            ds=[]; fs=[]
            for r in range(s['repeats']):
                for label,fn in ([('d',dft),('f',fft)] if r%2==0 else [('f',fft),('d',dft)]):
                    t=time.perf_counter(); fn(); elapsed=time.perf_counter()-t
                    (ds if label=='d' else fs).append(elapsed if label=='d' else elapsed/s['fft_batch'])
            error=float(np.max(np.abs(ref[:n//2+1]-spectrum)))
            assert np.allclose(ref[:n//2+1],spectrum,rtol=1e-9,atol=1e-8)
            np.savez(root/f'spectrum_{n}.npz',signal=x,spectrum=spectrum)
            rows.append(dict(n=n,dft=summary(ds),fftw=summary(fs),planning_s=planning,max_error=error,
                             strongest_bins=np.argsort(np.abs(spectrum))[-2:][::-1].tolist()))
        finally: fftw.fftw_destroy_plan(plan)
    return dict(status='PASS',rows=rows,compiler=checked([compiler,'--version'],root).splitlines()[0],
                compiler_command=command,fftw_library=libname,fftw_version=ct.string_at(ct.addressof((ct.c_char*1).in_dll(fftw,'fftw_version'))).decode(),
                note='Serial. Forward unnormalized transforms; compare N/2+1 real-input bins. FFT timing includes amortized Python-call overhead, excludes planning.')

def io_lab(s,root):
    # Distinct records expose missing, reordered or corrupt content.
    payloads=[bytes([i%251])*s['bytes_per_file'] for i in range(s['files'])]
    expected=hashlib.sha256(b''.join(payloads)).hexdigest(); samples={'many':[],'one':[]}
    for r in range(s['repeats']):
        for case in (['many','one'] if r%2==0 else ['one','many']):
            with tempfile.TemporaryDirectory(prefix=f'io-{case}-',dir=root) as tmp:
                d=Path(tmp); t=time.perf_counter()
                if case=='many':
                    for i,payload in enumerate(payloads):
                        with open(d/f'{i:06d}.bin','wb') as f:
                            f.write(payload)
                            if s['fsync']: f.flush(); os.fsync(f.fileno())
                    samples[case].append(time.perf_counter()-t)
                    paths=sorted(d.glob('*.bin'))
                else:
                    with open(d/'all.bin','wb') as f:
                        for payload in payloads: f.write(payload)
                        if s['fsync']: f.flush(); os.fsync(f.fileno())
                    samples[case].append(time.perf_counter()-t)
                    paths=[d/'all.bin']
                assert len(paths)==(s['files'] if case=='many' else 1)
                assert sum(p.stat().st_size for p in paths)==s['files']*s['bytes_per_file']
                h=hashlib.sha256()
                for p in paths: h.update(p.read_bytes())
                assert h.hexdigest()==expected
    return dict(status='PASS',many=summary(samples['many']),one=summary(samples['one']),bytes=s['files']*s['bytes_per_file'],
                payload_sha256=expected,path=str(root),fsync=s['fsync'],
                note='Single writer, same ordered payload, creation/writes/close timed; directory creation, readback and removal excluded. Warm page cache; no physical disk bandwidth claim. fsync=False does not measure durable checkpoint cost.')

def heat_step(a):
    out=a.copy(); out[1:-1]=a[1:-1]+0.2*(a[:-2]-2*a[1:-1]+a[2:]); return out

def heat_child(config,root,mode):
    s=json.loads(Path(config).read_text()); root=Path(root); n=s['heat_points']
    identity={'points':n,'alpha':0.2,'steps':s['heat_steps']}; cp=root/'checkpoint.json'; start=0
    a=np.sin(np.linspace(0,np.pi,n)); a[0]=a[-1]=0
    if mode=='resume':
        state=json.loads(cp.read_text()); assert state['schema']==1 and state['parameters']==identity
        a=np.array(state['grid'],dtype=float); start=state['step']
        assert a.shape==(n,) and np.all(np.isfinite(a)) and 0<=start<=s['heat_steps']
        assert hashlib.sha256(a.tobytes()).hexdigest()==state['grid_sha256']
    writes=0; checkpoint_s=0.0
    for step in range(start+1,s['heat_steps']+1):
        a=heat_step(a)
        if mode=='fail' and step==s['failure_step']:
            save(root/'failure.json',{'failure_step':step,'last_checkpoint':(step-1)//s['checkpoint_interval']*s['checkpoint_interval'],'checkpoint_writes':writes,'checkpoint_seconds':checkpoint_s})
            os._exit(17) # controlled process termination; latest complete checkpoint survives
        if mode=='fail' and step%s['checkpoint_interval']==0:
            t=time.perf_counter(); save(cp,{'schema':1,'step':step,'parameters':identity,'grid':a.tolist(),'grid_sha256':hashlib.sha256(a.tobytes()).hexdigest()})
            checkpoint_s+=time.perf_counter()-t; writes+=1
    np.save(root/(mode+'_final.npy'),a)
    save(root/(mode+'_stats.json'),{'start_step':start,'checkpoint_writes':writes,'checkpoint_seconds':checkpoint_s})

def checkpoint_lab(s,root,config):
    cmd=[sys.executable,str(Path(__file__).resolve()),'--heat-child',str(config),str(root)]
    checked(cmd+['reference'],root)
    ref=np.load(root/'reference_final.npy').copy()
    p=subprocess.run(cmd+['fail'],cwd=root,capture_output=True,text=True,timeout=120)
    assert p.returncode==17, p.stderr
    cp=root/'checkpoint.json'; loss=json.loads((root/'failure.json').read_text())
    if cp.exists():
        checked(cmd+['resume'],root); restarted=root/'resume_final.npy'; resumed=json.loads((root/'resume_stats.json').read_text())['start_step']
    else:
        # Failure before first checkpoint: restart from initial state, explicitly recorded.
        checked(cmd+['reference'],root); restarted=root/'reference_final.npy'; resumed=0
    actual=np.load(restarted)
    assert np.array_equal(ref,actual)
    return dict(status='PASS',failure_exit_code=17,failure_step=s['failure_step'],resumed_from_step=resumed,
                completed_steps_lost=s['failure_step']-1-resumed,final_identical=True,
                checkpoint_bytes=cp.stat().st_size if cp.exists() else 0,checkpoint_writes=loss['checkpoint_writes'],checkpoint_seconds=loss['checkpoint_seconds'],
                note='Application checkpoint: step, parameters, full grid and integrity hash. Separate processes demonstrate restart. Atomic rename protects incomplete files; this classroom version does not fsync or promise power-loss durability.')

def linalg_reference(root):
    compiler=shutil.which('gcc')
    if not compiler: return None, None
    cmd=[compiler,'-O3','-march=native','-Wall','-Wextra','-Werror','-shared','-fPIC',
         str(Path(__file__).with_name('m3s4_linalg.c')),'-o',str(root/'manual_linalg.so')]
    checked(cmd,root); lib=ct.CDLL(str(root/'manual_linalg.so'))
    for name in ['matmul_ijk','matmul_ikj']:
        f=getattr(lib,name); f.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p]; f.restype=None
    lib.gaussian.argtypes=[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_int];lib.gaussian.restype=ct.c_int
    return lib,cmd

def library_candidates(package,envroot):
    paths=[]
    if os.getenv(envroot):
        root=Path(os.environ[envroot])
        for folder in ['lib','lib64']:
            paths.extend(str(p) for p in sorted((root/folder).glob('lib'+package+'.so*')))
    found=ctypes.util.find_library(package)
    if found: paths.append(found)
    return list(dict.fromkeys(paths))

def load_linalg(symbol,lapack=False):
    errors=[]
    packages=[('openblas','EBROOTOPENBLAS')]
    if lapack: packages.append(('lapack','EBROOTLAPACK'))
    else: packages.append(('blas','EBROOTBLAS'))
    for package,env in packages:
        for path in library_candidates(package,env):
            try:
                lib=ct.CDLL(path)
                if hasattr(lib,'openblas_get_config'):
                    lib.openblas_get_config.restype=ct.c_char_p
                    config=lib.openblas_get_config().decode()
                    if 'USE64BITINT' in config:
                        errors.append(path+': ILP64 not supported by this LP64 example');continue
                else: config='System LP64 '+package+'; implementation/version not reported'
                getattr(lib,symbol)
                return lib,path,config
            except (OSError,AttributeError) as e: errors.append(path+': '+str(e))
    return None,None,'; '.join(errors) or 'No suitable shared library found'

def blas_lab(s,root):
    manual,cmd=linalg_reference(root)
    if manual is None: return dict(status='SKIPPED',reason='gcc unavailable')
    lib,path,config=load_linalg('cblas_dgemm')
    if lib is None: return dict(status='SKIPPED',reason=config)
    f=lib.cblas_dgemm
    f.argtypes=[ct.c_int]*6+[ct.c_double,ct.c_void_p,ct.c_int,ct.c_void_p,ct.c_int,ct.c_double,ct.c_void_p,ct.c_int]
    f.restype=None
    requested=s['blas_threads']; setter=getattr(lib,'openblas_set_num_threads',None)
    getter=getattr(lib,'openblas_get_num_threads',None)
    if setter: setter.argtypes=[ct.c_int];setter.restype=None
    if getter: getter.restype=ct.c_int
    if requested>1 and (setter is None or getter is None):
        return dict(status='SKIPPED',reason='Thread experiment requires OpenBLAS thread controls; use blas_threads=1 for system BLAS')
    available=int(os.getenv('SLURM_CPUS_PER_TASK',str(len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else 1)))
    if requested>available: raise ValueError('blas_threads exceeds allocated/available CPUs')
    rows=[];rng=np.random.default_rng(s['seed'])
    try:
        for n in s['blas_sizes']:
            # Same seeded dense inputs, both manual baselines in compiled C.
            a=rng.normal(size=(n,n));b=rng.normal(size=(n,n));c=np.empty_like(a)
            def product(): f(101,111,111,n,n,n,1.0,a.ctypes.data,n,b.ctypes.data,n,0.0,c.ctypes.data,n)
            if setter: setter(1)
            if getter and getter()!=1: raise RuntimeError('Could not enforce single-thread BLAS')
            product(); reference=c.copy();samples={'manual_ijk':[],'manual_ikj':[],'blas_1':[]};errors={}
            funcs={'manual_ijk':lambda:manual.matmul_ijk(n,a.ctypes.data,b.ctypes.data,c.ctypes.data),
                   'manual_ikj':lambda:manual.matmul_ikj(n,a.ctypes.data,b.ctypes.data,c.ctypes.data),'blas_1':product}
            for name,fn in funcs.items():
                fn();assert np.allclose(c,reference,rtol=1e-10,atol=1e-10),name+' result mismatch'
                errors[name]=float(np.max(np.abs(c-reference)))
            # Rotate order; validation/allocations/random numbers are outside timing.
            names=list(funcs)
            for r in range(s['repeats']):
                for name in names[r%3:]+names[:r%3]:
                    start=time.perf_counter();funcs[name]();samples[name].append(time.perf_counter()-start)
                    assert np.allclose(c,reference,rtol=1e-10,atol=1e-10)
            actual=1 if getter else None
            if requested>1:
                setter(requested);actual=getter()
                if actual!=requested: raise RuntimeError(f'OpenBLAS configured {actual}, requested {requested}')
                product();samples['blas_requested']=[]
                for _ in range(s['repeats']):
                    start=time.perf_counter();product();samples['blas_requested'].append(time.perf_counter()-start)
                    assert np.allclose(c,reference,rtol=1e-10,atol=1e-10)
            rows.append(dict(n=n,timings={k:summary(v) for k,v in samples.items()},max_abs_errors=errors,
                             requested_threads=requested,configured_threads=actual,
                             blas_speedup_over_ijk=statistics.median(samples['manual_ijk'])/statistics.median(samples['blas_1']),
                             blas_speedup_over_ikj=statistics.median(samples['manual_ikj'])/statistics.median(samples['blas_1'])))
    finally:
        if setter: setter(1)
    return dict(status='PASS',library=path,configuration=config,compile_command=cmd,rows=rows,
                note='Dense C=A B can represent applying many sensor/calibration transforms. All methods compiled; warmup and repeated timings. CBLAS row-major, no transpose, alpha=1 beta=0. Thread getter reports configured threads, not measured active workers. System BLAS may be unoptimized; speedup is measured, never guaranteed.')

def lapack_lab(s,root):
    manual,cmd=linalg_reference(root)
    if manual is None: return dict(status='SKIPPED',reason='gcc unavailable')
    lib,path,config=load_linalg('dgesv_',lapack=True)
    if lib is None: return dict(status='SKIPPED',reason=config)
    setter=getattr(lib,'openblas_set_num_threads',None)
    if setter: setter(1)
    solve=lib.dgesv_;solve.argtypes=[ct.c_void_p]*8;solve.restype=None
    def lapack(a,b):
        # Fortran expects column-major storage; DGESV overwrites both buffers.
        aa=np.array(a,order='F',copy=True);bb=b.copy();n=ct.c_int(len(b));one=ct.c_int(1)
        piv=np.empty(len(b),dtype=np.int32);info=ct.c_int()
        start=time.perf_counter()
        solve(ct.byref(n),ct.byref(one),aa.ctypes.data,ct.byref(n),piv.ctypes.data,bb.ctypes.data,ct.byref(n),ct.byref(info))
        return bb,info.value,time.perf_counter()-start,piv.tolist()
    def teaching(a,b,pivot=True):
        aa=np.array(a,order='C',copy=True);bb=b.copy();start=time.perf_counter()
        info=manual.gaussian(len(b),aa.ctypes.data,bb.ctypes.data,int(pivot))
        return bb,info,time.perf_counter()-start
    n=s['lapack_points'];left=s['rod_left'];right=s['rod_right'];source=s['rod_source']
    dx=1/(n+1);a=2*np.eye(n)-np.eye(n,k=1)-np.eye(n,k=-1)
    b=np.full(n,source*dx*dx);b[0]+=left;b[-1]+=right
    xx=np.arange(1,n+1)*dx;analytic=left+(right-left)*xx+0.5*source*xx*(1-xx)
    samples={'manual_pivot':[],'lapack':[]};results={}
    for method,fn in [('manual_pivot',teaching),('lapack',lapack)]:
        x,info,*_=fn(a,b);assert info==0
        for _ in range(s['repeats']):
            x,info,seconds,*_=fn(a,b);assert info==0;samples[method].append(seconds)
        residual=float(np.linalg.norm(a@x-b,np.inf)/(np.linalg.norm(a,np.inf)*np.linalg.norm(x,np.inf)+np.linalg.norm(b,np.inf)))
        assert residual<1e-12 and np.allclose(x,analytic,rtol=1e-9,atol=1e-9)
        results[method]=dict(relative_residual=residual,max_analytic_error=float(np.max(np.abs(x-analytic))))
    # Nonsingular matrix with zero leading diagonal: row exchange is necessary.
    tricky=np.array([[0.,1.],[1.,1.]]);rhs=np.array([1.,2.])
    _,nopivot,_=teaching(tricky,rhs,False);xp,pinfo,_=teaching(tricky,rhs);xl,linfo,_,piv=lapack(tricky,rhs)
    assert nopivot>0 and pinfo==linfo==0 and np.allclose(xp,[1,1]) and np.allclose(xl,[1,1])
    _,singular,_,_=lapack(np.array([[1.,2.],[2.,4.]]),np.array([3.,6.]))
    assert singular>0
    return dict(status='PASS',library=path,configuration=config,compile_command=cmd,points=n,
                boundaries=[left,right],source=source,timings={k:summary(v) for k,v in samples.items()},checks=results,
                temperatures=x.tolist(),positions=xx.tolist(),pivot_example=dict(no_pivot_info=nopivot,lapack_info=linfo,pivots=piv,solution=xl.tolist()),
                singular_info=singular,note='Steady rod: -T double-prime = source, unit length, fixed end temperatures. Both dense solvers use partial pivoting. DGESV uses 32-bit LAPACK integers, column-major A; INFO=0 success, >0 singular, <0 invalid argument. Copies outside timing. A tridiagonal solver would exploit this rod structure better than dense DGESV; dense solve is for learning the library interface.')


def hdf5_runtime_environment():
    # Compiler wrappers can find HDF5 while the dynamic loader misses its
    # indirect Szip/libaec dependencies. A RUNPATH on the executable alone
    # does not resolve every dependency of libhdf5. Use only directories
    # supplied by the current loaded module/toolchain environment.
    env=os.environ.copy();directories=[]
    for variable in ['LD_LIBRARY_PATH','LIBRARY_PATH']:
        directories += [v for v in env.get(variable,'').split(os.pathsep) if v]
    # Prefer the selected HDF5 and compression roots before other loaded roots.
    roots=['EBROOTHDF5','EBROOTSZIP','EBROOTLIBAEC']
    roots += sorted(k for k in env if k.startswith('EBROOT') and k not in roots)
    for key in roots:
        if env.get(key):
            for folder in ['lib','lib64']:
                candidate=Path(env[key])/folder
                if candidate.is_dir(): directories.append(str(candidate))
    directories=list(dict.fromkeys(v for v in directories if Path(v).is_dir()))
    if directories: env['LD_LIBRARY_PATH']=os.pathsep.join(directories)
    return env,directories

def hdf5_lab(s,root):
    try: import h5py
    except ImportError:
        # Parallel builds commonly provide h5pcc instead of h5cc.
        runtime_env,runtime_dirs=hdf5_runtime_environment()
        hdfroot=Path(os.environ['EBROOTHDF5']) if os.getenv('EBROOTHDF5') else None
        wrappers=[shutil.which('h5cc'),shutil.which('h5pcc')]
        if hdfroot: wrappers += [str(hdfroot/'bin'/name) for name in ['h5cc','h5pcc'] if (hdfroot/'bin'/name).is_file()]
        source=str(Path(__file__).with_name('m3s4_hdf5.c'));exe=str(root/'hdf5_demo')
        attempts=[[w,source,'-o',exe] for w in dict.fromkeys(wrappers) if w]
        if hdfroot:
            cc=shutil.which('mpicc') or shutil.which('gcc')
            for folder in ['lib','lib64']:
                libdir=hdfroot/folder
                if cc and (libdir/'libhdf5.so').exists():
                    attempts.append([cc,source,'-I'+str(hdfroot/'include'),'-L'+str(libdir),
                                     '-Wl,-rpath,'+str(libdir),'-lhdf5','-o',exe])
        if not attempts: return dict(status='SKIPPED',reason='No h5py, h5cc/h5pcc or usable EBROOTHDF5 compiler path')
        errors=[]
        for cmd in attempts:
            # Some wrapper installations record dependency -L paths without
            # exporting every dependency as an EasyBuild module variable.
            if Path(cmd[0]).name in ['h5cc','h5pcc']:
                show=subprocess.run([cmd[0],'-show'],cwd=root,capture_output=True,text=True,env=runtime_env,timeout=30)
                if show.returncode==0:
                    flags=shlex.split(show.stdout);extra=[]
                    for i,flag in enumerate(flags):
                        if flag=='-L' and i+1<len(flags): extra.append(flags[i+1])
                        elif flag.startswith('-L') and len(flag)>2: extra.append(flag[2:])
                    runtime_dirs=list(dict.fromkeys(runtime_dirs+[d for d in extra if Path(d).is_dir()]))
                    if runtime_dirs: runtime_env['LD_LIBRARY_PATH']=os.pathsep.join(runtime_dirs)
            try: checked(cmd,root,env=runtime_env);break
            except RuntimeError as e: errors.append(str(e))
        else: raise RuntimeError('HDF5 found but compilation failed: '+ '\n'.join(errors))
        args=[exe,*map(str,s['weather_shape']),str(s['weather_slice']),'1' if s['compression']=='gzip' else '0']
        loader=None
        if shutil.which('ldd'):
            loader=checked([shutil.which('ldd'),exe],root,env=runtime_env)
            if 'not found' in loader:
                raise RuntimeError('HDF5 runtime dependencies missing after module path repair:\n'+loader+
                                   '\nLoad the matching Szip/libaec module for the HDF5 toolchain; runtime dirs: '+str(runtime_dirs))
        result=json.loads(checked(args,root,env=runtime_env));assert result['passed']
        assert result['shape']==s['weather_shape'] and result['slice_time']==s['weather_slice']
        return dict(status='PASS',backend='C HDF5 serial writer',result=result,command=cmd,run_command=args,
                    bytes=(root/'weather.h5').stat().st_size,runtime_library_dirs=runtime_dirs,loader_dependencies=loader,
                    note='Editable shape, one-plane chunks, units, time axis, compression and hyperslab. Full values and time-axis readback verified. h5pcc selects a parallel-capable library; this program still uses one writer.')
    shape=tuple(s['weather_shape']); t,y,x=np.indices(shape); data=273.15+t+0.1*y+0.01*x
    path=root/'weather.h5'; start=time.perf_counter()
    with h5py.File(path,'w') as f:
        ds=f.create_dataset('temperature',data=data,chunks=(1,shape[1],shape[2]),compression=s['compression'])
        ds.attrs['units']='K'; ds.attrs['dimensions']='time,y,x'
        f.create_dataset('time',data=np.arange(shape[0])); f['time'].attrs['units']='hours since simulation start'
    write_s=time.perf_counter()-start
    with h5py.File(path,'r') as f:
        ds=f['temperature']; assert ds.shape==shape and ds.attrs['units']=='K'
        assert np.array_equal(ds[:],data)
        start=time.perf_counter(); region=ds[s['weather_slice'],:,:]; read_s=time.perf_counter()-start
        assert np.array_equal(region,data[s['weather_slice']])
        info=dict(shape=list(ds.shape),chunks=list(ds.chunks),compression=ds.compression,units=ds.attrs['units'])
    return dict(status='PASS',backend='h5py serial HDF5',h5py=h5py.__version__,hdf5=h5py.version.hdf5_version,
                dataset=info,bytes=path.stat().st_size,write_close_s=write_s,read_slice_s=read_s,
                note='Full readback plus selected time slice. Serial HDF5, not parallel HDF5; NetCDF would add standardized dimensions/conventions. No compression speedup assumed.')

def mpi_lab(s,root):
    cc=shutil.which('mpicc')
    if not cc: return dict(status='SKIPPED',reason='mpicc unavailable; load MPI toolchain')
    inside=bool(os.getenv('SLURM_JOB_ID'))
    allocated=int(os.getenv('SLURM_NTASKS','1') or '1')
    if inside and allocated<s['mpi_ranks']:
        return dict(status='SKIPPED',reason=f'current allocation has {allocated} task(s); MPI-IO needs {s["mpi_ranks"]}. Submit the notebook batch job instead of reusing the Jupyter allocation.')
    source=Path(__file__).with_name('m3s4_mpiio.c'); binary=root/'mpiio'
    compile_cmd=[cc,'-O2','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)]; checked(compile_cmd,root)

    launchers=[]
    if inside and shutil.which('srun'):
        launchers.append([shutil.which('srun'),'--exact','--ntasks='+str(s['mpi_ranks'])])
    if shutil.which('mpiexec'):
        launchers.append([shutil.which('mpiexec'),'-n',str(s['mpi_ranks'])])
    if not launchers: return dict(status='SKIPPED',reason='no usable MPI launcher (srun/mpiexec)')

    attempts=[]; text=None; command=None
    for prefix in launchers:
        (root/'mpi_output.bin').unlink(missing_ok=True)
        command=prefix+[str(binary),str(s['mpi_block'])]
        p=subprocess.run([str(x) for x in command],cwd=root,text=True,capture_output=True,timeout=120)
        attempts.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
        if p.returncode==0:
            text=p.stdout
            break
    if text is None:
        detail='\\n\\n'.join(
            'command: '+repr(a['command'])+'\\nrc='+str(a['returncode'])+'\\nstdout:\\n'+a['stdout']+'\\nstderr:\\n'+a['stderr']
            for a in attempts
        )
        raise RuntimeError('MPI launch failed with all available launchers:\\n'+detail)

    result=json.loads(next(line for line in text.splitlines() if line.startswith('{')))
    assert result['passed'] and result['ranks']==s['mpi_ranks']
    assert result['bytes']==s['mpi_ranks']*s['mpi_block']
    contents=(root/'mpi_output.bin').read_bytes()
    expected=b''.join(bytes((r*17+j)%251 for j in range(s['mpi_block'])) for r in range(s['mpi_ranks']))
    assert contents==expected
    return dict(status='PASS',command=command,attempts=attempts,compiler_command=compile_cmd,result=result,
                note='Real MPI processes, collective write_at_all to nonoverlapping byte ranges, sync/close, reopen and collective readback. Demonstrates correctness, not proof of scalability on a parallel filesystem.')

def simulation(data,seed,sigma):
    noise=np.random.default_rng(seed).normal(0,sigma,size=data.shape)
    return float(np.mean((data+noise)**2))

def replay(manifest_path):
    path=Path(manifest_path).resolve(); m=json.loads(path.read_text()); root=path.parent
    assert sha(Path(__file__))==m['sources']['m3s4_lab.py'], 'runner source changed'
    inp=root/m['input']['path']; assert sha(inp)==m['input']['sha256'], 'input checksum mismatch'
    params=m['parameters']; result=simulation(np.load(inp),params['seed'],params['noise_sigma'])
    assert math.isclose(result,m['result']['value'],rel_tol=1e-12,abs_tol=1e-12), 'scientific result differs'
    return dict(status='PASS',recorded=m['result']['value'],replayed=result,
                note='Numerical agreement within tolerance; replay timings are not expected to match.')

def main(config,root):
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=True)
    if (root/'report.json').exists(): raise RuntimeError('Use a new run directory; report already exists')
    s=json.loads(Path(config).read_text()); validate(s); save(root/'settings.json',s); config=root/'settings.json'
    report=dict(build=BUILD,settings=s,run_id=root.name,experiments={})
    for name,fn in [('blas',lambda:blas_lab(s,root)),('lapack',lambda:lapack_lab(s,root)),('fft',lambda:fft_lab(s,root)),('io',lambda:io_lab(s,root)),
                    ('checkpoint',lambda:checkpoint_lab(s,root,config)),('hdf5',lambda:hdf5_lab(s,root)),('mpiio',lambda:mpi_lab(s,root))]:
        try: report['experiments'][name]=fn()
        except Exception as e: report['experiments'][name]=dict(status='FAIL',reason=f'{type(e).__name__}: {e}')
        print(name,report['experiments'][name]['status'],flush=True); save(root/'report.json',report)
    data=np.random.default_rng(s['input_seed']).normal(size=s['input_elements']); np.save(root/'input.npy',data)
    samples=[]; result=None
    for _ in range(s['repeats']):
        t=time.perf_counter(); result=simulation(data,s['seed'],s['noise_sigma']); samples.append(time.perf_counter()-t)
    bad=[simulation(data,int.from_bytes(os.urandom(8),'little'),s['noise_sigma']) for _ in range(2)]
    sources={p.name:sha(p) for p in Path(__file__).parent.glob('m3s4_*') if p.suffix in ['.py','.c','.sbatch']}
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=Path(__file__).parent,capture_output=True,text=True)
    df=subprocess.run(['df','-T',str(root)],capture_output=True,text=True) if shutil.which('df') else None
    manifest=dict(build=BUILD,run_id=root.name,sources=sources,code_commit=git.stdout.strip() if git.returncode==0 else None,
                  input=dict(path='input.npy',sha256=sha(root/'input.npy'),elements=s['input_elements']),parameters=s,
                  command=[sys.executable,str(Path(__file__).resolve()),'--config','settings.json','--out',str(root)],
                  replay_command=[sys.executable,str(Path(__file__).resolve()),'--replay','manifest.json'],
                  software=dict(python=sys.version,numpy=np.__version__,modules=[m for m in os.getenv('LOADEDMODULES','').split(':') if m],
                                blas=report['experiments']['blas'],lapack=report['experiments']['lapack'],
                                fftw=report['experiments']['fft'],hdf5=report['experiments']['hdf5']),
                  system=dict(host=platform.node(),platform=platform.platform(),cpus=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                              filesystem=df.stdout if df else None),
                  resources={k:os.getenv(k) for k in ['SLURM_JOB_ID','SLURM_JOB_NUM_NODES','SLURM_NTASKS','SLURM_CPUS_PER_TASK','SLURM_JOB_PARTITION','SLURM_JOB_NODELIST','OMP_NUM_THREADS']},
                  result=dict(value=result,measurement=summary(samples)),
                  outputs={p.name:sha(p) for p in root.iterdir() if p.is_file() and p.name not in ['report.json','manifest.json']})
    save(root/'manifest.json',manifest); report['experiments']['replay']=replay(root/'manifest.json')
    report['unseeded_results']=bad
    report['experiments']['replay']['changed_seed_result']=simulation(data,s['seed']+1,s['noise_sigma'])
    report['experiments']['replay']['changed_sigma_result']=simulation(data,s['seed'],s['noise_sigma']*2)
    report['failed']=[k for k,v in report['experiments'].items() if v['status']=='FAIL']
    report['skipped']=[k for k,v in report['experiments'].items() if v['status']=='SKIPPED']
    report['core_pass']=all(report['experiments'][k]['status']=='PASS' for k in ['io','checkpoint','replay']) and not report['failed']
    save(root/'report.json',report); return 0 if report['core_pass'] else 1

if __name__=='__main__':
    if '--heat-child' in sys.argv: heat_child(*sys.argv[2:]); sys.exit(0)
    parser=argparse.ArgumentParser(); parser.add_argument('--config'); parser.add_argument('--out'); parser.add_argument('--replay')
    args=parser.parse_args()
    if args.replay: print(json.dumps(replay(args.replay),indent=2))
    elif args.config and args.out: sys.exit(main(args.config,args.out))
    else: parser.error('--config and --out required, or --replay')

