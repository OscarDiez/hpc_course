#!/usr/bin/env python3
"""Real measurements; standard library only. Called on the allocated compute node."""
import hashlib, json, math, os, platform, shutil, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.chdir(ROOT)
S=json.loads(Path(sys.argv[1] if len(sys.argv)>1 else 'settings.json').read_text())
REPORT=Path('report.json')
FLAGS=['-O3','-g','-fno-omit-frame-pointer','-std=c11']
CPUS=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
R={'schema':1,'settings':S,'host':platform.node(),'job_id':os.environ.get('SLURM_JOB_ID'),
   'compiler_flags':FLAGS,'allowed_cpus':CPUS,'measurements':{},'tools':{},'artifacts':{},
   'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.c')},'core_status':'RUNNING'}

def save():
    Path('report.json.tmp').write_text(json.dumps(R,indent=2));Path('report.json.tmp').replace(REPORT)

def command(argv, *, serial=True, env=None, timeout=90):
    child_env=os.environ.copy();child_env.update({'LC_ALL':'C','OMP_DYNAMIC':'FALSE','OMP_PROC_BIND':'close','OMP_PLACES':'cores'})
    if env:child_env.update(env)
    print('$ '+' '.join(map(str,argv)),flush=True)
    pin=(lambda:os.sched_setaffinity(0,{CPUS[0]})) if serial and CPUS else None
    t=time.perf_counter()
    try:
        p=subprocess.run(list(map(str,argv)),cwd=ROOT,env=child_env,capture_output=True,text=True,
                         timeout=timeout,preexec_fn=pin)
        result={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'process_wall':time.perf_counter()-t}
    except (subprocess.TimeoutExpired,OSError) as exc:
        result={'returncode':124,'stdout':'','stderr':str(exc),'process_wall':time.perf_counter()-t}
    print(result['stdout'],end='',flush=True)
    if result['stderr']:print(result['stderr'],file=sys.stderr,flush=True)
    return result

def core(argv,**kw):
    p=command(argv,**kw)
    if p['returncode']!=0:raise RuntimeError(f'Core command failed ({p["returncode"]}): {argv}')
    return p

def kv(text,prefix):
    return [dict(t.split('=',1) for t in line.split()[1:] if '=' in t)
            for line in text.splitlines() if line.startswith(prefix+' ')]

def checked(argv,prefix,**kw):
    p=core(argv,**kw);rows=kv(p['stdout'],prefix)
    if 'result=PASS' not in p['stdout'] or not rows:raise RuntimeError('Missing validation / experiment output')
    for row in rows:
        for key in ('seconds','compute','barrier_wait'):
            if key in row and (not math.isfinite(float(row[key])) or float(row[key])<0):raise RuntimeError('Invalid timing')
    return p,rows

def optional(name, executable, argv, *, serial=True, artifact=None, env=None, timeout=90):
    if not shutil.which(executable):R['tools'][name]={'status':'UNAVAILABLE','reason':f'{executable} not found'};return None
    p=command(argv,serial=serial,env=env,timeout=timeout)
    status='OK' if p['returncode']==0 else 'FAILED_OR_RESTRICTED'
    if status=='OK' and artifact and not (ROOT/artifact).exists():status='FAILED_OR_RESTRICTED'
    R['tools'][name]={'status':status,**p}
    if artifact and (ROOT/artifact).exists():R['artifacts'][name]=artifact
    return p

def main():
    if S['THREADS']>len(CPUS) and CPUS:raise RuntimeError('THREADS exceeds CPUs permitted by the allocation')
    compiler=shutil.which('gcc')
    if not compiler:raise RuntimeError('gcc is required for the core lab')
    R['compiler_version']=core([compiler,'--version'])['stdout'].splitlines()[0]
    specs=[('profile','m3s3_profile_demo.c',[]),('profile_pg','m3s3_profile_demo.c',['-pg']),
           ('omp_wait','m3s3_openmp_wait.c',['-fopenmp']),('io_demo','m3s3_syscall_demo.c',[]),
           ('memory_check','m3s3_memory_check.c',[])]
    for binary,source,extra in specs:core([compiler,*FLAGS,*extra,source,'-lm','-o',binary])
    def profile(mode,binary='profile'):return ['./'+binary,mode,str(S['N']),str(S['REPS']),str(S['OTHER_ITERS'])]
    # Warm up both modes, then alternate A/B order. Never mix profiled timings into this baseline.
    for mode in ('bad','good'):checked(profile(mode),'PROFILE_TOTAL')
    samples=[]
    for rep in range(S['RUNS']):
        for mode in (('bad','good') if rep%2==0 else ('good','bad')):
            p,total=checked(profile(mode),'PROFILE_TOTAL')
            samples.append({'run':rep+1,'mode':mode,'total':total[0],
                            'phases':kv(p['stdout'],'PROFILE_PHASE'),'process_wall':p['process_wall']})
    checksums=[float(x['total']['checksum']) for x in samples]
    if max(checksums)-min(checksums)>1e-8*max(1,max(checksums)):raise RuntimeError('Locality modes disagree')
    R['measurements']['locality']=samples
    omp=[];io=[]
    def omp_args(mode):return ['./omp_wait',mode,str(S['THREADS']),str(S['SKEW']),str(S['WORK_ITERS'])]
    def io_args(mode):return ['./io_demo',mode,str(S['RECORDS']),str(S['BUFFER_RECORDS'])]
    for mode in ('skewed','balanced'):checked(omp_args(mode),'OMP_TOTAL',serial=False)
    for mode in ('small','buffered'):checked(io_args(mode),'IO')
    for rep in range(S['RUNS']):
        for mode in (('skewed','balanced') if rep%2==0 else ('balanced','skewed')):
            p,rows=checked(omp_args(mode),'OMP_TOTAL',serial=False)
            omp.append({'run':rep+1,'mode':mode,'total':rows[0],'threads':kv(p['stdout'],'OMP_TRACE')})
        for mode in (('small','buffered') if rep%2==0 else ('buffered','small')):
            p,rows=checked(io_args(mode),'IO');io.append({'run':rep+1,**rows[0]})
    if len({x['total']['checksum'] for x in omp})!=1:raise RuntimeError('Parallel modes disagree')
    if len({(x['bytes'],x['checksum']) for x in io})!=1:raise RuntimeError('I/O modes disagree')
    R['measurements']['openmp']=omp;R['measurements']['io']=io
    if Path('/usr/bin/time').exists():optional('gnu_time','/usr/bin/time',['/usr/bin/time','-v',*profile('bad')])
    else:R['tools']['gnu_time']={'status':'UNAVAILABLE','reason':'/usr/bin/time missing'}
    # Instrumentation overhead: same flags and workload; paired order; raw measurements retained.
    pg=[]
    for rep in range(S['RUNS']):
        for binary in (('profile','profile_pg') if rep%2==0 else ('profile_pg','profile')):
            p,rows=checked(profile('bad',binary),'PROFILE_TOTAL')
            pg.append({'run':rep+1,'binary':binary,'seconds':rows[0]['seconds'],'process_wall':p['process_wall']})
    R['measurements']['instrumentation']=pg
    gp=optional('gprof','gprof',['gprof','./profile_pg','gmon.out'])
    if gp and gp['returncode']==0 and 'no time accumulated' in gp['stdout'].lower():
        R['tools']['gprof']['status']='NO_SAMPLES'
    for mode in ('small','buffered'):
        optional('strace_'+mode,'strace',['strace','-c',*io_args(mode)])
    optional('strace_events','strace',['strace','-tt','-T','-e','trace=openat,write,close,unlink','-o','strace_events.txt',*io_args('buffered')],artifact='strace_events.txt')
    if S['PERF']:
        for mode in ('bad','good'):
            result=optional('perf_stat_'+mode,'perf',['perf','stat','-x',';','-e','cycles,instructions,cache-references,cache-misses',*profile(mode)])
            if result and any(w in result['stderr'] for w in ('<not supported>','<not counted>')):
                R['tools']['perf_stat_'+mode]['status']='PARTIAL_COUNTERS'
        optional('perf_sampling','perf',['perf','record','-g','-o','perf.data','--',*profile('bad')],artifact='perf.data')
        if R['tools']['perf_sampling']['status']=='OK':optional('perf_report','perf',['perf','report','--stdio','--no-children','-i','perf.data'])
    else:R['tools']['perf_stat_bad']={'status':'DISABLED','reason':'PERF=False'}
    if S['MEMORY_CHECK']:
        for mode in ('leak','fixed'):
            p=optional('valgrind_'+mode,'valgrind',['valgrind','--leak-check=full','--show-leak-kinds=definite','--errors-for-leak-kinds=definite','--error-exitcode=23','./memory_check',mode])
            if p:
                expected=23 if mode=='leak' else 0
                verified=p['returncode']==expected and ('definitely lost: 16,384 bytes' in p['stderr'] if mode=='leak' else 'All heap blocks were freed' in p['stderr'])
                R['tools']['valgrind_'+mode]['status']='EXPECTED_LEAK_DETECTED' if mode=='leak' and verified else ('OK' if verified else 'FAILED_OR_RESTRICTED')
    else:R['tools']['valgrind']={'status':'DISABLED','reason':'MEMORY_CHECK=False'}
    if S['SCOREP']:
        if not shutil.which('scorep'):
            R['tools']['scorep']={'status':'UNAVAILABLE','reason':'Score-P module not available / not loaded'}
        else:
            for name,source,extra in [('serial','m3s3_profile_demo.c',[]),('openmp','m3s3_openmp_wait.c',['-fopenmp'])]:
                build=optional('scorep_build_'+name,'scorep',['scorep',compiler,*FLAGS,*extra,source,'-lm','-o','scorep_'+name])
                if not build or build['returncode']:continue
                argv=['./scorep_serial',*profile('bad')[2:]] if name=='serial' else ['./scorep_openmp',*omp_args('skewed')[1:]]
                if name=='serial':argv.insert(1,'bad')
                directory='scorep_'+name+'_profile'
                optional('scorep_profile_'+name,'scorep',argv,serial=name=='serial',env={'SCOREP_EXPERIMENT_DIRECTORY':str(ROOT/directory),'SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY':'true','SCOREP_ENABLE_PROFILING':'true','SCOREP_ENABLE_TRACING':'false'},artifact=directory+'/profile.cubex')
                if R['tools']['scorep_profile_'+name]['status']=='OK':
                    optional('scorep_score_'+name,'scorep-score',['scorep-score',directory+'/profile.cubex'])
                    optional('cube_'+name,'cube_dump',['cube_dump','-m','time','-t','aggr','-z','incl','-s','human',directory+'/profile.cubex'])
            if R['tools'].get('scorep_build_openmp',{}).get('status')=='OK':
                directory='scorep_openmp_trace'
                optional('scorep_trace','scorep',['./scorep_openmp',*omp_args('skewed')[1:]],serial=False,env={'SCOREP_EXPERIMENT_DIRECTORY':str(ROOT/directory),'SCOREP_OVERWRITE_EXPERIMENT_DIRECTORY':'true','SCOREP_ENABLE_PROFILING':'false','SCOREP_ENABLE_TRACING':'true','SCOREP_TOTAL_MEMORY':'64M'},artifact=directory+'/traces.otf2')
    else:R['tools']['scorep']={'status':'DISABLED','reason':'SCOREP=False'}
    R['core_status']='PASS';save()
    print('M3S3_CORE_RESULT=PASS',flush=True)
    for name,tool in R['tools'].items():print(f'TOOL_STATUS name={name} status={tool["status"]}',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        R['core_status']='FAIL';R['error']=str(exc);save();print('M3S3_CORE_RESULT=FAIL '+str(exc),file=sys.stderr);sys.exit(1)
