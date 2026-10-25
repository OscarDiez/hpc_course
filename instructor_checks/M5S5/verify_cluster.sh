#!/bin/bash -l
#SBATCH --job-name=m5s5-check
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:06:00
#SBATCH --output=m5s5-check-%j.out
set -euo pipefail
case "${1:-}" in
  --local)
    export M5S5_CHECK_MODE=LOCAL
    export M5S5_CHECK_DIR
    M5S5_CHECK_DIR=$(mktemp -d "$PWD/m5s5-preflight.XXXXXX")
    echo "LOCAL TEST ONLY. No SciTech execution claim. Results: $M5S5_CHECK_DIR"
    ;;
  --allocated)
    : "${SLURM_JOB_ID:?This mode requires a Slurm batch allocation}"
    export M5S5_CHECK_MODE=SLURM
    export M5S5_CHECK_DIR="${2:?Missing owned report directory}"
    ;;
  "")
    command -v sbatch >/dev/null || { echo 'NOT RUN: sbatch is unavailable. Run this on the SciTech login node.'; exit 2; }
    run_dir=$(mktemp -d "$PWD/m5s5-preflight.XXXXXX")
    cp -- "$0" "$run_dir/verify_cluster.sh"
    if [[ -n "${M5S5_SITE_ENV:-}" ]]; then
      cp -- "$M5S5_SITE_ENV" "$run_dir/site_env.sh"
      export M5S5_SITE_ENV="$run_dir/site_env.sh"
    fi
    job=$(sbatch --parsable --chdir="$run_dir" --output="$run_dir/slurm-%j.out" "$run_dir/verify_cluster.sh" --allocated "$run_dir")
    job=${job%%;*}
    echo "Submitted job $job. Queue wait is additional; submission is not PASS."
    echo "Check queue: squeue -j $job"
    echo "Watch output: tail -F '$run_dir/slurm-$job.out'"
    echo "Read summary after completion: cat '$run_dir/SUMMARY.txt'"
    echo "Detailed evidence: $run_dir/report.json"
    exit 0
    ;;
  *) echo 'Usage: bash verify_cluster.sh [--local]'; exit 2 ;;
esac
if [[ -n "${M5S5_SITE_ENV:-}" ]]; then source "$M5S5_SITE_ENV"; fi
command -v python3 >/dev/null || { echo 'FAIL: Python 3 missing. Load the site environment.'; exit 2; }
echo "Host: $(hostname)  Mode: $M5S5_CHECK_MODE  Job: ${SLURM_JOB_ID:-none}"
module -t list 2>&1 || true
command -v gcc || true
command -v mpicc || true
command -v apptainer || true
lscpu | head -25 || true
unset PYTHONOPTIMIZE
python3 - <<'M5S5_PYTHON_CHECK'
SOURCES = {'topk_baseline.c': '/* M5S5 individual assignment: supplied SERIAL baseline, not the optimized answer. */\n#define _POSIX_C_SOURCE 200809L\n#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <math.h>\n#include <time.h>\n#include <string.h>\nstatic double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }\nstatic void stop(const char *why) {\n fprintf(stderr,"FAIL: %s\\n",why);\n exit(2);\n}\n/* Stateless 32-bit hash: each rank can generate its own rows by global ID. */\nstatic uint32_t mix(uint32_t x) { x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; return x^(x>>16); }\nstatic void vector(float *v,int d,uint32_t id) {\n double norm=0;\n for(int j=0;j<d;j++){ v[j]=(float)((int)(mix(id*747796405U+(uint32_t)j*2891336453U)>>8)%2001-1000)/1000.0f; norm+=(double)v[j]*v[j]; }\n norm=sqrt(norm); if(norm==0) stop("zero vector");\n for(int j=0;j<d;j++)v[j]=(float)(v[j]/norm);\n}\nstatic int number(const char *s,int hi) { char *end; long v=strtol(s,&end,10); if(*end||v<1||v>hi)stop("invalid size");return (int)v; }\ntypedef struct { double score; int id; } Hit;\nstatic int compare(const void *pa,const void *pb) {\n const Hit *a=pa,*b=pb;\n if(a->score>b->score)return -1;\n if(a->score<b->score)return 1;\n return (a->id>b->id)-(a->id<b->id);\n}\nint main(int argc,char **argv) {\n if(argc!=5 && argc!=6)stop("usage: topk N D Q K [ties]");\n int n=number(argv[1],1000000),d=number(argv[2],512),nq=number(argv[3],128),k=number(argv[4],100);\n if(k>n)stop("K must not exceed N");\n int ties=argc==6;\n if(ties && strcmp(argv[5],"ties"))stop("optional argument must be ties");\n if((size_t)n*d*sizeof(float)>256U*1024U*1024U)stop("keep collection at or below 256 MiB");\n float *x=malloc((size_t)n*d*sizeof(float)),*q=malloc((size_t)nq*d*sizeof(float));\n Hit *scores=malloc((size_t)n*sizeof(Hit)),*answer=malloc((size_t)nq*k*sizeof(Hit));\n if(!x||!q||!scores||!answer)stop("allocation failed");\n for(int i=0;i<n;i++)vector(x+(size_t)i*d,d,ties?1U:(uint32_t)(i+1));\n for(int j=0;j<nq;j++)vector(q+(size_t)j*d,d,(uint32_t)(10000001+j));\n double compute=0,selection=0,start=now();\n for(int query=0;query<nq;query++) {\n  double t=now();\n  for(int i=0;i<n;i++) {\n   double dot=0;\n   for(int j=0;j<d;j++)dot+=(double)x[(size_t)i*d+j]*q[(size_t)query*d+j];\n   scores[i]=(Hit){dot,i};\n  }\n  compute+=now()-t;t=now();\n  /* Deliberately simple O(N log N) full sort. Students must investigate its cost. */\n  qsort(scores,(size_t)n,sizeof(Hit),compare);\n  for(int j=0;j<k;j++)answer[(size_t)query*k+j]=scores[j];\n  selection+=now()-t;\n }\n double elapsed=now()-start;\n printf("{\\"n\\":%d,\\"d\\":%d,\\"queries\\":%d,\\"k\\":%d,\\"ties\\":%s,\\"ranks\\":1,\\"threads\\":1,\\"seconds\\":%.9g,\\"compute_s\\":%.9g,\\"selection_s\\":%.9g,\\"hits\\":[",n,d,nq,k,ties?"true":"false",elapsed,compute,selection);\n for(int query=0;query<nq;query++) {\n  printf("%s[",query?",":"");\n  for(int j=0;j<k;j++){Hit h=answer[(size_t)query*k+j];printf("%s{\\"id\\":%d,\\"score\\":%.17g}",j?",":"",h.id,h.score);}\n  printf("]");\n }\n puts("]}");free(x);free(q);free(scores);free(answer);return 0;\n}\n', 'check_topk.py': '"""Compare every query\'s complete ordered top-k result; no timing/speedup acceptance.\nUsage: python3 check_topk.py serial_reference.json candidate.json\n"""\nimport json, math, sys\n\ndef check(reference, candidate):\n    for key in [\'n\',\'d\',\'queries\',\'k\',\'ties\']:\n        assert candidate[key] == reference[key], f\'changed problem: {key}\'\n    assert math.isfinite(candidate[\'seconds\']) and candidate[\'seconds\']>0, \'invalid timing\'\n    assert len(candidate[\'hits\']) == reference[\'queries\'], \'missing queries\'\n    max_error = 0.0\n    for q,(expected,actual) in enumerate(zip(reference[\'hits\'],candidate[\'hits\'])):\n        assert len(actual)==reference[\'k\'], f\'wrong K at query {q}\'\n        assert len({h[\'id\'] for h in actual})==len(actual), f\'duplicate ID at query {q}\'\n        for i,(e,a) in enumerate(zip(expected,actual)):\n            assert type(a[\'id\']) is int and 0<=a[\'id\']<reference[\'n\'], \'invalid ID\'\n            assert math.isfinite(a[\'score\']), \'non-finite score\'\n            assert a[\'id\']==e[\'id\'], f\'query {q}, position {i}: ID {a["id"]} != {e["id"]}\'\n            err=abs(a[\'score\']-e[\'score\']);max_error=max(max_error,err)\n            assert math.isclose(a[\'score\'],e[\'score\'],rel_tol=1e-10,abs_tol=1e-10), f\'score mismatch: {err}\'\n            if i:\n                prev=actual[i-1]\n                assert prev[\'score\']>a[\'score\'] or (prev[\'score\']==a[\'score\'] and prev[\'id\']<a[\'id\']), \'unordered top-k\'\n    return max_error\n\nif __name__==\'__main__\':\n    try:\n        if len(sys.argv)!=3:raise ValueError(\'usage: check_topk.py reference.json candidate.json\')\n        reference=json.load(open(sys.argv[1]));candidate=json.load(open(sys.argv[2]))\n        error=check(reference,candidate)\n        print(f\'PASS: every query and all ordered top-k IDs/scores; max error={error:.3g}\')\n    except Exception as e:\n        print(\'FAIL:\',e,file=sys.stderr);sys.exit(1)\n', 'search.c': '/* M5S5 guided exact top-1 search. Deterministic synthetic vectors, not text embeddings.\n * Serial: gcc -O3 -std=c11 search.c -lm -o search_serial\n * OpenMP: gcc -O3 -std=c11 -fopenmp search.c -lm -o search_omp\n * Hybrid: mpicc -O3 -std=c11 -fopenmp -DUSE_MPI search.c -lm -o search_mpi\n */\n#define _POSIX_C_SOURCE 200809L\n#include <stdio.h>\n#include <stdlib.h>\n#include <stdint.h>\n#include <math.h>\n#include <time.h>\n#ifdef _OPENMP\n#include <omp.h>\n#endif\n#ifdef USE_MPI\n#include <mpi.h>\n#endif\nstatic double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec*1e-9; }\nstatic void stop(const char *why) {\n fprintf(stderr,"FAIL: %s\\n",why);\n#ifdef USE_MPI\n MPI_Abort(MPI_COMM_WORLD,2);\n#endif\n exit(2);\n}\n/* Stateless 32-bit hash: each rank can generate its own rows by global ID. */\nstatic uint32_t mix(uint32_t x) { x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; return x^(x>>16); }\nstatic void vector(float *v,int d,uint32_t id) {\n double norm=0;\n for(int j=0;j<d;j++){ v[j]=(float)((int)(mix(id*747796405U+(uint32_t)j*2891336453U)>>8)%2001-1000)/1000.0f; norm+=(double)v[j]*v[j]; }\n norm=sqrt(norm); if(norm==0) stop("zero vector");\n for(int j=0;j<d;j++)v[j]=(float)(v[j]/norm);\n}\nstatic int number(const char *s,int hi) { char *end; long v=strtol(s,&end,10); if(*end||v<1||v>hi)stop("invalid size");return (int)v; }\nint main(int argc,char **argv) {\n int rank=0,ranks=1,threads=1;\n#ifdef USE_MPI\n int provided;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);\n if(provided<MPI_THREAD_FUNNELED)stop("MPI thread support too low");\n MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&ranks);\n#endif\n if(argc!=4)stop("usage: search N D Q (N<=1000000,D<=512,Q<=128)");\n int n=number(argv[1],1000000),d=number(argv[2],512),nq=number(argv[3],128);\n if(ranks>n)stop("more ranks than vectors");\n#ifdef _OPENMP\n #pragma omp parallel\n {\n  #pragma omp single\n  threads=omp_get_num_threads();\n }\n#endif\n int begin=(int)((long long)n*rank/ranks),end=(int)((long long)n*(rank+1)/ranks),local=end-begin;\n float *x=malloc((size_t)local*d*sizeof(float)),*q=malloc((size_t)nq*d*sizeof(float));\n double *scores=malloc((size_t)local*sizeof(double)),*best=malloc((size_t)nq*sizeof(double));\n int *ids=malloc((size_t)nq*sizeof(int));\n double *gather_scores=malloc((size_t)ranks*sizeof(double));int *gather_ids=malloc((size_t)ranks*sizeof(int));\n if(!x||!q||!scores||!best||!ids||!gather_scores||!gather_ids)stop("allocation failed");\n for(int i=0;i<local;i++)vector(x+(size_t)i*d,d,(uint32_t)(begin+i+1));\n for(int k=0;k<nq;k++)vector(q+(size_t)k*d,d,(uint32_t)(10000001+k));\n /* One untimed dot-product pass warms memory/code. It does not initialize results. */\n volatile double warm=0; for(int j=0;j<d;j++)warm+=(double)x[j]*q[j];(void)warm;\n#ifdef USE_MPI\n MPI_Barrier(MPI_COMM_WORLD);\n#endif\n double start=now(),compute=0,selection=0,collective=0;\n for(int k=0;k<nq;k++) {\n  double t=now();\n  /* Each thread writes a different scores[i]; no shared best-index race. */\n  #pragma omp parallel for schedule(static)\n  for(int i=0;i<local;i++) {\n   double dot=0;\n   for(int j=0;j<d;j++)dot+=(double)x[(size_t)i*d+j]*q[(size_t)k*d+j];\n   scores[i]=dot;\n  }\n  compute+=now()-t;t=now();\n  int id=begin;double value=scores[0];\n  for(int i=1;i<local;i++)if(scores[i]>value||(scores[i]==value&&begin+i<id)){value=scores[i];id=begin+i;}\n  selection+=now()-t;t=now();\n#ifdef USE_MPI\n  /* All MPI calls are outside OpenMP regions: FUNNELED is sufficient. */\n  MPI_Gather(&value,1,MPI_DOUBLE,gather_scores,1,MPI_DOUBLE,0,MPI_COMM_WORLD);\n  MPI_Gather(&id,1,MPI_INT,gather_ids,1,MPI_INT,0,MPI_COMM_WORLD);\n  if(rank==0)for(int r=0;r<ranks;r++)if(gather_scores[r]>value||(gather_scores[r]==value&&gather_ids[r]<id)){value=gather_scores[r];id=gather_ids[r];}\n#endif\n  if(ranks>1) collective+=now()-t;\n  best[k]=value;ids[k]=id;\n }\n double elapsed=now()-start;\n#ifdef USE_MPI\n double times[4]={elapsed,compute,selection,collective},maxima[4];\n MPI_Reduce(times,maxima,4,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);\n elapsed=maxima[0];compute=maxima[1];selection=maxima[2];collective=maxima[3];\n#endif\n if(rank==0){\n  printf("{\\"n\\":%d,\\"d\\":%d,\\"queries\\":%d,\\"ranks\\":%d,\\"threads\\":%d,\\"seconds\\":%.9g,\\"compute_max_s\\":%.9g,\\"selection_max_s\\":%.9g,\\"collective_max_s\\":%.9g,\\"hits\\":[",n,d,nq,ranks,threads,elapsed,compute,selection,collective);\n  for(int k=0;k<nq;k++)printf("%s{\\"id\\":%d,\\"score\\":%.17g}",k?",":"",ids[k],best[k]);\n  puts("]}");\n }\n free(x);free(q);free(scores);free(best);free(ids);free(gather_scores);free(gather_ids);\n#ifdef USE_MPI\n MPI_Finalize();\n#endif\n return 0;\n}\n'}
SOURCE_BLOBS = {'topk_baseline.c': '8b374f82cdd580e8c1f7a120d408ba6ad03f9b0a', 'check_topk.py': '95a6a56871673312f2e2ed1ca42d2c6e51839ec4', 'search.c': '7160b32ead3a2f2002381d5167764b42144ce8e6'}
"""Instructor pre-release checks; embedded in the standalone shell launcher."""
from pathlib import Path
import csv, hashlib, importlib.util, json, math, os, platform, shutil, struct, subprocess, sys, time, traceback
from statistics import median

ROOT=Path(os.environ['M5S5_CHECK_DIR']).resolve()
ROOT.mkdir(parents=True,exist_ok=True);os.chdir(ROOT)
LOCAL=os.environ.get('M5S5_CHECK_MODE')=='LOCAL'
report=dict(mode='LOCAL' if LOCAL else 'SLURM',host=platform.node(),
            job_id=os.environ.get('SLURM_JOB_ID'),python=sys.version.split()[0],
            started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),checks=[],measurements=[],
            limitations=['No student parallel/optimized top-k solution is supplied or validated.',
                         'This validates the supplied baseline/checker and guided-code capabilities, not student authorship or Blackboard grading.'])

def save():
    report['overall']='FAIL' if any(c['status']=='FAIL' for c in report['checks']) else 'PASS'
    (ROOT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['M5S5 instructor preflight',f"Mode: {report['mode']}",f"Host: {report['host']}  Job: {report['job_id']}"]
    lines += [f"{c['status']:8} {c['name']}: {c.get('detail','')}" for c in report['checks']]
    lines += [f"Overall attempted checks: {report['overall']}"] + report['limitations']
    (ROOT/'SUMMARY.txt').write_text('\n'.join(lines)+'\n')

def status(name,value,detail):
    report['checks'].append(dict(name=name,status=value,detail=detail));save()
    print(f'{value:8} {name}: {detail}',flush=True)

def cmd(args,env=None,limit=60):
    p=subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=limit,env=env)
    with (ROOT/'commands.log').open('a') as f:
        f.write(json.dumps([str(x) for x in args])+'\n'+p.stdout+p.stderr+f'\nexit={p.returncode}\n')
    if p.returncode: raise RuntimeError(f'command exited {p.returncode}: {args!r}\n{p.stderr[-1800:]}')
    return p.stdout

class Skip(Exception):pass

def test(name,fn):
    try:
        detail=fn();status(name,'PASS',str(detail or 'completed'))
        return True
    except Skip as e:
        status(name,'SKIPPED',str(e));return False
    except Exception as e:
        status(name,'FAIL',str(e));return False

def step(args,ranks=1,threads=1,container=False):
    env=os.environ.copy();env.update(OMP_NUM_THREADS=str(threads),OMP_DYNAMIC='FALSE',OMP_PLACES='cores',OMP_PROC_BIND='close')
    if container:
        env.update(APPTAINERENV_OMP_NUM_THREADS=str(threads),APPTAINERENV_OMP_DYNAMIC='FALSE',APPTAINERENV_OMP_PLACES='cores',APPTAINERENV_OMP_PROC_BIND='close')
    args=[str(x) for x in args]
    if not LOCAL:
        launch=['srun','--exact','--exclusive','-N1',f'-n{ranks}',f'-c{threads}','--cpu-bind=cores']
        if ranks>1 and os.environ.get('M5S5_SRUN_MPI'):launch+=['--mpi='+os.environ['M5S5_SRUN_MPI']]
        args=launch+args
    elif ranks>1:
        args=[shutil.which('mpiexec') or 'mpiexec','-n',str(ranks)]+args
    return cmd(args,env=env)

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def mix(x):
    x&=0xffffffff;x^=x>>16;x=(x*0x7feb352d)&0xffffffff
    x^=x>>15;x=(x*0x846ca68b)&0xffffffff
    return x^(x>>16)
def vector(identifier,d):
    v=[f32(f32((mix(identifier*747796405+j*2891336453)>>8)%2001-1000)/f32(1000)) for j in range(d)]
    length=math.sqrt(sum(x*x for x in v));return [f32(x/length) for x in v]

def independent(result):
    n,d,q,k=(result[x] for x in ['n','d','queries','k'])
    data=[vector(1 if result['ties'] else i+1,d) for i in range(n)]
    assert len(result['hits'])==q
    for query,hits in enumerate(result['hits']):
        v=vector(10000001+query,d)
        scores=[sum(x*y for x,y in zip(row,v)) for row in data]
        ids=sorted(range(n),key=lambda i:(-scores[i],i))[:k]
        assert [h['id'] for h in hits]==ids
        assert all(math.isclose(h['score'],scores[h['id']],rel_tol=1e-10,abs_tol=1e-10) for h in hits)

def build_baseline():
    report['gcc']=cmd(['gcc','--version']).splitlines()[0]
    cmd(['gcc','-O3','-std=c11','-Wall','-Wextra','-Werror','topk_baseline.c','-lm','-o','topk_baseline'])
    return report['gcc']

def fixtures():
    for name,args in [('small',[19,7,3,5]),('ties',[19,7,3,5,'ties']),('boundary',[1,7,1,1]),('k_equals_n',[13,5,2,13])]:
        r=json.loads(step([ROOT/'topk_baseline']+args));independent(r)
        (ROOT/(name+'.json')).write_text(json.dumps(r,indent=2)+'\n')
        cmd(['python3','check_topk.py',name+'.json',name+'.json'])
    return 'C agrees with independent Python: normal, exact ties, N=1, K=N'

def checker_rejects():
    spec=importlib.util.spec_from_file_location('m5s5_checker',ROOT/'check_topk.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    reference=json.loads((ROOT/'small.json').read_text())
    for problem in ['wrong_tail','duplicate','missing_query','bad_score','changed_queries']:
        bad=json.loads(json.dumps(reference))
        if problem=='wrong_tail':bad['hits'][-1][-1]['id']=(bad['hits'][-1][-1]['id']+1)%19
        if problem=='duplicate':bad['hits'][0][1]['id']=bad['hits'][0][0]['id']
        if problem=='missing_query':bad['hits'].pop()
        if problem=='bad_score':bad['hits'][-1][-1]['score']+=0.01
        if problem=='changed_queries':bad['queries']*=2
        try:checker.check(reference,bad)
        except AssertionError:continue
        raise AssertionError('Checker accepted deliberate corruption: '+problem)
    return 'five deliberately incorrect outputs rejected'

def full_baseline():
    spec=importlib.util.spec_from_file_location('m5s5_checker',ROOT/'check_topk.py')
    checker=importlib.util.module_from_spec(spec);spec.loader.exec_module(checker)
    timings={}
    for queries in [32,64]:
        reference=None;times=[]
        for repetition in [1,2,3]:
            r=json.loads(step([ROOT/'topk_baseline',65536,128,queries,10]))
            if reference is None:reference=r
            checker.check(reference,r)
            (ROOT/f'baseline-q{queries}-r{repetition}.json').write_text(json.dumps(r,indent=2)+'\n')
            times.append(r['seconds']);report['measurements'].append(dict(program='serial_topk',queries=queries,repetition=repetition,seconds=r['seconds'],mode=report['mode']))
        timings[str(queries)]=dict(median_s=median(times),min_s=min(times),max_s=max(times))
    report['baseline_timings']=timings
    return '3 main + 3 sensitivity runs; all queries/top-k checked for consistency; actual times in report.json'

def guided_cpu():
    cmd(['gcc','-O3','-std=c11','-Wno-unknown-pragmas','search.c','-lm','-o','guided_serial'])
    cmd(['gcc','-O3','-std=c11','-fopenmp','search.c','-lm','-o','guided_omp'])
    reference=json.loads(step([ROOT/'guided_serial',65536,128,16]));report['guided_reference']=reference
    for threads in [1,2,4]:
        for repetition in [1,2,3]:
            r=json.loads(step([ROOT/'guided_omp',65536,128,16],threads=threads))
            assert r['threads']==threads and r['ranks']==1
            assert len(r['hits'])==len(reference['hits'])
            for a,b in zip(r['hits'],reference['hits']):
                assert a['id']==b['id'] and math.isclose(a['score'],b['score'],rel_tol=1e-10,abs_tol=1e-10)
            (ROOT/f'guided-omp-{threads}-r{repetition}.json').write_text(json.dumps(r,indent=2)+'\n')
            report['measurements'].append(dict(program='guided_omp',threads=threads,repetition=repetition,seconds=r['seconds'],mode=report['mode']))
    return 'real guided C code: actual teams 1/2/4 and top-1 agreement, three runs each'

def hybrid():
    cmd(['mpicc','-O3','-std=c11','-fopenmp','-DUSE_MPI','search.c','-lm','-o','guided_mpi'])
    r=json.loads(step([ROOT/'guided_mpi',65536,128,16],ranks=2,threads=2))
    reference=report['guided_reference'];assert r['ranks']==2 and r['threads']==2
    assert len(r['hits'])==len(reference['hits'])
    for a,b in zip(r['hits'],reference['hits']):
        assert a['id']==b['id'] and math.isclose(a['score'],b['score'],rel_tol=1e-10,abs_tol=1e-10)
    (ROOT/'guided-hybrid.json').write_text(json.dumps(r,indent=2)+'\n')
    return 'guided 2-rank x 2-thread search launched and numerically checked; not a student top-k solution'

def container():
    image=Path(os.environ['M5S5_IMAGE']).expanduser().resolve()
    prefix=['apptainer','exec','--cleanenv','--bind',str(ROOT)+':/work','--pwd','/work',str(image)]
    # Probe for a compiler before calling the image incompatible. A later compile/run error is FAIL.
    probe=step(prefix+['sh','-c','if command -v gcc >/dev/null 2>&1; then echo GCC_AVAILABLE; else echo NO_GCC; fi'],container=True)
    if 'GCC_AVAILABLE' not in probe:raise Skip('Image has no gcc; use an existing compatible C/OpenMP image or the module-based alternative')
    step(prefix+['gcc','-O3','-std=c11','-fopenmp','/work/search.c','-lm','-o','/work/container_guided'],container=True)
    r=json.loads(step(prefix+['/work/container_guided',65536,128,16],threads=4,container=True))
    reference=report['guided_reference'];assert r['threads']==4
    assert len(r['hits'])==len(reference['hits'])
    for a,b in zip(r['hits'],reference['hits']):
        assert a['id']==b['id'] and math.isclose(a['score'],b['score'],rel_tol=1e-10,abs_tol=1e-10)
    h=hashlib.sha256()
    with image.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    report['image']=dict(path=str(image),sha256=h.hexdigest())
    (ROOT/'container-guided.json').write_text(json.dumps(r,indent=2)+'\n')
    return 'existing-image compilation, four-thread execution and native-result agreement'

try:
    for name,text in SOURCES.items():(ROOT/name).write_text(text)
    report['source_sha256']={name:hashlib.sha256(text.encode()).hexdigest() for name,text in SOURCES.items()}
    report['source_git_blobs']=SOURCE_BLOBS
    if not LOCAL:
        assert os.environ.get('SLURM_JOB_ID') and shutil.which('srun'),'Not inside a usable Slurm allocation'
        status('Slurm allocation','PASS','batch started; launcher and job ID present')
    else:status('Slurm allocation','NOT RUN','explicit local validation; no cluster claim')
    baseline_ok=test('Serial build',build_baseline)
    fixture_ok=baseline_ok and test('Independent correctness fixtures',fixtures)
    if not baseline_ok:status('Independent correctness fixtures','NOT RUN','serial build failed')
    if fixture_ok:test('Checker rejects bad results',checker_rejects)
    else:status('Checker rejects bad results','NOT RUN','fixture/build failure')
    if baseline_ok:test('Required workload feasibility',full_baseline)
    else:status('Required workload feasibility','NOT RUN','serial build failed')
    omp_ok=test('Guided OpenMP execution',guided_cpu)
    if not shutil.which('mpicc'):
        status('MPI hybrid execution','SKIPPED','mpicc unavailable; load the site-compatible MPI environment before retesting, or use the assignment CPU alternative')
    elif LOCAL and not shutil.which('mpiexec'):
        status('MPI hybrid execution','SKIPPED','local MPI launcher unavailable')
    elif not omp_ok:status('MPI hybrid execution','NOT RUN','guided CPU reference unavailable')
    else:test('MPI hybrid execution',hybrid)
    image=os.environ.get('M5S5_IMAGE')
    if not image:status('Apptainer execution','SKIPPED','M5S5_IMAGE not set; documented module-based route remains available')
    elif not shutil.which('apptainer'):status('Apptainer execution','SKIPPED','apptainer unavailable; load site environment')
    elif not Path(image).expanduser().is_file():status('Apptainer execution','FAIL','M5S5_IMAGE does not name an accessible local file')
    elif not omp_ok:status('Apptainer execution','NOT RUN','guided CPU reference unavailable')
    else:test('Apptainer execution',container)
except Exception as e:
    status('Preflight setup','FAIL',str(e))
finally:
    save()
    print('\n'+(ROOT/'SUMMARY.txt').read_text(),flush=True)
    print('REPORT DIRECTORY:',ROOT,flush=True)
    if report['mode']=='LOCAL':print('SciTech validation remains NOT RUN.',flush=True)
sys.exit(1 if report['overall']=='FAIL' else 0)

M5S5_PYTHON_CHECK
