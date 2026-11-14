"""Bounded discovery and real CPU examples using existing local course images."""
import json, math, os, platform, shutil, subprocess, sys
from pathlib import Path

CASE_NAMES=('container_image','container_environment','container_bind','container','container_numpy')

def discover_images(explicit, roots, limit=12):
    """Inspect configured course directories only; no recursive filesystem search."""
    candidates=[]; diagnostics=[]; seen=set()
    def add(path):
        try:
            p=Path(path).expanduser().resolve()
            if p in seen or not p.is_file() or not os.access(p,os.R_OK):return
            st=p.stat();seen.add(p)
            candidates.append({'path':str(p),'size_bytes':st.st_size,'mtime_ns':st.st_mtime_ns})
        except OSError as e:diagnostics.append(str(e))
    if explicit:
        add(explicit)
        if not candidates:diagnostics.append('Configured IMAGE is missing or unreadable: '+str(explicit))
        return candidates,diagnostics
    # Prefer the current course directory; the sponsor path is a historical fallback.
    for folder in roots:
        p=Path(folder).expanduser()
        if not p.is_dir():diagnostics.append('Directory unavailable: '+str(p));continue
        try:
            entries=[]
            with os.scandir(p) as scan:
                for i,entry in enumerate(scan):
                    if i>=128:break
                    if entry.name.lower().endswith(('.sif','.simg')):entries.append(Path(entry.path))
            entries.sort(key=lambda x:(not any(t in x.name.lower() for t in ('python','numpy','pytorch')),x.name))
            for path in entries:
                add(path)
                if len(candidates)>=limit:return candidates,diagnostics
        except OSError as e:diagnostics.append(str(e))
    return candidates,diagnostics

def run_examples(cfg, root, report, record, sha, agree):
    root=Path(root)
    def remaining(status,reason):
        for name in CASE_NAMES:
            if name not in report['cases']:record(name,status,reason=reason)
    if not cfg.get('run_container',True):
        remaining('SKIPPED','RUN_CONTAINER=False; deliberately disabled');return
    runtime=shutil.which('apptainer') or shutil.which('singularity')
    if not runtime:
        runtime=next((p for p in ('/usr/bin/apptainer','/usr/bin/singularity') if os.access(p,os.X_OK)),None)
    if not runtime:
        remaining('SKIPPED','No Apptainer/Singularity runtime visible inside the compute job');return
    candidates,diagnostics=discover_images(cfg.get('image',''),cfg['image_dirs'])
    report['container_discovery']={'runtime':runtime,'candidates':candidates,'diagnostics':diagnostics}
    if not candidates:
        remaining('SKIPPED','No readable image in configured course directories; inspect container_discovery or set IMAGE');return
    # Preserve the image environment but remove host bind/env overrides for this experiment.
    env={k:v for k,v in os.environ.items() if not k.startswith(('APPTAINERENV_','SINGULARITYENV_'))
         and k not in ('APPTAINER_BIND','APPTAINER_BINDPATH','SINGULARITY_BIND','SINGULARITY_BINDPATH')}
    controls='OMP_NUM_THREADS=2,OPENBLAS_NUM_THREADS=2,MKL_NUM_THREADS=2,NUMEXPR_NUM_THREADS=2'
    prefix=[runtime,'exec','--cleanenv','--containall','--bind',str(root)+':/work','--pwd','/work','--env',controls]
    def execute(args,timeout=25):
        return subprocess.run([str(x) for x in args],capture_output=True,text=True,timeout=timeout,env=env,cwd=root)
    def checked(args,timeout=25):
        r=execute(args,timeout)
        if r.returncode:raise RuntimeError('exit '+str(r.returncode)+': '+(r.stderr+r.stdout)[-2000:])
        return r.stdout.strip()
    selected=None;probe=None;attempts=[]
    for item in candidates[:4]:
        image=Path(item['path'])
        for python in ('python3','python'):
            args=prefix+[image,python,'environment_demo.py']
            try:
                r=execute(args,8)
                if r.returncode:raise RuntimeError((r.stderr+r.stdout)[-1200:])
                value=json.loads(r.stdout)
                version=tuple(int(x) for x in value['python'].split('.')[:2])
                if version<(3,9):raise RuntimeError('Python 3.9+ is needed for the embedded payloads')
                selected=(image,python,args);probe=value;break
            except (OSError,ValueError,KeyError,subprocess.TimeoutExpired,RuntimeError) as e:
                attempts.append({'image':str(image),'python':python,'diagnostic':str(e)})
        if selected:break
    report['container_discovery']['attempts']=attempts
    if not selected:
        record('container_image','FAIL',reason='Readable images were found, but no attempted image launched a suitable Python; inspect diagnostics')
        remaining('SKIPPED','No compatible image was selected');return
    image,python,probe_args=selected
    try:
        before=image.stat();image_hash=sha(image);after=image.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Image changed during fingerprinting')
        if cfg.get('expected_image_sha256') and image_hash!=cfg['expected_image_sha256']:
            raise RuntimeError('Image SHA-256 differs from EXPECTED_IMAGE_SHA256')
        report['container_image']={'path':str(image),'sha256':image_hash,'size_bytes':after.st_size,
                                   'mtime_ns':after.st_mtime_ns,'python_command':python,
                                   'runtime_version':checked([runtime,'--version'])}
        record('container_image','PASS',**report['container_image'])
    except Exception as e:
        record('container_image','FAIL',reason=str(e));remaining('SKIPPED','Image identity check failed');return
    def run_python(script,*args):return checked(prefix+[image,python,script,*args])
    # Example 1: a real process inside the image, compared with the compute host.
    try:
        native=json.loads(checked([sys.executable,'environment_demo.py']))
        ok=probe['kernel']==native['kernel'] and probe['machine']==native['machine']
        record('container_environment','PASS' if ok else 'FAIL',host=native,inside=probe,
               command=[str(a) for a in probe_args],note='Python versions may differ; kernel and machine should agree on this host')
    except Exception as e:record('container_environment','FAIL',reason=str(e))
    # Example 2: read the host CSV through /work and verify the written output on the host.
    try:
        target=root/'written_by_container.json';target.unlink(missing_ok=True)
        printed=json.loads(run_python('bind_demo.py'));written=json.loads(target.read_text())
        ok=printed==written and written['input_sha256']==sha(root/'temperatures.csv') and written['rows']==cfg['reference']['count']
        record('container_bind','PASS' if ok else 'FAIL',result=written,host_output=str(target))
    except Exception as e:record('container_bind','FAIL',reason=str(e))
    # Example 3: unchanged scientific code, input and threshold in a different Python runtime.
    try:
        (root/'container.json').unlink(missing_ok=True)
        run_python('analysis.py','--input','temperatures.csv','--threshold',cfg['threshold_c'],'--output','container.json')
        got=json.loads((root/'container.json').read_text())
        ref=json.loads((root/'native.json').read_text())['result']
        ok=agree(got['result'],ref) and got['input_sha256']==sha(root/'temperatures.csv') and got['code_sha256']==sha(root/'analysis.py') and got['parameters']['threshold_c']==cfg['threshold_c']
        record('container','PASS' if ok else 'FAIL',result=got['result'],runtime=got['runtime'],image_sha256=image_hash)
    except Exception as e:record('container','FAIL',reason=str(e))
    # Example 4: use NumPy already supplied by the image, if available; no installation.
    try:
        scale=cfg['load_scale'];results=[]
        for factor in (scale,2*scale):
            r=execute(prefix+[image,python,'numpy_demo.py',str(factor)])
            if r.returncode==77:
                record('container_numpy','SKIPPED',reason=r.stdout.strip());break
            if r.returncode:raise RuntimeError((r.stderr+r.stdout)[-2000:])
            got=json.loads(r.stdout);d=got['displacements'];f=[factor*x for x in (5.,6.,7.)]
            if len(d)!=3 or not all(math.isfinite(v) for v in d):raise RuntimeError('Invalid displacement vector')
            k=((12.,-4.,2.),(-4.,8.,-2.),(2.,-2.,6.))
            residual=max(abs(sum(a*b for a,b in zip(row,d))-force) for row,force in zip(k,f))
            if got['forces']!=f or residual>1e-10:raise RuntimeError('Independent stiffness-system residual check failed')
            results.append(got)
        else:
            ok=all(math.isclose(b,2*a,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(results[0]['displacements'],results[1]['displacements']))
            record('container_numpy','PASS' if ok else 'FAIL',results=results,note='Doubling force should double displacement; CPU only')
    except Exception as e:record('container_numpy','FAIL',reason=str(e))
    final=image.stat()
    if (after.st_size,after.st_mtime_ns)!=(final.st_size,final.st_mtime_ns):
        record('container_image','FAIL',reason='Image changed while examples were running')
        record('container','FAIL',reason='Image changed while examples were running; cannot accept the recorded identity')
