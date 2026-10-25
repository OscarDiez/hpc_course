"""Guided native-C -> container-validation workflow inside a Slurm allocation.
Requires a successful CPU run and an existing Python 3 SIF; no gcc in the image.
"""
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time
from container_replay import sha

ROOT=Path(__file__).resolve().parent

def validate_manifest(root,manifest,require_slurm=True):
    if require_slurm and manifest.get('execution')!='SLURM':
        raise ValueError('Required CPU-stage evidence must come from Slurm, not a local debugging run')
    for key,hash_key in [('reference','reference_sha256'),('candidate','candidate_sha256')]:
        if sha(root/manifest[key])!=manifest[hash_key]:
            raise ValueError('Stale workflow input: '+key+'; rerun CPU stage')
    if sha(root/'search.c')!=manifest['source_sha256']:
        raise ValueError('Source changed: rebuild and rerun CPU stage')

def main():
    os.chdir(ROOT)
    if not os.environ.get('SLURM_JOB_ID'):
        print('NOT RUN: submit container.sbatch; do not benchmark in this kernel.');return 2
    out=ROOT/'replay';out.mkdir(exist_ok=True)
    label=f"job-{os.environ['SLURM_JOB_ID']}-{time.time_ns()}"
    report=dict(mode='container_replay',execution='SLURM',job_id=os.environ['SLURM_JOB_ID'],
                status='NOT RUN',checks={})
    def run(command,env=None,allow_failure=False):
        p=subprocess.run([str(x) for x in command],text=True,capture_output=True,env=env,timeout=60)
        with (out/(label+'-commands.log')).open('a') as f:
            f.write(json.dumps([str(x) for x in command])+'\n'+p.stdout+p.stderr+f'\nexit={p.returncode}\n')
        if p.returncode and not allow_failure:raise RuntimeError(p.stderr[-2000:] or p.stdout[-2000:])
        return p
    try:
        manifest_path=ROOT/'workflow_manifest.json'
        if not manifest_path.is_file():raise ValueError('Run cpu.sbatch first: workflow_manifest.json is missing')
        manifest=json.loads(manifest_path.read_text());report['manifest']=manifest
        validate_manifest(ROOT,manifest)
        native=out/(label+'-native.json')
        env=os.environ.copy();env['HPC_M5S5_HOST_SENTINEL']='host-only'
        args=['--reference',ROOT/manifest['reference'],'--candidate',ROOT/manifest['candidate']]
        run([sys.executable,ROOT/'container_replay.py']+args+['--output',native],env=env)
        n=json.loads(native.read_text());report['checks']['native_replay']='PASS'
        image=os.environ.get('M5S5_IMAGE')
        if not shutil.which('apptainer') or not image or not Path(image).expanduser().is_file():
            report['reason']='Required Apptainer step NOT RUN: load Apptainer and configure M5S5_IMAGE to an existing Python 3 SIF'
            report['checks']['apptainer_replay']='NOT RUN';return 2
        image=Path(image).expanduser().resolve();report['image']=dict(path=str(image),sha256=sha(image))
        prefix=['srun','--exact','-N1','-n1','-c1','apptainer','exec','--cleanenv',
                '--bind',str(ROOT)+':/work:ro','--bind',str(out)+':/out','--pwd','/work',str(image),
                'python3','/work/container_replay.py','--reference','/work/'+manifest['reference'],
                '--candidate','/work/'+manifest['candidate']]
        # No GCC or host C runtime is needed in this Python validation container.
        env.pop('APPTAINERENV_HPC_M5S5_HOST_SENTINEL',None)
        env.pop('SINGULARITYENV_HPC_M5S5_HOST_SENTINEL',None)
        clean_name=label+'-container-clean.json'
        run(prefix+['--output','/out/'+clean_name],env=env)
        c=json.loads((out/clean_name).read_text())
        if c['result']!=n['result'] or c['inputs']!=n['inputs'] or c['provenance']['script_sha256']!=n['provenance']['script_sha256']:raise ValueError('Native/container scientific results or input hashes differ')
        report['checks']['apptainer_replay']='PASS'
        env['APPTAINERENV_HPC_M5S5_HOST_SENTINEL']='explicitly-forwarded'
        forwarded_name=label+'-container-forwarded.json'
        run(prefix+['--output','/out/'+forwarded_name],env=env)
        f=json.loads((out/forwarded_name).read_text())
        if f['provenance']['marker']!='explicitly-forwarded' or f['result']!=n['result']:
            raise ValueError('Environment forwarding/result check failed')
        report['checks']['explicit_environment']='PASS'
        report['observed_markers']=dict(native=n['provenance']['marker'],clean=c['provenance']['marker'],forwarded=f['provenance']['marker'])
        # A deliberately corrupted copy is rejected inside the container.
        bad=json.loads((ROOT/manifest['candidate']).read_text());bad['hits'][0]['score']+=0.01
        bad_name=label+'-deliberately-bad.json';(out/bad_name).write_text(json.dumps(bad))
        bad_cmd=prefix.copy();bad_cmd[bad_cmd.index('--candidate')+1]='/out/'+bad_name
        rejected_name=label+'-rejected.json'
        p=run(bad_cmd+['--output','/out/'+rejected_name],env=env,allow_failure=True)
        rejected=json.loads((out/rejected_name).read_text()) if (out/rejected_name).is_file() else {}
        if p.returncode==0 or rejected.get('status')!='FAIL':raise ValueError('Corrupted output was not numerically rejected')
        report['checks']['deliberate_error_rejected']='PASS'
        report['status']='PASS'
    except Exception as e:report.update(status='FAIL',reason=str(e));return 1
    finally:
        (out/(label+'-workflow.json')).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        print('This validates native C outputs inside Apptainer; C/MPI execution inside the image is not claimed.')
    return 0

if __name__=='__main__':sys.exit(main())
