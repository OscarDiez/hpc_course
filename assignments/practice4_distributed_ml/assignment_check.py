"""Assignment capability check and student-run evidence harness.
Preflight never implements or claims to validate the student's missing function.
"""
import argparse, hashlib, json, os, shutil, socket, subprocess, sys, time
from pathlib import Path
import torch
# Import validated helpers; no training occurs during import.
import ml_lab as lab
ROOT=Path(__file__).resolve().parent

def emit(report):lab.emit(ROOT/'assignment_report.json',report)
def run(layout,mode,name,expected=0):
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    old=os.environ.get('M4S5_LAYOUT');os.environ['M4S5_LAYOUT']=layout
    try:
        lab.launch([sys.executable,'-m','torch.distributed.run','--nnodes=1','--master-addr=127.0.0.1',f'--master-port={port}',
                    '--nproc_per_node=2','ml_lab.py','--mode',mode,'--output',name],name+'.log',expected=expected)
    finally:
        if old is None:os.environ.pop('M4S5_LAYOUT',None)
        else:os.environ['M4S5_LAYOUT']=old
    r=json.loads((ROOT/(name+'.json')).read_text())
    if r['layout']!=layout or r['world_size']!=2:raise AssertionError('Unexpected layout/world size')
    return r

def check_mode(mode):
    cfg=json.loads((ROOT/'config.json').read_text())
    if cfg['global_batch'] not in (128,256) or cfg['samples']!=4096 or cfg['epochs']!=6:
        raise ValueError('Required assignment settings: N=4096, epochs=6, global_batch=128 or 256')
    if (ROOT/'assignment_report.json').exists():
        history=ROOT/'previous_outputs'/'reports';history.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'assignment_report.json',history/(str(time.time_ns())+'.json'))
    report={'status':'RUNNING','mode':mode,'execution':'SLURM' if os.getenv('SLURM_JOB_ID') else 'LOCAL',
            'job_id':os.getenv('SLURM_JOB_ID'),'host':socket.gethostname(),'config':cfg,'checks':[],
            'student_source_sha256':lab.digest(ROOT/'student.py'),'harness_sha256':lab.digest(__file__),
            'lab_source_sha256':lab.digest(ROOT/'ml_lab.py'),'started_at_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    emit(report)
    try:
        # Prevent an old output from being mistaken for this attempt's result.
        outputs=[]
        for pattern in ('train-w*-r*.json','train-w*-r*.pt','candidate-*.json','candidate-*.pt','unequal-*.json','unequal-*.pt','*.log'):
            outputs.extend(path for path in ROOT.glob(pattern) if path.name not in ('c-result.log','c-build.log'))
        for name in ('report.json','selection.json','equivalence.json','equivalence.pt','checkpoint.pt',
                     'uninterrupted.pt','uninterrupted.json','resumed.pt','resumed.json','SUMMARY.txt','ASSIGNMENT_SUMMARY.txt'):
            if (ROOT/name).exists():outputs.append(ROOT/name)
        if outputs:
            archive=ROOT/'previous_outputs'/str(time.time_ns());archive.mkdir(parents=True)
            for path in set(outputs):
                shutil.copy2(path,archive/path.name);path.unlink()
        if lab.workflow()!=0:raise RuntimeError('Supplied balanced workload failed; inspect SUMMARY.txt/report.json')
        baseline=json.loads((ROOT/'report.json').read_text())
        if baseline['status']!='PASS':raise RuntimeError('Required baseline did not PASS')
        report['checks'].append({'name':'Supplied baseline, C, real DDP, candidate workflow, recovery and inference','status':'PASS'})
        report['baseline_report']='report.json';emit(report)
        bad=run('unequal_naive','equivalence','unequal-naive',expected=1)
        if bad['status']!='FAIL' or bad['serial_update_max_difference']<=1e-10 or bad['replica_max_parameter_difference']>=1e-10:
            raise AssertionError('Naive unequal update did not produce the expected serial mismatch with synchronized replicas')
        report['naive_unequal']=bad
        report['checks'].append({'name':'Intentional unequal-shard error detected against serial reference','status':'PASS'})
        if mode=='preflight':
            report['checks'].append({'name':'Student unequal-shard correction and full-training comparison','status':'NOT RUN',
                                    'reason':'Preflight validates supplied infrastructure, not a missing student implementation'})
            report['status']='PASS';report['scope']='PREFLIGHT ONLY: student implementation NOT RUN'
        else:
            good=run('unequal_corrected','equivalence','unequal-corrected-equivalence')
            if good['status']!='PASS':raise AssertionError('Student one-step update does not match serial reference')
            trials=[]
            for rep in range(1,4):
                # Match initialization, data, global-batch sequence, updates and precision.
                candidate=run('unequal_corrected','train',f'unequal-corrected-r{rep}')
                a=torch.load(ROOT/f'train-w1-r{rep}.pt',weights_only=True)
                b=torch.load(ROOT/f'unequal-corrected-r{rep}.pt',weights_only=True)
                diff=max(float((a[k]-b[k]).abs().max()) for k in a)
                if diff>=1e-9:raise AssertionError(f'Full unequal training differs from reference: {diff}')
                candidate['full_training_max_parameter_difference']=diff;trials.append(candidate)
            report['corrected_equivalence']=good;report['corrected_trials']=trials
            report['checks'].append({'name':'Student unequal-shard correction and full-training comparison','status':'PASS'})
            report['status']='PASS';report['scope']='THIS CONFIGURATION: student numerical checks and supplied workflow passed; interpretation/authorship not assessed'
    except Exception as exc:
        report['status']='FAIL';report['error']=str(exc)
        report['checks'].append({'name':'Required assignment check','status':'FAIL','error':str(exc)})
    emit(report)
    lines=['M4S5 individual assignment check','Mode: '+mode,'Execution: '+report['execution'],
           'Job: '+str(report['job_id']),'Configuration: '+json.dumps(cfg)]
    lines += [c['status']+'  '+c['name'] for c in report['checks']]
    lines += ['Overall: '+report['status'],report.get('scope',report.get('error',''))]
    (ROOT/'ASSIGNMENT_SUMMARY.txt').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    return 0 if report['status']=='PASS' else 1
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['preflight','full'],default='preflight')
    sys.exit(check_mode(ap.parse_args().mode))
