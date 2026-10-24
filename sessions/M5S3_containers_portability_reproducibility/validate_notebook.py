import ast,contextlib,importlib.util,io,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
BASE=Path(__file__).resolve().parent
nb=json.loads((BASE/'M5S3_containers_portability_reproducibility.ipynb').read_text())
assert len({c['id'] for c in nb['cells']})==len(nb['cells'])
code=[c for c in nb['cells'] if c['cell_type']=='code']
for c in code:ast.parse(c['source']);assert c['execution_count'] is None and not c['outputs']
setup=code[0]['source'];assert 'RUN_CONTAINER = True' in setup
sources=ast.literal_eval(ast.parse(next(c['source'] for c in code if c['source'].startswith('SOURCES = '))).body[0].value)
for name,value in sources.items():
 if name.endswith('.py'):ast.parse(value)
result={'context':'LOCAL_REVISION_TESTS_NOT_SCITECH_OR_APPTAINER','checks':{}}
with tempfile.TemporaryDirectory(prefix='m5s3-v3-') as temp:
    temp=Path(temp);old=Path.cwd();os.chdir(temp)
    try:
        ns={'__name__':'__main__'};log=io.StringIO()
        with contextlib.redirect_stdout(log):
            for i,c in enumerate(code):
                exec(compile(c['source'],f'notebook-code-{i+1}','exec'),ns)
                if i==0:ns['RUN_CLUSTER']=False
        root=ns['ROOT'];run=ns['RUN_DIR']
        assert all(x=='PASS' for x in ns['EVIDENCE'].values())
        assert not ns['JOB_ID']
        result['checks']['all_21_python_cells']={'status':'PASS','harness':'sequential Python execution; RUN_CLUSTER=False only in test harness','evidence':ns['EVIDENCE']}
        assert subprocess.run(['bash','-n',str(run/'run.sbatch')]).returncode==0
        result['checks']['bash_syntax']='PASS'
        # Exercise the real native worker on the local CPU with containers explicitly disabled.
        cfg=json.loads((run/'settings.json').read_text());cfg['run_container']=False
        (run/'settings.json').write_text(json.dumps(cfg))
        env=dict(os.environ,OMP_NUM_THREADS='2',OMP_DYNAMIC='FALSE')
        native=subprocess.run([sys.executable,'worker.py'],cwd=run,capture_output=True,text=True,env=env,timeout=90)
        assert native.returncode==0,native.stdout+native.stderr
        report=json.loads((run/'report.json').read_text());assert report['cases']['python']['status']=='PASS'
        assert report['cases']['c_serial']['status']=='PASS';assert report['cases']['c_openmp']['status']=='PASS'
        assert report['cases']['c_openmp']['result']['threads']==2
        assert all(report['cases'][key]['status']=='SKIPPED' for key in ['container_image','container_environment','container_bind','container','container_numpy'])
        result['checks']['native_worker_actual_python_C_OpenMP']={'status':'PASS','cases':report['cases']}
        spec=importlib.util.spec_from_file_location('container_examples',run/'container_examples.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        image_dir=temp/'course images';image_dir.mkdir();image=image_dir/'python teaching.sif';image.write_text('UNIT TEST PLACEHOLDER, NOT A CONTAINER')
        found,diag=mod.discover_images('',[str(image_dir)]);assert len(found)==1 and found[0]['path']==str(image)
        assert mod.discover_images(str(temp/'absent.sif'),[str(image_dir)])[0]==[]
        result['checks']['image_inventory_spaces_and_missing_override']='PASS'
        # A CLI test double launches payloads on the HOST, solely to test orchestration.
        fake=temp/'fakebin';fake.mkdir();shim=fake/'apptainer'
        shim.write_text('''#!'''+sys.executable+'''
import os,sys,subprocess
args=sys.argv[1:]
if args==['--version']:print('CLI TEST DOUBLE; NOT APPTAINER');raise SystemExit(0)
if os.environ.get('TEST_REJECT_IMAGE'):print('incompatible image',file=sys.stderr);raise SystemExit(9)
assert args.pop(0)=='exec'
while args and args[0].startswith('--'):
 key=args.pop(0)
 if key in ('--bind','--pwd','--env'):args.pop(0)
image=args.pop(0)
if os.path.basename(image).startswith('bad'):raise SystemExit(9)
if args[0] in ('python','python3'):args[0]=sys.executable
raise SystemExit(subprocess.run(args,env=dict(os.environ,OMP_NUM_THREADS='2')).returncode)
''');shim.chmod(0o755)
        from unittest.mock import patch
        oldpath=os.environ.get('PATH','');testenv=dict(os.environ,PATH=str(fake)+os.pathsep+oldpath)
        cfg.update(run_container=True,image=str(image),image_dirs=[str(image_dir)],expected_image_sha256='',load_scale=1.)
        def run_case(settings,extra=None):
            r={'cases':{}}
            def record(name,status,**kw):r['cases'][name]=dict(status=status,**kw)
            with patch.dict(os.environ,dict(testenv,**(extra or {})),clear=True):
                mod.run_examples(settings,run,r,record,ns['sha'],lambda a,b:a['count']==b['count'] and all(__import__('math').isclose(a[k],b[k],rel_tol=1e-12,abs_tol=1e-12) for k in ('mean_c','fraction_above')))
            return r
        # native.json already comes from the actual native worker run above.
        happy=run_case(cfg)
        for key in ['container_image','container_environment','container_bind','container']:assert happy['cases'][key]['status']=='PASS',(key,happy)
        assert happy['cases']['container_numpy']['status'] in ('PASS','SKIPPED'),happy
        assert (run/'written_by_container.json').exists()
        result['checks']['CLI_double_control_flow_ONLY']={'status':'PASS','note':'Host subprocesses under a fake CLI; not evidence of an actual container launch.','numpy_payload_status':happy['cases']['container_numpy']['status']}
        wrong=run_case(dict(cfg,expected_image_sha256='0'*64));assert wrong['cases']['container_image']['status']=='FAIL';assert wrong['cases']['container']['status']=='SKIPPED'
        broken=run_case(cfg,{'TEST_REJECT_IMAGE':'1'});assert broken['cases']['container_image']['status']=='FAIL';assert broken['container_discovery']['attempts']
        absent=run_case(dict(cfg,image=str(temp/'missing.sif')));assert all(c['status']=='SKIPPED' for c in absent['cases'].values())
        bad=image_dir/'bad-python.sif';bad.write_text('bad test placeholder')
        fallback=run_case(dict(cfg,image=''));assert fallback['cases']['container']['status']=='PASS';assert fallback['container_discovery']['attempts']
        load=run_case(dict(cfg,load_scale=.5));assert load['cases']['container_numpy']['status'] in ('PASS','SKIPPED')
        result['checks']['wrong_hash_rejected']='PASS';result['checks']['attempted_incompatible_images_FAIL']='PASS';result['checks']['missing_image_SKIPPED']='PASS';result['checks']['fallback_to_compatible_candidate']='PASS';result['checks']['changed_load_scale']='PASS'
        # Replay from an extracted ZIP remains independent of notebook memory.
        archive=ns['archive'];ex=temp/'extracted';ex.mkdir()
        import zipfile
        with zipfile.ZipFile(archive) as z:z.extractall(ex)
        replay=subprocess.run([sys.executable,str(ex/'replay.py')],cwd=ex,capture_output=True,text=True,timeout=20);assert replay.returncode==0,replay.stdout+replay.stderr
        result['checks']['extracted_handover_replay']='PASS'
    finally:os.chdir(old)
result['actual_apptainer_execution']='NOT RUN: no runtime or real SIF in the local author environment'
(BASE/'v3_validation_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v if isinstance(v,str) else v.get('status') for k,v in result['checks'].items()},indent=2))
