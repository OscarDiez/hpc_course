"""Submit frozen Slurm jobs once, wait, validate and keep a persistent run report."""
from pathlib import Path
import os,json,subprocess,time,shutil,csv,statistics,html

TERMINAL_BAD=('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED','REVOKED','DEADLINE')
ACTIVE={'PENDING','CONFIGURING','RUNNING','COMPLETING','SUSPENDED','REQUEUED','RESIZING','STAGE_OUT'}

def command(args,cwd=None,env=None,timeout=25):
 return subprocess.run(args,cwd=cwd,env=env,capture_output=True,text=True,timeout=timeout)
def save(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)
def state(job):
 if shutil.which('squeue'):
  q=command(['squeue','-h','-j',job,'-o','%T|%R'])
  if q.returncode==0 and q.stdout.strip():
   a,_,reason=q.stdout.strip().splitlines()[0].partition('|');return a.strip(),reason.strip(),None
 if shutil.which('sacct'):
  q=command(['sacct','-n','-P','-X','-j',job,'-o','State,ExitCode'])
  if q.returncode==0 and q.stdout.strip():
   parts=q.stdout.strip().splitlines()[0].split('|');return parts[0].strip().split()[0].rstrip('+'),'accounting',parts[1].strip() if len(parts)>1 else None
 return 'UNKNOWN','Queue/accounting unavailable',None

def execute(root,device,settings,emit=print):
 root=Path(root);marker=root/f'automation-{device}.json'
 if not shutil.which('sbatch'):return {'device':device,'status':'NOT RUN','reason':'sbatch unavailable; open on SciTech JupyterHub'}
 # Reuse a saved submission rather than creating another queued job.
 if marker.exists():
  record=json.loads(marker.read_text())
  if not record.get('job_id'):
   return dict(record,status='UNKNOWN',reason='Submission response uncertain. Inspect the saved folder and Slurm before retrying; no duplicate was sent.')
  emit(f"Reusing {device} job {record['job_id']}")
 else:
  # A separate immutable directory for each device/settings root, saved before submission.
  directory=root/'runs'/device;directory.mkdir(parents=True,exist_ok=True)
  for name in ('src','scripts'):shutil.copytree(root/name,directory/name,dirs_exist_ok=False,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  record={'device':device,'run_dir':str(directory),'status':'SUBMITTING','job_id':None}
  save(marker,record)
  # Slurm may reject conflicting inherited memory-request variables.
  env=os.environ.copy()
  for key in ('SLURM_MEM_PER_CPU','SLURM_MEM_PER_GPU','SLURM_MEM_PER_NODE'):env.pop(key,None)
  env.update(HEAT_INPUT=settings.get('heat_input',''),PARAVIEW_MODULE=settings.get('paraview_module',''),N=str(settings['n']),STEPS=str(settings['steps']),REPEATS=str(settings['repeats']),STAGE='all',MPI_LAUNCHER=settings.get('launcher','auto'))
  try:
   q=command(['sbatch','--parsable',f'scripts/{device}.sbatch'],cwd=directory,env=env,timeout=35)
  except (OSError,subprocess.TimeoutExpired) as exc:
   record.update(status='UNKNOWN',reason=f'Submission response uncertain: {exc}');save(marker,record);return record
  if q.returncode:
   record.update(status='FAIL',reason=q.stderr.strip() or q.stdout.strip());save(marker,record);return record
  job=q.stdout.strip().split(';')[0]
  if not job.isdigit():
   record.update(status='UNKNOWN',reason='Unexpected sbatch response: '+q.stdout);save(marker,record);return record
  record.update(job_id=job,status='SUBMITTED');save(marker,record)
  (directory/'job_id.txt').write_text(job+'\n')
  emit(f'Submitted {device} job {job}; folder {directory}')
 directory=Path(record['run_dir']);job=record['job_id'];log_path=directory/f'{device}-{job}.out'
 deadline=time.monotonic()+settings.get('wait_seconds',1800);offset=0;last=None;last_notice=0;completed_since=None
 final_marker={'cpu':'CPU_GUIDED_LAB_DONE','gpu':'GPU_GUIDED_LAB_DONE','profile':'PROFILE_DONE','paraview':'PARAVIEW_DONE'}[device]
 while True:
  try:
   st,reason,exit_code=state(job)
  except (OSError,subprocess.TimeoutExpired) as exc:st,reason,exit_code='UNKNOWN',str(exc),None
  log=log_path.read_text(errors='replace') if log_path.exists() else ''
  # Stream only newly written log text while the job runs.
  if len(log)>offset:
   emit(log[offset:].rstrip());offset=len(log)
  if st!=last or time.monotonic()-last_notice>=30:
   emit(f'{device} job {job}: {st} {reason}');last=st;last_notice=time.monotonic()
  # A completion-looking log must never overrule failed scheduler accounting.
  if st.startswith(TERMINAL_BAD) or (exit_code and exit_code!='0:0'):
   record.update(status='FAIL',reason=f'Slurm {st}, exit={exit_code}');break
  if device=='paraview' and 'SKIPPED_PARAVIEW:' in log and st not in ACTIVE:
   record.update(status='SKIPPED',reason=next(line for line in log.splitlines() if 'SKIPPED_PARAVIEW:' in line));break
  if device=='profile' and 'SKIPPED_PROFILE:' in log and st not in ACTIVE:
   record.update(status='SKIPPED',reason=next(line for line in log.splitlines() if 'SKIPPED_PROFILE:' in line));break
  if final_marker in log and st not in ACTIVE:
   if device=='paraview':
    evidence=directory/'results'/f'paraview-{job}'
    check=evidence/'validation.json'
    if not check.exists():record.update(status='FAIL',reason='Missing ParaView validation')
    else:
     data=json.loads(check.read_text());png=evidence/'paraview-temperature.png'
     valid=data.get('status')=='PASS' and data.get('points')==settings['n']**2 and png.exists() and png.read_bytes()[:8]==b'\x89PNG\r\n\x1a\n'
     if valid:record.update(status='PASS',paraview_dir=str(evidence),screenshot=str(png),accounting=st)
     else:record.update(status='FAIL',reason='ParaView field or screenshot check failed')
   elif device=='profile':
    profiles=list((directory/'results').glob('profile-*/gprof.txt'))
    if not profiles:record.update(status='FAIL',reason='Profile marker present but gprof.txt missing')
    else:record.update(status='PASS',profile=str(profiles[0]),accounting=st)
   else:
    evidence=directory/'results'/f'{device}-{job}'
    q=command(['python3','scripts/summarize.py',str(evidence)],cwd=directory,timeout=60)
    if q.returncode:record.update(status='FAIL',reason=q.stderr or q.stdout)
    else:
     rows=list(csv.DictReader((evidence/'measurements.csv').open()))
     # Require every expected mode, input and repetition before calling a run PASS.
     expected={'naive','serial_O0','serial','openmp','mpi','hybrid','library_v1','library_v2','library_fixed'} if device=='cpu' else {'serial','cuda'}
     present={r['mode'] for r in rows}
     count_ok=all(len([r for r in rows if r['mode']==m])==settings['repeats'] for m in present)
     inputs_ok=all(int(r['n'])==settings['n'] and int(r['steps'])==settings['steps'] for r in rows)
     # Incomplete experiments never become PASS merely because some fields exist.
     if not expected.issubset(present) or not count_ok or not inputs_ok:
      record.update(status='FAIL',reason='Missing modes, repeats or matching inputs')
     else:
      from visualize import write_plots
      plots=write_plots(evidence)
      from export_vtk import export
      export(evidence/'reference-r0.bin',settings['n'],evidence/'temperature.vtk')
      record.update(status='PASS',evidence=str(evidence),plots=[str(p) for p in plots],accounting=st,
                    skipped=['OpenACC compiler unavailable'] if device=='gpu' and 'openacc' not in present else [])
      emit(q.stdout.rstrip())
   if st=='UNKNOWN':emit('Numerical evidence checked; scheduler accounting unavailable. See saved log.')
   break
  if st=='COMPLETED':
   # Allow a brief shared-filesystem visibility delay before declaring the missing marker a failure.
   if completed_since is None:completed_since=time.monotonic()
   if time.monotonic()-completed_since>5:record.update(status='FAIL',reason='Job completed without its completion marker');break
  if time.monotonic()>=deadline:
   record.update(status='WAIT LIMIT',reason=f'{st}: notebook stopped waiting; job {job} was not resubmitted or cancelled');break
  time.sleep(settings.get('poll_seconds',5))
 save(marker,record);return record

def show_result(record,display,HTML):
 display(HTML(f'<h3>{html.escape(record["device"])} - {html.escape(record["status"])}</h3>'))
 if record.get('reason'):print(record['reason'])
 if record.get('job_id') and record.get('run_dir'):
  environment=Path(record['run_dir'])/'results'/f"{record['device']}-{record['job_id']}"/'environment.txt'
  if environment.exists():
   display(HTML('<details><summary>Modules, compiler and allocation</summary><pre>'+html.escape(environment.read_text(errors='replace')[:6000])+'</pre></details>'))
 if record.get('paraview_dir'):
  from IPython.display import Image
  display(Image(filename=record['screenshot']))
  print('ParaView files:',record['paraview_dir'])
 if record.get('profile'):
  print(Path(record['profile']).read_text()[:9000])
 if record.get('evidence'):
  print('VTK field:',Path(record['evidence'])/'temperature.vtk')
  rows=list(csv.DictReader((Path(record['evidence'])/'measurements.csv').open()))
  cells=['<table><tr><th>Mode</th><th>Layout</th><th>Repeats</th><th>Median loop (s)</th><th>Range (s)</th><th>Check</th></tr>']
  for mode in dict.fromkeys(r['mode'] for r in rows):
   subset=[r for r in rows if r['mode']==mode];r=subset[0];values=[float(v['loop_s']) for v in subset]
   cells.append('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [mode,f'{r["ranks"]} × {r["threads"]}',len(values),f'{statistics.median(values):.6g}',f'{min(values):.6g}-{max(values):.6g}',r['scientific_status']])+'</tr>')
  display(HTML(''.join(cells)+'</table>'))
 for path in record.get('plots',[]):display(HTML(Path(path).read_text()))
 for item in record.get('skipped',[]):print('SKIPPED:',item)


def show_final(report,display,HTML):
 """Display the actual ParaView image last, with an explicit GPU completion status."""
 records={r['device']:r for r in report}
 gpu=records.get('gpu',{})
 if gpu.get('status')=='PASS':
  rows=list(csv.DictReader((Path(gpu['evidence'])/'measurements.csv').open()))
  modes={r['mode'] for r in rows}
  print('CUDA heat solver: PASS; the GPU field matched its serial reference.')
  print('OpenACC heat solver: '+('PASS; field validated.' if 'openacc' in modes else 'SKIPPED; compiler unavailable.'))
 else:
  print('GPU experiment:',gpu.get('status','NOT RUN'))
  print(gpu.get('reason','No completed GPU validation is available.'))
 pv=records.get('paraview',{})
 display(HTML('<h3>ParaView result</h3>'))
 image_path=Path(pv['screenshot']) if pv.get('screenshot') else None
 if pv.get('status')=='PASS' and image_path and image_path.is_file():
  from IPython.display import Image
  print('Rendered by ParaView job',pv.get('job_id',''),'; input field and point count validated.')
  display(Image(filename=str(image_path)))
  print('Image:',image_path)
  print('VTI, centre-line CSV and saved state:',pv.get('paraview_dir',''))
 else:
  print('ParaView:',pv.get('status','NOT RUN'))
  print(pv.get('reason','No verified ParaView screenshot is available for this run.'))
  cpu=records.get('cpu',{})
  if cpu.get('status')=='PASS' and cpu.get('evidence'):
   print('You can open this saved field in desktop ParaView:',Path(cpu['evidence'])/'temperature.vtk')
