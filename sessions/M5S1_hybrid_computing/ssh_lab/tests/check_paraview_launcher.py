"""Test batch selection/environment and failure propagation, without a renderer."""
from pathlib import Path
import array,json,os,shutil,subprocess,tempfile
lab=Path(__file__).resolve().parents[1]
for executable,exit_code in [('pvbatch',0),('pvpython',0),('pvbatch',11)]:
 with tempfile.TemporaryDirectory(prefix='m5s1-pv-launch-') as tmp:
  root=Path(tmp);(root/'bin').mkdir();shutil.copytree(lab/'scripts',root/'scripts')
  # Only process orchestration is mocked. The VTK exporter reads a real field.
  field=root/'field.bin';field.write_bytes(array.array('d',[100.,100.,100.,0.,25.,0.,0.,0.,0.]).tobytes())
  init=root/'init.sh';init.write_text('module() { if [[ "$*" == *avail* ]]; then echo "ParaView/test" >&2; fi; }\n')
  for name,content in {
   'scontrol':'exit 0',
   executable:'''if [[ "$1" == --version ]];then echo 'mock launcher';exit 0;fi
python3 - <<'CHECK'
import os,json
keys=['LIBGL_ALWAYS_SOFTWARE','GALLIUM_DRIVER','EGL_PLATFORM','LP_NUM_THREADS']
assert [os.environ[k] for k in keys]==['true','llvmpipe','surfaceless','1']
assert not any(k in os.environ for k in ['DISPLAY','WAYLAND_DISPLAY','PV_DEBUG_REMOTE_RENDERING'])
print('LAUNCH_ENV_PASS')
CHECK
exit '''+str(exit_code),
  }.items():
   p=root/'bin'/name;p.write_text('#!/bin/bash\n'+content+'\n');p.chmod(0o755)
  env=os.environ.copy();env.update(PATH=str(root/'bin')+':'+env['PATH'],BASH_ENV=str(init),SLURM_SUBMIT_DIR=str(root),SLURM_JOB_ID='123',SLURM_CPUS_PER_TASK='1',HEAT_INPUT=str(field),N='3',DISPLAY=':99',WAYLAND_DISPLAY='wayland-0',PV_DEBUG_REMOTE_RENDERING='1')
  # Hide host ParaView executables, if installed, to exercise the fallback exactly.
  if executable=='pvpython' and shutil.which('pvbatch'):continue
  run=subprocess.run(['bash',str(root/'scripts/paraview.sbatch')],env=env,capture_output=True,text=True)
  assert run.returncode==exit_code,(run.returncode,run.stdout,run.stderr)
  assert 'LAUNCH_ENV_PASS' in run.stdout,run.stdout+run.stderr
  assert ('PARAVIEW_DONE' in run.stdout)==(exit_code==0),run.stdout
  assert f'PARAVIEW_EXECUTABLE={root}/bin/{executable}' in run.stdout
script=(lab/'scripts/paraview_heat.py').read_text()
import ast
calls=[n.func for n in ast.walk(ast.parse(script)) if isinstance(n,ast.Call)]
assert not any(isinstance(f,ast.Name) and f.id=='Render' for f in calls)
assert any(isinstance(f,ast.Attribute) and f.attr=='StillRender' for f in calls)
print('PASS launcher selection, software environment, failure propagation and interactor-free render call. No rendering was simulated or verified.')
