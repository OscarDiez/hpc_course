"""Collect actual RESULT rows; repeat 0 is a warmup, not an observation."""
from pathlib import Path
import csv,re,statistics,sys,subprocess
folder=Path(sys.argv[1]);rows=[]
for log in sorted(folder.glob('*-r*.log')):
 match=re.search(r'-r(\d+)\.log$',log.name)
 if not match or int(match[1])==0:continue
 lines=[s for s in log.read_text().splitlines() if s.startswith('RESULT ')]
 if len(lines)!=1:raise SystemExit(f'FAIL: expected exactly one RESULT in {log.name}; inspect MPI launcher')
 values=dict(piece.split('=',1) for piece in lines[0].split()[1:]);values['mode']=log.name.rsplit('-r',1)[0]
 values['repeat']=int(match[1]);values['log']=log.name;values['job_dir']=str(folder.resolve())
 check=[sys.executable,str(Path(__file__).with_name('validate.py')),str(folder/'reference-r0.bin'),str(log.with_suffix('.bin')),values['n']]
 if values['mode']=='library_v2':check.append('--expect-different')
 subprocess.run(check,check=True,capture_output=True,text=True)
 values['scientific_status']='EXPECTED_MISMATCH' if values['mode']=='library_v2' else 'PASS'
 rows.append(values)
if not rows:raise SystemExit('NOT RUN: no measured repeats found')
with (folder/'measurements.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for mode in sorted({r['mode'] for r in rows}):
 times=[float(r['loop_s']) for r in rows if r['mode']==mode]
 print(f'{mode:15} repeats={len(times)} median_loop_s={statistics.median(times):.9f} range=[{min(times):.9f}, {max(times):.9f}]')
print('Saved',folder/'measurements.csv')
