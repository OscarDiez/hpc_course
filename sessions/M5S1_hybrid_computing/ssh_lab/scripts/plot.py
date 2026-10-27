"""Plot saved evidence, after validate.py. Run on Jupyter or a CPU allocation."""
from pathlib import Path
import csv,sys,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
folder=Path(sys.argv[1]);records=list(csv.DictReader((folder/'measurements.csv').open()))
records=[r for r in records if r['scientific_status']!='EXPECTED_MISMATCH']
if not records:raise SystemExit('NOT RUN: no correct measured results')
n=int(records[0]['n']);ref=np.fromfile(folder/'reference-r0.bin',dtype=np.float64).reshape(n,n)
fig,axes=plt.subplots(1,3,figsize=(15,4))
im=axes[0].imshow(ref,cmap='inferno',vmin=0,vmax=100);axes[0].set(title=f'Final temperature, N={n}',xlabel='x cell',ylabel='y cell')
fig.colorbar(im,ax=axes[0],label='°C')
axes[1].plot(ref[:,n//2],np.arange(n));axes[1].invert_yaxis();axes[1].set(title='Centre column',xlabel='Temperature (°C)',ylabel='y cell')
modes=list(dict.fromkeys(r['mode'] for r in records))
for x,mode in enumerate(modes):
 vals=[float(r['loop_s']) for r in records if r['mode']==mode]
 axes[2].scatter([x]*len(vals),vals,color='grey');axes[2].plot(x,statistics.median(vals),'o',color='navy')
axes[2].set(xticks=range(len(modes)),xticklabels=modes,ylabel='Measured loop seconds',title='Actual runs: dots and medians')
if all(float(r['loop_s'])>0 for r in records):
 axes[2].set_yscale('log')
 axes[2].set_ylabel('Measured loop seconds (log scale)')
axes[2].tick_params(axis='x',rotation=45)
fig.tight_layout();fig.savefig(folder/'evidence.png',dpi=160)
print(folder/'evidence.png')
