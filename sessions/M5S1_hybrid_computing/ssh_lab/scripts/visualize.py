"""Plots from saved doubles and CSV; standard library only, no Matplotlib required."""
import array,csv,html,math,statistics,sys
from pathlib import Path

def heat_svg(values,n,title='Temperature field',width=430):
 if len(values)!=n*n or not all(math.isfinite(x) for x in values):raise ValueError('Invalid heat field')
 stride=max(1,math.ceil(n/64));size=320;cell=size/math.ceil(n/stride)
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="410" viewBox="0 0 {width} 410">',
 '<rect width="100%" height="100%" fill="white"/>',f'<text x="20" y="25" font-family="sans-serif" font-size="18">{html.escape(title)}</text>']
 for row,y in enumerate(range(0,n,stride)):
  for col,x in enumerate(range(0,n,stride)):
   v=max(0,min(1,values[y*n+x]/100));color=f'rgb({int(255*v)},{int(90*v)},{int(150*(1-v))})'
   parts.append(f'<rect x="{20+col*cell:.2f}" y="{45+row*cell:.2f}" width="{cell+.1:.2f}" height="{cell+.1:.2f}" fill="{color}"/>')
 parts += ['<text x="20" y="390" font-family="sans-serif" font-size="13">0 °C (blue) to 100 °C (red). Top edge is y=0.</text>','</svg>']
 return ''.join(parts)

def write_plots(folder):
 folder=Path(folder);records=list(csv.DictReader((folder/'measurements.csv').open()))
 correct=[r for r in records if r['scientific_status']=='PASS']
 if not correct:raise ValueError('No validated records')
 n=int(correct[0]['n']);a=array.array('d');a.frombytes((folder/'reference-r0.bin').read_bytes())
 (folder/'heat.svg').write_text(heat_svg(a,n,f'Final temperature, N={n}'))
 # Centre-column profile from the actual saved reference, sampled only for display.
 points=' '.join(f'{40+a[y*n+n//2]*3.2:.2f},{40+300*y/(n-1):.2f}' for y in range(n))
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="440" height="400"><rect width="100%" height="100%" fill="white"/><text x="30" y="24" font-family="sans-serif" font-size="18">Centre-column temperature</text><path d="M40 40 V340 H360" fill="none" stroke="grey"/><polyline points="{points}" fill="none" stroke="#124b78" stroke-width="2"/><text x="40" y="370" font-family="sans-serif">0 °C</text><text x="300" y="370" font-family="sans-serif">100 °C</text><text x="5" y="43" font-family="sans-serif">0</text><text x="5" y="340" font-family="sans-serif">{n-1}</text></svg>'
 (folder/'profile.svg').write_text(svg)
 modes=list(dict.fromkeys(r['mode'] for r in correct));height=75+38*len(modes)
 chart=[f'<svg xmlns="http://www.w3.org/2000/svg" width="850" height="{height}"><rect width="100%" height="100%" fill="white"/><text x="20" y="24" font-family="sans-serif" font-size="18">Measured update-loop time: dots and median (seconds)</text>']
 all_times=[float(r['loop_s']) for r in correct];max_time=max(all_times) or 1
 for k,m in enumerate(modes):
  vals=[float(r['loop_s']) for r in correct if r['mode']==m];median=statistics.median(vals);y=55+k*38
  chart.append(f'<text x="20" y="{y+4}" font-family="sans-serif">{html.escape(m)}</text><path d="M165 {y} H{165+480*median/max_time:.2f}" stroke="#124b78" stroke-width="5"/>')
  for v in vals:chart.append(f'<circle cx="{165+480*v/max_time:.2f}" cy="{y}" r="3" fill="#7d8790"/>')
  chart.append(f'<text x="670" y="{y+4}" font-family="sans-serif" font-size="12">{median:.6g} [{min(vals):.3g}, {max(vals):.3g}]</text>')
 chart.append('</svg>');(folder/'timings.svg').write_text(''.join(chart))
 return [folder/'heat.svg',folder/'profile.svg',folder/'timings.svg']

if __name__=='__main__':
 for p in write_plots(sys.argv[1]):print(p)
