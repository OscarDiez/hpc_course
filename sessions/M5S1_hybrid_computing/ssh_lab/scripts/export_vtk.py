"""Export the native-double field to an ASCII VTK structured-points file."""
from pathlib import Path
import array,math,sys

def export(source,n,target):
 # Native doubles are in row-major order: x changes fastest, as VTK expects.
 a=array.array('d');a.frombytes(Path(source).read_bytes())
 if len(a)!=n*n or not all(math.isfinite(v) for v in a):raise ValueError('Invalid field')
 # Add geometry and a scalar name while preserving each temperature value.
 with Path(target).open('w') as f:
  f.write(f'# vtk DataFile Version 3.0\nHeat diffusion, Celsius\nASCII\nDATASET STRUCTURED_POINTS\nDIMENSIONS {n} {n} 1\nORIGIN 0 0 0\nSPACING 1 1 1\nPOINT_DATA {n*n}\nSCALARS Temperature double 1\nLOOKUP_TABLE default\n')
  for v in a:f.write(f'{v:.17g}\n')
 return {'n':n,'points':len(a),'minimum':min(a),'maximum':max(a)}
if __name__=='__main__':print(export(sys.argv[1],int(sys.argv[2]),sys.argv[3]))
