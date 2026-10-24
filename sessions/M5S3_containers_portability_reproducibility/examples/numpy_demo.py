"""Small stiffness-system solve, on CPU only; no package installation."""
import importlib.util, json, sys
if importlib.util.find_spec('numpy') is None:
    print('NumPy is not installed in this image.'); raise SystemExit(77)
import numpy as np
scale=float(sys.argv[1])
k=np.array([[12.,-4.,2.],[-4.,8.,-2.],[2.,-2.,6.]])
f=scale*np.array([5.,6.,7.])
d=np.linalg.solve(k,f)
print(json.dumps({'scale':scale, 'forces':f.tolist(), 'displacements':d.tolist(),
                  'residual_inf':float(np.max(np.abs(k@d-f))), 'numpy':np.__version__}))
