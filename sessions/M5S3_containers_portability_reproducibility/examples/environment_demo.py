import json, os, platform, sys
print(json.dumps({'python':sys.version.split()[0], 'executable':sys.executable,
                  'kernel':platform.release(), 'machine':platform.machine(),
                  'omp_threads':os.environ.get('OMP_NUM_THREADS','unset')}))
