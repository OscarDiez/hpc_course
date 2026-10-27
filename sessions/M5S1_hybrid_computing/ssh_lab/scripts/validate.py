"""Full-field numerical checks; no third-party packages required."""
import argparse, array, math
p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('n',type=int)
p.add_argument('--expect-different',action='store_true');args=p.parse_args();n=args.n

def read(path):
 a=array.array('d');a.frombytes(open(path,'rb').read())
 if len(a)!=n*n:raise SystemExit(f'FAIL {path}: expected {n*n} doubles, got {len(a)}')
 if not all(math.isfinite(x) for x in a):raise SystemExit('FAIL: non-finite temperature')
 return a
ref=read(args.reference);got=read(args.candidate)
err=max(abs(a-b) for a,b in zip(ref,got))
boundaries=(all(abs(got[x]-100)<1e-10 for x in range(n)) and
 all(abs(got[(n-1)*n+x])<1e-10 for x in range(n)) and
 all(abs(got[y*n])<1e-10 and abs(got[y*n+n-1])<1e-10 for y in range(1,n-1)))
bounded=min(got)>=-1e-10 and max(got)<=100+1e-10
matches=err<=1e-10 and boundaries and bounded
if args.expect_different:
 # The v2 Kelvin example is an intentionally incorrect Celsius field.
 if err<=1e-10:raise SystemExit('FAIL: expected the incompatible library to change the field')
 print(f'EXPECTED MISMATCH (not a correct heat solution): max_abs_error={err:.6g}, boundaries_ok={boundaries}')
else:
 if not matches:raise SystemExit(f'FAIL max_abs_error={err:.6g} boundaries_ok={boundaries} bounded={bounded}')
 print(f'PASS full field; max_abs_error={err:.6g}; boundaries and range correct')
