"""Portable result validation, used unchanged by native Python and a Python SIF.
This validates native C outputs; it does not claim C execution inside the image.
"""
from pathlib import Path
import argparse, hashlib, json, math, os, platform, sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def require(condition,message):
    if not condition:raise ValueError(message)

def compare(reference,candidate):
    for key in ['n','d','queries']:
        require(candidate.get(key)==reference.get(key),'changed problem: '+key)
    topk='k' in reference
    require(('k' in candidate)==topk,'changed result format')
    if topk:
        for key in ['k','ties']:require(candidate.get(key)==reference.get(key),'changed problem: '+key)
    q=reference['queries'];k=reference['k'] if topk else 1
    require(len(reference['hits'])==q and len(candidate['hits'])==q,'missing queries')
    require(math.isfinite(candidate['seconds']) and candidate['seconds']>0,'invalid recorded time')
    error=0.0
    for expected,actual in zip(reference['hits'],candidate['hits']):
        expected=expected if topk else [expected];actual=actual if topk else [actual]
        require(len(expected)==k and len(actual)==k,'wrong neighbour count')
        require(len({h['id'] for h in actual})==k,'duplicate neighbour ID')
        for position,(e,a) in enumerate(zip(expected,actual)):
            require(type(a['id']) is int and 0<=a['id']<reference['n'],'invalid neighbour ID')
            require(a['id']==e['id'],'ordered neighbour ID mismatch')
            require(math.isfinite(a['score']) and math.isfinite(e['score']),'non-finite score')
            error=max(error,abs(a['score']-e['score']))
            require(math.isclose(a['score'],e['score'],rel_tol=1e-10,abs_tol=1e-10),'score mismatch')
            if position:
                previous=actual[position-1]
                require(previous['score']>a['score'] or (previous['score']==a['score'] and previous['id']<a['id']),'incorrect tie/order rule')
    return dict(n=reference['n'],d=reference['d'],queries_checked=q,neighbours_per_query=k,
                max_score_error=error,recorded_search_s=candidate['seconds'],
                recorded_queries_per_s=q/candidate['seconds'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    record=dict(status='FAIL',provenance=dict(python=sys.version.split()[0],executable=sys.executable,
                 platform=platform.platform(),host=platform.node(),
                 marker=os.environ.get('HPC_M5S5_HOST_SENTINEL'),script_sha256=sha(__file__)))
    try:
        record['inputs']=dict(reference_sha256=sha(args.reference),candidate_sha256=sha(args.candidate))
        record['result']=compare(json.loads(Path(args.reference).read_text()),json.loads(Path(args.candidate).read_text()))
        record['status']='PASS'
    except Exception as e:record['reason']=str(e)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    return 0 if record['status']=='PASS' else 1

if __name__=='__main__':sys.exit(main())
