"""Compare every query's complete ordered top-k result; no timing/speedup acceptance.
Usage: python3 check_topk.py serial_reference.json candidate.json
"""
import json, math, sys

def check(reference, candidate):
    for key in ['n','d','queries','k','ties']:
        assert candidate[key] == reference[key], f'changed problem: {key}'
    assert math.isfinite(candidate['seconds']) and candidate['seconds']>0, 'invalid timing'
    assert len(candidate['hits']) == reference['queries'], 'missing queries'
    max_error = 0.0
    for q,(expected,actual) in enumerate(zip(reference['hits'],candidate['hits'])):
        assert len(actual)==reference['k'], f'wrong K at query {q}'
        assert len({h['id'] for h in actual})==len(actual), f'duplicate ID at query {q}'
        for i,(e,a) in enumerate(zip(expected,actual)):
            assert type(a['id']) is int and 0<=a['id']<reference['n'], 'invalid ID'
            assert math.isfinite(a['score']), 'non-finite score'
            assert a['id']==e['id'], f'query {q}, position {i}: ID {a["id"]} != {e["id"]}'
            err=abs(a['score']-e['score']);max_error=max(max_error,err)
            assert math.isclose(a['score'],e['score'],rel_tol=1e-10,abs_tol=1e-10), f'score mismatch: {err}'
            if i:
                prev=actual[i-1]
                assert prev['score']>a['score'] or (prev['score']==a['score'] and prev['id']<a['id']), 'unordered top-k'
    return max_error

if __name__=='__main__':
    try:
        if len(sys.argv)!=3:raise ValueError('usage: check_topk.py reference.json candidate.json')
        reference=json.load(open(sys.argv[1]));candidate=json.load(open(sys.argv[2]))
        error=check(reference,candidate)
        print(f'PASS: every query and all ordered top-k IDs/scores; max error={error:.3g}')
    except Exception as e:
        print('FAIL:',e,file=sys.stderr);sys.exit(1)
