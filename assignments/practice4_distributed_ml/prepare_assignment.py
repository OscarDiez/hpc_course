"""Extract the validated teaching sources and add the individual exercise hooks."""
import ast, hashlib, json, pathlib, re, sys
root=pathlib.Path.cwd();nb=json.loads((root/'guided_source.ipynb').read_text());values={}
for cell in nb['cells']:
    if cell['cell_type']=='code' and 'assets' in cell.get('metadata',{}).get('tags',[]):
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id in ('c_source','ml_source'):
                    values[node.targets[0].id]=ast.literal_eval(node.value)
for cell in nb['cells']:
    if cell['cell_type']=='code' and 'write-files' in cell.get('metadata',{}).get('tags',[]):
        node=ast.parse(''.join(cell['source'])).body[0]
        values['batch_source']=ast.literal_eval(node.value)
assert set(values)=={'c_source','ml_source','batch_source'}
s=values['ml_source']
def replace_once(old,new):
    global s
    if s.count(old)!=1:raise RuntimeError('Validated source changed at '+old[:70])
    s=s.replace(old,new)
replace_once('def update(m,opt,x,y):','def update(m,opt,x,y,factor=1.0):')
replace_once('loss=nn.functional.cross_entropy(m(x),y)','loss=nn.functional.cross_entropy(m(x),y)*factor')
replace_once('ids=batch_ids(step,len(x),b);local=ids.chunk(world)[rank]\n            update(ddp,opt,x[local],y[local])',
'''ids=batch_ids(step,len(x),b)
            layout=os.getenv('M4S5_LAYOUT','balanced')
            if layout not in ('balanced','unequal_naive','unequal_corrected'):
                raise ValueError('Unknown layout '+layout)
            if layout=='balanced' or world==1:
                chunks=list(ids.chunk(world))
            else:
                if world!=2 or b%4: raise ValueError('Unequal exercise requires two ranks and global batch divisible by four')
                cut=3*b//4
                chunks=[ids[:cut],ids[cut:]]
            local=chunks[rank]
            factor=1.0
            if layout=='unequal_corrected':
                import math
                from student import loss_multiplier
                factor=float(loss_multiplier(len(local),b,world))
                if not math.isfinite(factor): raise ValueError('Loss multiplier must be finite')
            update(ddp,opt,x[local],y[local],factor)''')
replace_once("'mode':args.mode,'world_size':world,'local_batch':b//world,'global_batch':b,'steps':steps,",
"'mode':args.mode,'layout':layout,'world_size':world,'local_batch':len(local),'rank_sample_counts':[len(c) for c in chunks],'global_batch':b,'steps':steps,")
(root/'ml_lab.py').write_text(s)
(root/'gradient.c').write_text(values['c_source'])
# The same CPU resources, compiler setup and shared image as the validated lab.
batch=values['batch_source']
for mode in ('baseline','full'):
    new=batch.replace('--job-name=m4s5-guided',f'--job-name=m4s5-{mode}')
    new=new.replace('python3 /work/ml_lab.py --mode workflow',f'python3 /work/run_experiments.py --mode {mode}')
    new=new.replace('cat SUMMARY.txt','cat ASSIGNMENT_SUMMARY.txt')
    (root/f'run_{mode}.sbatch').write_text(new)
digits=''.join(re.findall(r'\d',sys.argv[1]))
if not digits:raise ValueError('Student identifier must contain at least one digit')
seed=2026+int(digits[-4:])
(root/'config.json').write_text(json.dumps({'samples':4096,'epochs':6,'global_batch':128,'seed':seed},indent=2)+'\n')
(root/'provenance.json').write_text(json.dumps({'guided_source_commit':'616f5fbe63e2e0d63fbecbcae0232defd44d72ce',
 'guided_notebook_sha256':hashlib.sha256((root/'guided_source.ipynb').read_bytes()).hexdigest(),
 'seed_rule':'2026 + final four numeric digits of student ID','seed':seed},indent=2)+'\n')
print('Created assignment files in',root)
print('Personal seed:',seed)
print('First run: sbatch run_baseline.sbatch')
print('After implementing student.py: sbatch run_full.sbatch')
