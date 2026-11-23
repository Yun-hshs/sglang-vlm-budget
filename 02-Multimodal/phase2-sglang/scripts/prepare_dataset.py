"""Deterministic subset of an authorized local manifest; no silent public data downloads."""
import argparse,json,random,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
p.add_argument('--count',type=int,default=100);p.add_argument('--seed',type=int,default=20261111)
a=p.parse_args();source=Path(a.input).resolve();target=Path(a.output).resolve()
rows=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
if not 0<a.count<=len(rows):raise ValueError('Count must be within dataset size')
rows=sorted(rows,key=lambda r:str(r['id']));random.Random(a.seed).shuffle(rows);rows=rows[:a.count]
for r in rows:r['images']=[str((source.parent/x).resolve()) for x in r['images']]
target.parent.mkdir(parents=True,exist_ok=True)
target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
target.with_suffix('.provenance.json').write_text(json.dumps({'seed':a.seed,'count':len(rows),
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'ids':[r['id'] for r in rows]},indent=2))
