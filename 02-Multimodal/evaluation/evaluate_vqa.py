import argparse
import json
from pathlib import Path
from evaluation.metrics import score
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);a=p.parse_args()
    rows=[json.loads(x) for x in Path(a.predictions).read_text().splitlines() if x.strip()]
    if not rows: raise ValueError('No predictions')
    values=[score(r['prediction'],r['answers'],r.get('metric','em')) for r in rows]
    print(json.dumps({'count':len(values),'mean_score':sum(values)/len(values),'normalizer':'project_lightweight_v1'},indent=2))
