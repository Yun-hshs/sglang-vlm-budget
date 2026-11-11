"""Plots only measured rows; does not manufacture absent token or memory values."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def percentile(values,q):
    v=sorted(values)
    if not v:return None
    pos=(len(v)-1)*q; i=int(pos); fraction=pos-i
    return v[i]*(1-fraction)+v[min(i+1,len(v)-1)]*fraction

def analyze(source,out):
    rows=[json.loads(l) for l in Path(source).read_text().splitlines() if l.strip()]
    if not rows:raise ValueError('No measured rows; no figures generated')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);groups=defaultdict(list)
    for r in rows:groups[(r['strategy'],r['image_count'],r['concurrency'],r['cache_mode'])].append(r)
    summary=[]
    for key,rs in groups.items():
        ok=[r for r in rs if r['status']=='ok']; n=len(ok)
        record=dict(zip(['strategy','image_count','concurrency','cache_mode'],key))
        record.update(truncated_rate=sum(r.get('finish_reason')=='length' for r in ok)/n if n else None,requests=len(rs),success_rate=n/len(rs),score=sum(r['score'] for r in ok)/n if n else None)
        for metric in ['ttft_s','tpot_s','e2e_s','client_total_s','visual_tokens_actual']:
            vals=[r[metric] for r in ok if r.get(metric) is not None]
            for q in [0.5,0.9,0.99]:record[metric+'_p'+str(round(q*100))]=percentile(vals,q)
        rounds={r['repeat']:r['group_wall_s'] for r in rs}
        tokens=[r.get('completion_tokens') for r in ok]
        record['output_tokens_per_s']=sum(tokens)/sum(rounds.values()) if n and all(t is not None for t in tokens) else None
        summary.append(record)
    with (out/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    for count in sorted({r['image_count'] for r in summary}):
        for c in sorted({r['concurrency'] for r in summary}):
            for cache in sorted({r['cache_mode'] for r in summary}):
                subset=[r for r in summary if r['image_count']==count and r['concurrency']==c and r['cache_mode']==cache and r['score'] is not None]
                if not subset:continue
                fig,ax=plt.subplots()
                for r in subset:ax.scatter(r['e2e_s_p50'],r['score'],label=r['strategy'])
                ax.set(xlabel='Median client E2E latency (s)',ylabel='Mean quality score',title=f'{count} images / concurrency {c} / {cache}')
                ax.legend();fig.tight_layout();fig.savefig(out/f'accuracy_latency_n{count}_c{c}_{cache}.png');plt.close(fig)
    # True visual-token data will be populated only after server instrumentation.
    valid=[r for r in rows if r['status']=='ok' and r.get('visual_tokens_actual') is not None]
    if valid:
        fig,ax=plt.subplots();ax.scatter([r['visual_tokens_actual'] for r in valid],[r['e2e_s'] for r in valid])
        ax.set(xlabel='Measured visual tokens',ylabel='Client E2E latency (s)');fig.savefig(out/'visual_tokens_latency.png');plt.close(fig)
    (out/'analysis_notes.json').write_text(json.dumps({'visual_token_plot':'generated' if valid else 'unavailable: no instrumentation',
        'gpu_memory':'Use benchmark/gpu_memory.py samples; device-wide sampled peak is NOT allocator peak.'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args();analyze(a.input,a.output)
