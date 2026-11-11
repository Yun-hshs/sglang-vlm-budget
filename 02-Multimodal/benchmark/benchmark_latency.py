"""Streaming workload runner. Run after server/model/environment are pinned."""
import argparse
import concurrent.futures
import hashlib
import itertools
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
import yaml
from api.client import VLMClient,user_message
from preprocessing.image_budget import prepare_image
from evaluation.metrics import score

def load_samples(path):
    path=Path(path).resolve(); samples=[]; ids=set()
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        s=json.loads(line)
        if s['id'] in ids: raise ValueError('Duplicate sample ID')
        ids.add(s['id'])
        s['images']=[str((path.parent/x).resolve()) for x in s['images']]
        if not s['images'] or not s['question'].strip() or not s['answers']: raise ValueError('Invalid sample')
        for img in s['images']:
            if not Path(img).is_file(): raise ValueError('Missing image: '+img)
        samples.append(s)
    if not samples: raise ValueError('Empty dataset')
    if len({(s.get('task','unknown'),s.get('metric','em')) for s in samples})!=1:
        raise ValueError('Run each task and metric separately; mixed quality scales are invalid')
    return samples

def request(sample,strategy,count,cache,max_tokens):
    start=time.perf_counter()
    row={'started_unix':time.time(),'sample_id':sample['id'],'task':sample.get('task','unknown'),'strategy':strategy,
         'image_count':count,'status':'error','score':None,'visual_tokens_actual':None,
         'gpu_peak_memory_mib':None,'result_kind':'real_server_unverified_hardware'}
    try:
        # Only authentic multi-image examples: never duplicate an image to fake scaling.
        images=[prepare_image(x,strategy,sample['question']) for x in sample['images']]
        row['preprocessing_s']=time.perf_counter()-start
        row['images']=[im.metadata for im in images]
        with VLMClient() as client:
            if cache=='cold': client.flush_cache()
            result=client.complete([user_message(sample['question'],images)],max_tokens)
        row.update(result.to_dict())
        row['score']=score(result.text,sample['answers'],sample.get('metric','em'))
        row['status']='ok'
    except Exception as exc:
        row['error']=type(exc).__name__+': '+str(exc)
    row['finished_unix']=time.time()
    row['client_total_s']=time.perf_counter()-start
    return row

def run(args):
    config=yaml.safe_load(Path(args.config).read_text()); samples=load_samples(args.dataset)
    if config['cache_mode']=='cold' and any(c!=1 for c in config['concurrency']):
        raise ValueError('Cold cache flush requires concurrency=1 on an isolated server')
    if any(c<1 for c in config['concurrency']) or config['repeats']<1: raise ValueError('Invalid workload')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    digests={x:hashlib.sha256(Path(x).read_bytes()).hexdigest() for s in samples for x in s['images']}
    manifest={'status':'REAL_SERVER_RUN','config':config,'model':os.getenv('VLM_MODEL','Qwen/Qwen2.5-VL-7B-Instruct'),
        'dataset_sha256':hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        'image_sha256':digests,'platform':platform.platform(),'started_unix':time.time(),
        'experiment_contract':yaml.safe_load(Path('configs/model.yaml').read_text()),
        'note':'TTFT includes network/server queue; TPOT estimated from stream and usage; GPU memory from external sampler.'}
    if 'REQUIRED' in json.dumps(manifest['experiment_contract']):
        raise ValueError('Pin model_revision and sglang_version in configs/model.yaml before real runs')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    groups=list(itertools.product(config['strategies'],config['image_counts'],config['concurrency']))
    random.Random(config['seed']).shuffle(groups)
    skipped=[]
    with (out/'requests.jsonl').open('w') as f:
        for strategy,count,concurrency in groups:
            eligible=[s for s in samples if len(s['images'])==count]
            if not eligible:
                skipped.append({'strategy':strategy,'image_count':count,'concurrency':concurrency,'reason':'no authentic samples'}); continue
            for i in range(config['warmup_requests']):
                warm=request(eligible[i%len(eligible)],strategy,count,config['cache_mode'],config['max_tokens'])
                if warm['status']!='ok': raise RuntimeError('Warmup failed: '+warm.get('error','unknown'))
            for repeat in range(config['repeats']):
                begin=time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                    rows=list(pool.map(lambda s: request(s,strategy,count,config['cache_mode'],config['max_tokens']),eligible))
                duration=time.perf_counter()-begin
                for row in rows:
                    row.update(concurrency=concurrency,repeat=repeat,cache_mode=config['cache_mode'],
                               group_wall_s=duration,group_requests=len(rows))
                    f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
    (out/'skipped.json').write_text(json.dumps(skipped,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True);parser.add_argument('--config',default='configs/benchmark.yaml')
    parser.add_argument('--output',required=True)
    run(parser.parse_args())
