"""Device-wide sampled memory. Run on the server, not the Mac client."""
import argparse,csv,subprocess,time
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--interval',type=float,default=0.2);a=p.parse_args()
if a.interval<=0:raise ValueError('Positive interval required')
with open(a.output,'w') as f:
    w=csv.writer(f);w.writerow(['unix_s','gpu_uuid','memory_used_mib','utilization_pct'])
    try:
        while True:
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            for line in raw.splitlines():w.writerow([time.time(),*next(csv.reader([line]))])
            f.flush();time.sleep(a.interval)
    except KeyboardInterrupt:pass
