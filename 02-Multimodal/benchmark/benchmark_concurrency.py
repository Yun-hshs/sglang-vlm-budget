"""Concurrency is controlled by configs/benchmark.yaml, not server batch size."""
from benchmark.benchmark_latency import run
import argparse
if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True)
    p.add_argument('--config',default='configs/benchmark.yaml');p.add_argument('--output',required=True)
    run(p.parse_args())
