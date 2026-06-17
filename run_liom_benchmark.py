#!/usr/bin/env python3
"""
Reproduce all LIOM benchmark cases using Hybrid A* search.

Requirements:
  - Python 3.9 (x86_64 Linux) — required by the prebuilt HybridAstar .so module
  - Packages in requirements.txt

Usage:
  python run_liom_benchmark.py

Output:
  Saves .npy path files to results/PlanningRes/
"""
import sys
import os
import glob

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(REPO_ROOT, 'occ_planner')
sys.path.insert(0, CORE_DIR)

from HybridAstarpy import Hybrid_A_Star
from TPCAP_Cases import Case
from map_test import MakeGridMap
from HybridAstar import HybridAstar
from config.read_config import read_config

BENCHMARK_DIR = os.path.join(REPO_ROOT, 'benchmarks', 'LIOM')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'PlanningRes')

SKIP_CASES = [7]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    config = read_config('config')
    csv_files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, 'Case*.csv')))
    csv_files = [f for f in csv_files if int(os.path.basename(f).replace('Case', '').replace('.csv', '')) not in SKIP_CASES]

    total = len(csv_files)
    success = 0
    failed = []

    for idx, csv_path in enumerate(csv_files, 1):
        case_id = int(os.path.basename(csv_path).replace('Case', '').replace('.csv', ''))
        print(f"\n[LIOM] Processing Case {case_id} ({idx}/{total})")

        res = 0.1
        case = Case(csv_path, discrete_size=0.01, MapgridSize=res)

        grid_binary = MakeGridMap(case, grid_size=res)
        cor = HybridAstar(72)
        cor.Init(grid_binary, res, res, case.xmax, case.ymax, case.xmin, case.ymin)

        planner = Hybrid_A_Star(cor, config)
        status, path, direction = planner.Search(case.GetStart(), case.GetGoal())

        if status:
            path_arr = np.asarray(path)
            direction_arr = np.asarray(direction)
            np.save(os.path.join(OUTPUT_DIR, f'LIOM_Case_{case_id}_Hybrid_A_star.npy'), path_arr)
            np.save(os.path.join(OUTPUT_DIR, f'LIOM_Case_{case_id}_Hybrid_A_star_Direction.npy'), direction_arr)
            print(f"  PASS ({len(path_arr)} points)")
            success += 1
        else:
            print(f"  FAIL — no path found")
            failed.append(case_id)

    print(f"\n{'='*50}")
    print(f"LIOM Benchmark Complete: {success}/{total} passed")
    if failed:
        print(f"Failed cases: {failed}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
