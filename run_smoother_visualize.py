#!/usr/bin/env python3
"""
Load a Hybrid A* result, run the EmbodySmoother, and visualize the result.

Usage:
  python run_smoother_visualize.py [case_id] [--save]

Examples:
  python run_smoother_visualize.py          # case 1, interactive plot
  python run_smoother_visualize.py 15 --save  # case 15, save to Figure/
"""
import sys
import os
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.join(REPO_ROOT, 'occ_planner')
sys.path.insert(0, CORE_DIR)

from down_sample import down_sample
from Optimize_util import downsample_trajectory
from TPCAP_Cases import Case
from map_test import MakeGridMap
from HybridAstar import HybridAstar
from Smoother_v2 import MakeCorridor, EmbodySmoother

FIGURE_DIR = os.path.join(REPO_ROOT, 'Figure', 'LIOM_Figure')


def run(case_id: int, show: bool = True, save: bool = False):
    os.makedirs(FIGURE_DIR, exist_ok=True)

    res = 0.1
    csv_path = os.path.join(REPO_ROOT, 'benchmarks', 'LIOM', f'Case{case_id}.csv')
    result_path = os.path.join(REPO_ROOT, 'results', 'PlanningRes',
                               f'LIOM_Case_{case_id}_Hybrid_A_star.npy')

    case = Case(csv_path, discrete_size=0.01, MapgridSize=res)
    PlanningResult = np.load(result_path)

    dist_1 = np.linalg.norm(case.GetGoal()[:-1] - PlanningResult[0, :2])
    dist_2 = np.linalg.norm(case.GetStart()[:-1] - PlanningResult[0, :2])
    reversed = False if dist_1 > dist_2 else True

    PlanningResult = downsample_trajectory(PlanningResult, target_num=100)
    if reversed:
        PlanningResult = PlanningResult[::-1]

    grid_binary = MakeGridMap(case, grid_size=res)
    Cor = HybridAstar(72)
    Cor.Init(grid_binary, res, res, case.xmax, case.ymax, case.xmin, case.ymin)

    init_res, init_control, Tf = down_sample(PlanningResult)
    HalfSpace = MakeCorridor(PlanningResult, Cor)

    smoother = EmbodySmoother(init_res, HalfSpace)

    start = time.time()
    smoother.IterativeSolve(Cor)
    elapsed = time.time() - start

    PlanningRes = np.hstack((
        smoother.x_opt, smoother.y_opt,
        smoother.theta_opt, smoother.v_opt,
        smoother.steer_opt
    ))
    ContorlRes = smoother.t_opt[-1].toarray()

    print(f"\nSmoothed: {elapsed:.1f}s | Tf = {float(smoother.TF_opt):.2f}s")
    case.ShowRes(PlanningResult, PlanningRes, ContorlRes, case_id,
                 show=show, save=save, Benchmark='LIOM')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    flags = set(a for a in sys.argv[1:] if a.startswith('--'))
    case_id = int(args[0]) if args else 1
    run(case_id, show=('--save' not in flags), save=('--save' in flags))
