#!/usr/bin/env python3
"""
Generate a GIF animation of the smoothed trajectory.
Usage: python scripts/make_teaser_gif.py [case_id]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'occ_planner'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from TPCAP_Cases import Case
from matplotlib.patches import Polygon as MplPolygon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE = os.path.join(REPO, 'Figure')

def make_gif(case_id=1, interval=80):
    os.makedirs(FIGURE, exist_ok=True)

    path = np.load(os.path.join(REPO, f'results/PlanningRes/LIOM_Case_{case_id}_Hybrid_A_star.npy'))
    case = Case(os.path.join(REPO, f'benchmarks/LIOM/Case{case_id}.csv'), 0.01, 0.1)

    veh = __import__('KinematicModel', fromlist=['Vehicle']).Vehicle()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')

    for obs in case.obs:
        ax.fill(obs[:,0], obs[:,1], facecolor='k', alpha=0.5)

    ax.arrow(case.x0, case.y0, np.cos(case.theta0), np.sin(case.theta0),
             width=0.2, color='gold')
    ax.arrow(case.xf, case.yf, np.cos(case.thetaf), np.sin(case.thetaf),
             width=0.2, color='gold')

    traj_line, = ax.plot([], [], 'b-', linewidth=1.5, alpha=0.6, label='Trajectory')
    veh_patch = ax.fill([], [], facecolor='SteelBlue', edgecolor='navy', linewidth=1.5, alpha=0.9)[0]
    start_poly = MplPolygon(veh.create_polygon(case.x0, case.y0, case.theta0),
                             closed=True, fill=False, edgecolor='green', linestyle='--', linewidth=1.5)
    goal_poly = MplPolygon(veh.create_polygon(case.xf, case.yf, case.thetaf),
                            closed=True, fill=False, edgecolor='red', linestyle='--', linewidth=1.5)
    ax.add_patch(start_poly)
    ax.add_patch(goal_poly)

    margin = 3
    all_x = [p[0] for p in path]
    all_y = [p[1] for p in path]
    for obs in case.obs:
        all_x.extend(obs[:,0]); all_y.extend(obs[:,1])
    ax.set_xlim(min(all_x)-margin, max(all_x)+margin)
    ax.set_ylim(min(all_y)-margin, max(all_y)+margin)
    ax.legend(fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    fig.tight_layout()

    skip = max(1, len(path)//60)
    frames = list(range(0, len(path), skip))

    def update(i):
        traj_line.set_data(path[:i+1,0], path[:i+1,1])
        poly = veh.create_polygon(path[i,0], path[i,1], path[i,2])
        veh_patch.set_xy(poly)
        return traj_line, veh_patch

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
    gif_path = os.path.join(FIGURE, f'teaser_case{case_id}.gif')
    anim.save(gif_path, writer='pillow', fps=15, dpi=100)
    plt.close()
    print(f'GIF saved: {gif_path}')

if __name__ == '__main__':
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    make_gif(cid)
