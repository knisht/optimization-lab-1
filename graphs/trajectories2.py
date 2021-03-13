from math import log, floor
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import hashlib

from graphs.trajectories import __plot_trajectory
from optimize.multidimensional.OptimizationResult import OptimizationResult


def generate_level_markers(minvalue: float, maxvalue: float):
    amount = 10
    level_separator = (maxvalue - minvalue * 0.001) / float(amount)
    level_lines = [minvalue * 0.001 + level_separator * (1.1 ** i) * float(i) for i in range(amount)]
    if maxvalue < 0:
        level_separator = (maxvalue * 0.5 - minvalue) / float(amount)
        level_lines = [minvalue + level_separator * float(i) for i in range(amount)]
        level_separator *= 2
    return level_lines


color_roulette = ["b", "g", "r", "c", "m", "y", "k", "lime", "navy", "darkred", "gold"]

def hash(line: str) -> str:
    return str(int(hashlib.sha256(line.encode('utf-8')).hexdigest(), 16) % 10 ** 8)

def plot_trajectory(results: List[OptimizationResult]):
    any_result = results[0]
    all_trajectories = list(map(lambda t: t.trajectory, results))
    all_points = [point for trajectory in all_trajectories for point in trajectory]
    v_func = np.vectorize(lambda x, y: any_result.oracle.f(x, y))
    xl = min(list(map(lambda t: t[0], all_points)))
    xr = max(list(map(lambda t: t[0], all_points)))
    yl = min(list(map(lambda t: t[1], all_points)))
    yr = max(list(map(lambda t: t[1], all_points)))
    xrange = xr - xl
    yrange = yr - yl
    xx, yy = np.meshgrid(np.linspace(xl - xrange * 0.2, xr + xrange * 0.5, 100),
                         np.linspace(yl - yrange * 0.2, yr + yrange * 0.5, 100))

    fig, ax = plt.subplots()
    min_value = min(list(map(lambda p: any_result.oracle.f(*p), all_points)))
    max_value = max(list(map(lambda p: any_result.oracle.f(*p), all_points)))
    min_point = min(list(map(lambda p: p.optimal_point, results)), key=lambda x: any_result.oracle.f(*x))
    qx = ax.contour(xx, yy, v_func(xx, yy), generate_level_markers(min_value, max_value),
                    linestyles='solid')
    ax.plot(min_point[0], min_point[1], color=color_roulette[0], marker='o', markersize=10)
    for trajectory, color, idx in zip(all_trajectories, color_roulette[1:], range(len(results))):
        __plot_trajectory(ax, trajectory, str(trajectory[0]) + f" {results[idx].name}", color)
    ax.clabel(qx, fontsize=5, fmt='%.2f', inline=1)
    ax.legend()
    ax.set_title(any_result.oracle.representation + ", " + str(any_result.trajectory[0]))
    plt.savefig(f"results/{hash(any_result.oracle.representation)}__{hash(str(any_result.trajectory[0]))}.png", dpi=300)
    plt.show()
