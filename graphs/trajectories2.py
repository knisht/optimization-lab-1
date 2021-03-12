from typing import List

import numpy as np
import matplotlib.pyplot as plt

from graphs.trajectories import __plot_trajectory
from optimize.multidimensional.OptimizationResult import OptimizationResult


def generate_level_markers(minvalue: float, maxvalue: float):
    level_separator = (maxvalue - minvalue * 0.9) / 15.0
    return [minvalue + level_separator * float(i) for i in range(15)]


color_roulette = ["b", "g", "r", "c", "m", "y", "k", "lime", "navy", "darkred", "gold"]


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
    for trajectory, color in zip(all_trajectories, color_roulette[1:]):
        __plot_trajectory(ax, trajectory, str(trajectory[0]), color)
    ax.clabel(qx, fontsize=5, fmt='%.2f', inline=1)
    ax.legend()
    ax.set_title(any_result.oracle.representation + ", " + any_result.name)
    plt.savefig("results/" + any_result.name + ".png", dpi=300)
    plt.show()
