"""Draws the figures of the README from data/ and results/.

Usage:
    python make_figures.py      (after experiments.py and world_model.py)
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from world_model import WorldModel, rollout

STRATEGIES = {                      # name in the code -> label, color, marker
    "faithful": ("faithful copy", "#eb6834", "o"),
    "robot_style": ("robot style", "#2a78d6", "s"),
    "anticipatory": ("anticipatory", "#1baf7a", "^"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
})


def read(path):
    return list(csv.DictReader(open(path)))


def mean_by(rows, x_key, value):
    """{strategy: (sorted x values, mean of value for each x)}"""
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups[r["strategy"]][float(r[x_key])].append(value(r))
    return {s: (sorted(g), [np.mean(g[x]) for x in sorted(g)]) for s, g in groups.items()}


def plot_strategies(ax, curves, ylabel):
    for name, (label, color, marker) in STRATEGIES.items():
        x, y = curves[name]
        ax.plot(x, y, color=color, marker=marker, markersize=7, label=label)
    ax.set_ylabel(ylabel)


def fig_pushes():
    rows = read("data/pushes.csv")
    targets = sorted({int(r["target_deg"]) for r in rows})
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    for i, target in enumerate(targets):
        sizes = [-float(r["size_deg"]) for r in rows if int(r["target_deg"]) == target]
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(sizes)), sizes, s=36, color="#2a78d6",
                   edgecolor="#fcfcfb", linewidth=1)
    all_sizes = [-float(r["size_deg"]) for r in rows if int(r["target_deg"]) >= 180]
    ax.axhline(np.median(all_sizes), color=MUTED, linestyle="--", linewidth=1)
    ax.text(-0.4, np.median(all_sizes) + 3, f"median push, 180-540 deg clips: {np.median(all_sizes):.0f} deg",
            color=MUTED, ha="left")
    ax.set_xticks(range(len(targets)), [f"{t} deg" for t in targets])
    ax.set_xlabel("angle I turned the lid by")
    ax.set_ylabel("size of each push (deg)")
    ax.set_title("My push size does not grow with the total angle", loc="left")
    fig.tight_layout()
    fig.savefig("figures/pushes.png", dpi=150)


def fig_strategies():
    rows = read("results/episodes.csv")
    for r in rows:
        r["angle"] = r["clip"][6:9]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    plot_strategies(left, mean_by(rows, "angle", lambda r: float(r["time_s"])), "time (s)")
    plot_strategies(right, mean_by(rows, "angle", lambda r: abs(float(r["error_deg"]))),
                    "final error (deg)")
    for ax in (left, right):
        ax.set_xlabel("target angle (deg)")
    left.set_title("Time to turn the lid", loc="left")
    right.set_title("Distance to the target angle", loc="left")
    left.legend(frameon=False)
    fig.suptitle("Panda wrist (+-166 deg), mean over 15 clips x 6 starting wrist angles",
                 x=0.01, ha="left", color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/strategies.png", dpi=150)


def fig_wrist_range():
    rows = read("results/wrist_range.csv")
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8))
    plot_strategies(left, mean_by(rows, "wrist_limit", lambda r: float(r["time_s"])), "time (s)")
    plot_strategies(right, mean_by(rows, "wrist_limit", lambda r: abs(float(r["error_deg"]))),
                    "final error (deg)")
    right.set_yscale("log")
    for ax in (left, right):
        ax.set_xlabel("wrist range (+- deg)")
        ax.set_xticks([30, 45, 60, 90, 120, 166])
    left.set_title("Time to turn the lid", loc="left")
    right.set_title("Distance to the target angle (log scale)", loc="left")
    left.legend(frameon=False)
    fig.suptitle("360 and 540 deg clips, wrist starting at 0", x=0.01, ha="left",
                 color=MUTED, fontsize=9)
    fig.tight_layout()
    fig.savefig("figures/wrist_range.png", dpi=150)


def fig_world_model():
    model = WorldModel.load()
    data = np.load("results/histories.npz")
    history = data["geste_360.03|60|faithful"]       # a test episode, never seen in training
    start, horizon = 30, 100
    predicted, real = rollout(model, history, start, horizon)
    t = np.arange(1, horizon + 1) / 20
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(t, real, color=MUTED, label="simulation")
    ax.plot(t, predicted, color="#2a78d6", linestyle="--", label="world model, open loop")
    ax.set_xlabel("time after the last real observation (s)")
    ax.set_ylabel("lid rotation (deg)")
    ax.set_title("5 s imagined by the world model vs the simulator (faithful copy, test clip)", loc="left", fontsize=9.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/world_model.png", dpi=150)


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)
    fig_pushes()
    fig_strategies()
    fig_wrist_range()
    fig_world_model()
