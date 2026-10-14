"""Runs the three strategies on my clips, in two sweeps.

1. Starting wrist angle (Panda range): every clip, 6 starting angles.
   -> results/episodes.csv, and every episode's step-by-step history in
      results/histories.npz (used to train the world model).
2. Wrist range: the 360 and 540 degree clips, wrist starting at 0, with the
   wrist limited to +-30 ... +-166 degrees (166 is the real Panda).
   -> results/wrist_range.csv

Usage:
    python experiments.py
"""

import csv
import os

import numpy as np

from sim import LidSim
from strategies import anticipatory, faithful, load_pushes, load_total, robot_style

WRIST_STARTS = [-120, -60, 0, 60, 120, 150]   # degrees; 45 is the robosuite default
WRIST_LIMITS = [30, 45, 60, 90, 120, 166]      # degrees; a human hand turns a lid by about 55 per push
STRATEGIES = ["faithful", "robot_style", "anticipatory"]
CLIPS = sorted({r["clip"] for r in csv.DictReader(open("data/pushes.csv"))})


def run_episode(sim, strategy, clip, wrist_start, wrist_limit=166):
    sizes, durations = load_pushes(clip)
    target = load_total(clip)
    sim.reset(wrist_start, wrist_limit)
    if strategy == "faithful":
        faithful(sim, sizes, durations)
    elif strategy == "robot_style":
        robot_style(sim, target)
    else:
        anticipatory(sim, target)
    return {
        "clip": clip, "strategy": strategy, "wrist_start": wrist_start, "wrist_limit": wrist_limit,
        "target_deg": round(target, 1),
        "lid_deg": round(sim.lid_angle, 1),
        "error_deg": round(sim.lid_angle - target, 1),
        "time_s": sim.time(),
        "grasps": sim.grasps,
        "limit_hits": sim.wrist_limit_hits,
        "room_left_deg": round(sim.room_ccw(), 1),     # rotation still possible at the end
        "drift_mm": round(1000 * sim.lid_drift(), 1),
    }


def save(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def sweep_wrist_start(sim):
    rows, histories = [], {}
    for clip in CLIPS:
        for wrist_start in WRIST_STARTS:
            for strategy in STRATEGIES:
                rows.append(run_episode(sim, strategy, clip, wrist_start))
                histories[f"{clip}|{wrist_start}|{strategy}"] = np.array(sim.history, dtype=float)
                print(rows[-1])
    save(rows, "results/episodes.csv")
    np.savez_compressed("results/histories.npz", **histories)


def sweep_wrist_range(sim):
    rows = []
    for clip in [c for c in CLIPS if c[6:9] in ("360", "540")]:
        for wrist_limit in WRIST_LIMITS:
            for strategy in STRATEGIES:
                rows.append(run_episode(sim, strategy, clip, 0, wrist_limit))
                print(rows[-1])
    save(rows, "results/wrist_range.csv")


def main():
    os.makedirs("results", exist_ok=True)
    sim = LidSim()
    sweep_wrist_start(sim)
    sweep_wrist_range(sim)


if __name__ == "__main__":
    main()
