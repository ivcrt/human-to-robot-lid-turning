"""Runs the three strategies on every clip, for several starting wrist angles.

Writes one row per episode to results/episodes.csv, and every episode's
step-by-step history to results/histories.npz (used to train the world model).

Usage:
    python experiments.py
"""

import csv
import os

import numpy as np

from sim import LidSim
from strategies import anticipatory, faithful, load_pushes, load_total, robot_style

WRIST_STARTS = [-120, -60, 0, 60, 120, 150]   # degrees; 45 is the robosuite default
CLIPS = sorted({r["clip"] for r in csv.DictReader(open("data/pushes.csv"))})


def run_episode(sim, strategy, clip, wrist_start):
    sizes, durations = load_pushes(clip)
    target = load_total(clip)
    sim.reset(wrist_start)
    if strategy == "faithful":
        faithful(sim, sizes, durations)
    elif strategy == "robot_style":
        robot_style(sim, target)
    else:
        anticipatory(sim, target)
    return {
        "clip": clip, "strategy": strategy, "wrist_start": wrist_start,
        "target_deg": round(target, 1),
        "lid_deg": round(sim.lid_angle, 1),
        "error_deg": round(sim.lid_angle - target, 1),
        "time_s": sim.time(),
        "grasps": sim.grasps,
        "limit_hits": sim.wrist_limit_hits,
        "room_left_deg": round(sim.room_ccw(), 1),     # rotation still possible at the end
        "drift_mm": round(1000 * sim.lid_drift(), 1),
    }


def main():
    os.makedirs("results", exist_ok=True)
    sim = LidSim()
    rows, histories = [], {}
    for clip in CLIPS:
        for wrist_start in WRIST_STARTS:
            for strategy in ["faithful", "robot_style", "anticipatory"]:
                row = run_episode(sim, strategy, clip, wrist_start)
                rows.append(row)
                histories[f"{clip}|{wrist_start}|{strategy}"] = np.array(sim.history, dtype=float)
                print(" | ".join(f"{k} {v}" for k, v in row.items()))
    with open("results/episodes.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed("results/histories.npz", **histories)


if __name__ == "__main__":
    main()
