"""Three ways for the robot to turn the lid by the angle I turned it.

1. faithful: replays my pushes (sizes and speeds), regrasping after each one.
2. robot_style: only keeps my total angle; turns as far as the wrist allows.
3. anticipatory: before each grasp, turns the open gripper back so that the
   whole stroke fits before the wrist limit.
"""

import csv

import numpy as np

RETURN_SPEED = 180      # deg/s, open gripper going back


def load_pushes(clip, path="data/pushes.csv"):
    """Sizes (deg, positive = counterclockwise) and durations (s) of a clip's pushes."""
    rows = [r for r in csv.DictReader(open(path)) if r["clip"] == clip]
    sizes = np.array([-float(r["size_deg"]) for r in rows])   # image angles are negative when unscrewing
    durations = np.array([float(r["duration_s"]) for r in rows])
    return sizes, durations


def load_total(clip, path="data/pushes.csv"):
    """Final angle of the lid in my clip (deg, positive = counterclockwise)."""
    row = next(r for r in csv.DictReader(open(path)) if r["clip"] == clip)
    return -float(row["total_deg"])


def faithful(sim, sizes, durations):
    """Replays my pushes: same sizes, same speeds, a regrasp after each one."""
    sim.grasp()
    for i, (size, duration) in enumerate(zip(sizes, durations)):
        sim.turn(size, size / duration)
        sim.release()
        if i < len(sizes) - 1:              # not the last push: go back and regrasp
            sim.turn(-size, RETURN_SPEED)
            sim.grasp()


def robot_style(sim, total, speed=90, tol=5):
    """Only my total angle: turns as far as the wrist allows, regrasps only when it must."""
    sim.grasp()
    turned = 0
    while turned < total - tol:
        size = min(total - turned, sim.room_ccw())
        turned += sim.turn(size, speed)
        if turned < total - tol:            # still some angle to do: regrasp
            sim.release()
            sim.turn(-sim.room_cw(), RETURN_SPEED)
            sim.grasp()
    sim.release()


def anticipatory(sim, total, speed=90, tol=5):
    """Before each grasp, turns the open gripper back so that the next stroke fits before the limit."""
    turned = 0
    while turned < total - tol:
        stroke = min(total - turned, sim.full_range())
        if sim.room_ccw() < stroke:                            # not enough room: prepare first
            sim.turn(-(stroke - sim.room_ccw()), RETURN_SPEED)  # gripper still open
        sim.grasp()
        turned += sim.turn(stroke, speed)
        sim.release()
