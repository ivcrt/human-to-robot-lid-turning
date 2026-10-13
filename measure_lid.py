"""Measure the rotation of the lid in each clip and split it into pushes.

The lid has a dark line drawn across its diameter. On every frame we find the
pixels of that line inside the lid and fit a straight line to them. Its angle,
followed over time, gives the rotation of the lid. A push is a stretch of
frames where the lid turns faster than a speed threshold.

Angles are in image coordinates (y points down), so unscrewing gives
negative angles.

Usage:
    python measure_lid.py path/to/clips data/pushes.csv

Clips are named geste_<target angle>.<take>.MOV, e.g. geste_360.03.MOV.
The videos are not in the repository; data/pushes.csv is the output on mine.
"""

import argparse
import csv
import glob
import os

import cv2
import numpy as np


def read_frame(video_path, index):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = video.read()
    video.release()
    return frame


def find_lid(video_path):
    """Centre and radius of the lid, found on the first frame (no hand on it yet)."""
    gray = cv2.medianBlur(cv2.cvtColor(read_frame(video_path, 0), cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.5, minDist=500,
                               param1=100, param2=40, minRadius=70, maxRadius=107)
    return circles[0][0]


def line_pixels(frame, lid, max_value=140, max_saturation=90):
    """Pixels that are dark and grey (the drawn line) inside the lid."""
    cx, cy, r = lid
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    grey = (hsv[:, :, 2] < max_value) & (hsv[:, :, 1] < max_saturation)
    ys, xs = np.nonzero(grey)
    # 0.85 * r keeps us away from the rim of the lid
    inside = (xs - cx) ** 2 + (ys - cy) ** 2 < (0.85 * r) ** 2
    return xs[inside], ys[inside]


def line_angle(frame, lid):
    """Angle of the drawn line in degrees, or None if it is hidden by the fingers."""
    xs, ys = line_pixels(frame, lid)
    if len(xs) < 50:
        return None
    points = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.degrees(np.arctan2(vy, vx))


def raw_angles(video_path):
    """Line angle on every frame of the clip (nan when the line is hidden)."""
    lid = find_lid(video_path)
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    angles = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        a = line_angle(frame, lid)
        angles.append(np.nan if a is None else a)
    video.release()
    angles = np.array(angles)
    return np.arange(len(angles)) / fps, angles


def lid_rotation(video_path):
    """Time and lid rotation since the first frame, in degrees."""
    times, raw = raw_angles(video_path)
    seen = ~np.isnan(raw)
    t = times[seen]
    # a line looks the same after 180 degrees, hence period=180
    rotation = np.unwrap(raw[seen], period=180)
    return t, rotation - rotation[0]


def split_pushes(t, rotation, speed_threshold=-15, min_size=-5):
    """Size (deg) and duration (s) of each push.

    speed_threshold: 3x the noise of a still lid, below the slowest real push.
    min_size: drops tiny detections caused by noise.
    """
    speed = np.gradient(rotation, t)
    pushing = speed < speed_threshold
    # pad with "not pushing" so that every start has an end
    changes = np.diff(pushing.astype(int), prepend=0, append=0)
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1
    sizes = rotation[ends] - rotation[starts]
    durations = t[ends] - t[starts]
    keep = sizes < min_size
    return sizes[keep], durations[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips_dir", help="folder with the geste_XXX.NN.MOV clips")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["clip", "target_deg", "total_deg", "push", "size_deg", "duration_s"])
        for path in sorted(glob.glob(os.path.join(args.clips_dir, "geste_*.MOV"))):
            clip = os.path.splitext(os.path.basename(path))[0]
            target = int(clip[6:9])
            t, rotation = lid_rotation(path)
            sizes, durations = split_pushes(t, rotation)
            for i, (size, duration) in enumerate(zip(sizes, durations)):
                writer.writerow([clip, target, round(rotation[-1], 1), i + 1,
                                 round(size, 1), round(duration, 3)])
            print(f"{clip} | target {target} | measured {rotation[-1]:.0f} | pushes {len(sizes)}")


if __name__ == "__main__":
    main()
