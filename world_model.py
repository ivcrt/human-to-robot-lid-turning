"""A small learned model of the robot and the lid, trained on the simulated episodes.

Given the last few steps and the next command, it predicts how much the lid
and the wrist turn during the next step (1/20 s). Everything is expressed
relative to the current state, so the model does not depend on absolute angles.

Data: results/histories.npz from experiments.py. Each episode is an array with
one row per step: time, lid angle, wrist angle, wrist target, grip command.
Episodes from my third take of each angle (".03") are kept for testing.

Usage:
    python world_model.py
"""

import numpy as np
import torch
from torch import nn

WINDOW = 5      # number of past steps given to the model


def episode_features(history):
    """Per-step features and next-step targets for one episode.

    features[t] = (lid change, wrist change, target - wrist, grip) at step t
    targets[t]  = (lid change, wrist change) from step t to t+1
    """
    lid, wrist, target, grip = history[:, 1], history[:, 2], history[:, 3], history[:, 4]
    d_lid = np.diff(lid, prepend=lid[0])
    d_wrist = np.diff(wrist, prepend=wrist[0])
    features = np.column_stack([d_lid, d_wrist, target - wrist, grip])
    return features, np.column_stack([d_lid, d_wrist])


def make_samples(history):
    """Model inputs and outputs for every step of an episode.

    input  = the last WINDOW feature rows (flattened) + the next command
             (next target relative to the current wrist, next grip)
    output = lid and wrist change during the next step
    """
    features, changes = episode_features(history)
    target, grip, wrist = history[:, 3], history[:, 4], history[:, 2]
    inputs, outputs = [], []
    for t in range(WINDOW - 1, len(history) - 1):
        past = features[t - WINDOW + 1:t + 1].ravel()
        command = [target[t + 1] - wrist[t], grip[t + 1]]
        inputs.append(np.concatenate([past, command]))
        outputs.append(changes[t + 1])
    return np.array(inputs), np.array(outputs)


def load_dataset(path="results/histories.npz", test_take="03"):
    """Train/test arrays. Episodes from take 03 of every angle are kept for testing.

    Keys of the npz look like "geste_360.03|45|faithful".
    """
    data = np.load(path)
    splits = {"train": ([], []), "test": ([], [])}
    for key in data.files:
        clip = key.split("|")[0]
        split = "test" if clip.endswith(test_take) else "train"
        x, y = make_samples(data[key])
        splits[split][0].append(x)
        splits[split][1].append(y)
    return {name: (np.concatenate(xs), np.concatenate(ys)) for name, (xs, ys) in splits.items()}


# --- model and training ---

class DynamicsMLP(nn.Module):
    """Two hidden layers; works on normalized inputs and outputs."""

    def __init__(self, n_in, n_out=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.net(x)


class WorldModel:
    """The network plus the normalization learned on the training set."""

    def __init__(self, x_train, y_train):
        self.x_mean, self.x_std = x_train.mean(0), x_train.std(0) + 1e-6
        self.y_mean, self.y_std = y_train.mean(0), y_train.std(0) + 1e-6
        self.net = DynamicsMLP(x_train.shape[1], y_train.shape[1])

    def fit(self, x, y, epochs=20, batch=256, lr=1e-3, seed=0):
        torch.manual_seed(seed)
        xs = torch.tensor((x - self.x_mean) / self.x_std, dtype=torch.float32)
        ys = torch.tensor((y - self.y_mean) / self.y_std, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        for epoch in range(epochs):
            order = torch.randperm(len(xs))
            total = 0.0
            for i in range(0, len(xs), batch):
                idx = order[i:i + batch]
                loss = nn.functional.mse_loss(self.net(xs[idx]), ys[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(idx)
            print(f"epoch {epoch + 1:2d} | train loss {total / len(xs):.4f}")

    @classmethod
    def load(cls, path="results/world_model.pt"):
        saved = torch.load(path, weights_only=False)
        model = cls.__new__(cls)
        for key in ["x_mean", "x_std", "y_mean", "y_std"]:
            setattr(model, key, saved[key])
        model.net = DynamicsMLP(len(model.x_mean), len(model.y_mean))
        model.net.load_state_dict(saved["net"])
        return model

    def predict(self, x):
        """Lid and wrist change (deg) for each row of x."""
        with torch.no_grad():
            xs = torch.tensor((x - self.x_mean) / self.x_std, dtype=torch.float32)
            return self.net(xs).numpy() * self.y_std + self.y_mean


# --- evaluation ---

def one_step_errors(model, x, y):
    """Mean absolute error (deg per step) of the model and of a naive baseline.

    Baseline: the next change equals the current one (last row of the window).
    """
    n_feat = (x.shape[1] - 2) // WINDOW
    last = x[:, (WINDOW - 1) * n_feat:(WINDOW - 1) * n_feat + 2]     # current lid and wrist change
    return np.abs(model.predict(x) - y).mean(0), np.abs(last - y).mean(0)


def rollout(model, history, start, horizon):
    """Predicts the lid angle for `horizon` steps after `start`, feeding the model its own outputs.

    Only the commands (wrist target, grip) are taken from the real episode.
    Returns predicted and real lid angles, both relative to the lid at `start`.
    """
    features, _ = episode_features(history)
    window = list(features[start - WINDOW + 1:start + 1])
    target, grip = history[:, 3], history[:, 4]
    wrist = history[start, 2]
    lid = 0.0
    predicted = []
    for t in range(start, start + horizon):
        x = np.concatenate([np.ravel(window), [target[t + 1] - wrist, grip[t + 1]]])
        d_lid, d_wrist = model.predict(x[None])[0]
        lid += d_lid
        wrist += d_wrist
        predicted.append(lid)
        window = window[1:] + [np.array([d_lid, d_wrist, target[t + 1] - wrist, grip[t + 1]])]
    real = history[start + 1:start + horizon + 1, 1] - history[start, 1]
    return np.array(predicted), real


def rollout_errors(model, histories, horizons=(10, 20, 40, 100), starts_per_episode=5, seed=0):
    """Mean absolute lid-angle error (deg) after each horizon (in steps of 1/20 s)."""
    rng = np.random.default_rng(seed)
    errors = {h: [] for h in horizons}
    for history in histories:
        last_start = len(history) - max(horizons) - 2
        if last_start <= WINDOW:
            continue
        for start in rng.integers(WINDOW, last_start, starts_per_episode):
            predicted, real = rollout(model, history, start, max(horizons))
            for h in horizons:
                errors[h].append(abs(predicted[h - 1] - real[h - 1]))
    return {h: float(np.mean(e)) for h, e in errors.items()}


def main():
    ds = load_dataset()
    x_train, y_train = ds["train"]
    x_test, y_test = ds["test"]
    print("train", x_train.shape, "test", x_test.shape)

    model = WorldModel(x_train, y_train)
    model.fit(x_train, y_train)

    mae_model, mae_naive = one_step_errors(model, x_test, y_test)
    print(f"one step (deg), lid / wrist | model {mae_model.round(3)} | naive {mae_naive.round(3)}")

    data = np.load("results/histories.npz")
    test_histories = [data[k] for k in data.files if k.split("|")[0].endswith("03")]
    for h, e in rollout_errors(model, test_histories).items():
        print(f"open-loop rollout, {h / 20:.1f} s ahead: lid error {e:.1f} deg")
    torch.save({"net": model.net.state_dict(), "x_mean": model.x_mean, "x_std": model.x_std,
                "y_mean": model.y_mean, "y_std": model.y_std}, "results/world_model.pt")


if __name__ == "__main__":
    main()
