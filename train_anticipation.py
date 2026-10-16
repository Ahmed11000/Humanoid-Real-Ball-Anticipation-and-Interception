"""Train the 6-frame ball-motion predictor on the final 30 trajectories."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "trajectories"
MODELS = BASE / "models"
RESULTS = BASE / "results"
SPLITS = BASE / "data"
for folder in (MODELS, RESULTS, SPLITS):
    folder.mkdir(parents=True, exist_ok=True)

#number of consecutive past and current frames used to predict the ball's future position
HISTORY = 6
DT = 1 / 30

#Hoizons reprents the time of predcited future ball position
HORIZONS = [    
    0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.60
]

EPOCHS = 160
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# Original video-level split, mapped to the new 1–30 IDs.
split = {
    "train": [4, 5, 6, 8, 11, 13, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 30],
    "validation": [1, 7, 10, 14, 22],
    "test": [2, 3, 9, 12, 15, 29],
}
TRAIN = split["train"]
VALID = split["validation"]
TEST = split["test"]

ids = TRAIN + VALID + TEST
if len(ids) != 30 or set(ids) != set(range(1, 31)):
    raise ValueError("Split must contain each video ID 1..30 exactly once")

missing = [v for v in ids if not (DATA / f"{v}.csv").is_file()]
if missing:
    raise FileNotFoundError(f"Missing trajectory CSVs in {DATA}: {missing}")

split_file = SPLITS / "variable_anticipation_split_30.json"
split_file.write_text(json.dumps(split, indent=2), encoding="utf-8")

print("6-frame anticipation MLP | 30 videos")
print(f"Train {len(TRAIN)}: {TRAIN}")
print(f"Validation {len(VALID)}: {VALID}")
print(f"Test {len(TEST)}: {TEST}")
print(f"History: {HISTORY} frames; observation window: {(HISTORY-1)*DT:.3f} s")
print("Saved split:", split_file)


# Load the tracked XY positions and reset time to the first detection.
def read_track(video):

    file = DATA / f"{video}.csv"

    if not file.exists():
        raise FileNotFoundError(
            f"Missing trajectory: {file}"
        )

    df = pd.read_csv(file)

    df = (
        df.sort_values("time")
        .drop_duplicates("time")
    )

    t = df.time.to_numpy(dtype=float)

    xy = df[
        ["x", "y"]
    ].to_numpy(dtype=float)

    valid = (
        np.isfinite(t)
        & np.isfinite(xy).all(axis=1)
    )

    t, xy = t[valid], xy[valid]

    if len(t) < HISTORY:
        raise ValueError(
            f"Video {video}: insufficient samples"
        )

    return t - t[0], xy


# Generate training samples

# Build six-frame inputs and future-displacement targets.
def samples(video, horizon):

    t, xy = read_track(video)

    start = (HISTORY - 1) * DT
    stop = t[-1] - horizon

    if stop < start:
        return None

    now = np.arange(
        start,
        stop + 1e-8,
        DT
    )

    history_times = (
        now[:, None]
        - (
            np.arange(
                HISTORY - 1,
                -1,
                -1
            ) * DT
        )
    )

    past = np.stack([
        np.interp(
            history_times,
            t,
            xy[:, k]
        )
        for k in range(2)
    ], axis=-1)

    present = past[:, -1]

    future = np.column_stack([
        np.interp(
            now + horizon,
            t,
            xy[:, k]
        )
        for k in range(2)
    ])

    # Estimate velocity from the latest five observations.
    recent = past[:, -5:]

    centered = (
        np.arange(5) - 2
    ) * DT

    velocity = np.einsum(
        "nki,k->ni",
        recent,
        centered
    ) / (centered @ centered)

    # Constant-velocity baseline for comparison.
    cv = present + velocity * horizon

    features = np.concatenate([
        (
            past - present[:, None]
        ).reshape(len(now), -1),

        velocity,

        np.full(
            (len(now), 1),
            horizon
        )
    ], axis=1).astype(np.float32)

    return {
        "x": features,
        "target": (
            future - present
        ).astype(np.float32),
        "present": present,
        "actual": future,
        "cv": cv,
        "time": now
    }


# Combine dataset

# Pool samples across videos and prediction horizons.
def combine(videos):

    sets = [
        samples(v, h)
        for v in videos
        for h in HORIZONS
    ]

    sets = [
        s for s in sets
        if s is not None
    ]

    if not sets:
        raise RuntimeError(
            "No valid training windows"
        )

    X = np.concatenate([
        s["x"] for s in sets
    ])

    Y = np.concatenate([
        s["target"] for s in sets
    ])

    return X, Y


# Anticipation mlp

class AnticipationMLP(nn.Module):

    def __init__(self, n):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)


# Prepare training and validation data

print("\nPreparing training data...")

X, Y = combine(TRAIN)
VX, VY = combine(VALID)

# Normalise using training data only.
mean = X.mean(axis=0)

std = np.maximum(
    X.std(axis=0),
    1e-3
)

scale = max(
    float(Y.std()),
    1e-3
)

TX = torch.from_numpy(
    (X - mean) / std
).float()

TY = torch.from_numpy(
    Y / scale
).float()

TVX = torch.from_numpy(
    (VX - mean) / std
).float()

TVY = torch.from_numpy(
    VY / scale
).float()

print(f"Training windows: {len(TX)}")
print(f"Validation windows: {len(TVX)}")


# Initialise model

model = AnticipationMLP(
    X.shape[1]
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4
)

best = float("inf")
best_state = None
best_epoch = 0
patience = 0


# Training

print("\nTraining 6-frame MLP...")

# Keep the checkpoint with the lowest validation error.
for epoch in range(1, EPOCHS + 1):

    model.train()

    for batch in torch.randperm(
        len(TX)
    ).split(256):

        prediction = model(
            TX[batch]
        )

        loss = (
            prediction - TY[batch]
        ).square().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():

        val = (
            model(TVX) - TVY
        ).square().mean().item()

    if val < best - 1e-5:

        best = val

        best_state = deepcopy(
            model.state_dict()
        )

        best_epoch = epoch
        patience = 0

    else:
        patience += 1

    if epoch == 1 or epoch % 10 == 0:

        print(
            f"Epoch {epoch:3d} | "
            f"Validation MSE: {val:.5f}"
        )

    if patience >= 25:

        print("Early stopping")
        break


model.load_state_dict(best_state)
model.eval()


# Save trained model

checkpoint = (
    MODELS / "variable_anticipation_mlp_6frames_30.pt"
)

torch.save({
    "state_dict": model.state_dict(),
    "x_mean": mean,
    "x_std": std,
    "y_scale": scale,
    "history": HISTORY,
    "dt": DT,
    "horizons": HORIZONS,
    "hidden": 64,
    "best_epoch": best_epoch,
    "train_videos": TRAIN,
    "validation_videos": VALID,
    "test_videos": TEST,
    "seed": SEED
}, checkpoint)

print(f"\nBest epoch: {best_epoch}")
print(f"Model saved: {checkpoint.name}")


# Prediction function

def predict(s):

    with torch.no_grad():

        x = torch.from_numpy(
            (s["x"] - mean) / std
        ).float()

        return (
            s["present"]
            + model(x).numpy() * scale
        )


# Held-out evaluation

# Compare model forecasts with constant-velocity estimates on held-out videos.
rows = []

for video in TEST:

    for horizon in HORIZONS:

        s = samples(
            video,
            horizon
        )

        if s is None:
            continue

        for method, prediction in [
            ("CV", s["cv"]),
            ("MLP", predict(s))
        ]:

            error = np.linalg.norm(
                prediction - s["actual"],
                axis=1
            ) * 100

            rows.append({
                "video": video,
                "horizon_s": horizon,
                "method": method,
                "error_cm": float(
                    error.mean()
                ),
                "windows": len(error)
            })


results = pd.DataFrame(rows)

output_csv = (
    RESULTS / "variable_anticipation_errors_6frames_30.csv"
)

results.to_csv(
    output_csv,
    index=False
)


# Print test results

print("\nHELD-OUT PREDICTION ERROR (cm)")
print("Horizon       CV       MLP")

for horizon in HORIZONS:

    errors = []

    for method in ("CV", "MLP"):

        q = results[
            np.isclose(
                results.horizon_s,
                horizon
            )
            & (results.method == method)
        ]

        errors.append(
            np.average(
                q.error_cm,
                weights=q.windows
            )
            if len(q)
            else float("nan")
        )

    print(
        f"{horizon:5.2f} s    "
        f"{errors[0]:6.2f}    "
        f"{errors[1]:6.2f}"
    )


# Finished

print("\nTraining complete.")

print("Saved:", checkpoint.name)

print(
    "Saved:",
    split_file.name
)

print(
    "Saved:",
    output_csv.name
)

print("All IDs refer to the new video numbering (1–30).")
