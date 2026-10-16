
# Plot and save the ball speed profiles from all recorded trajectories.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
TRAJECTORIES = BASE / "data" / "trajectories"
RESULTS = BASE / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

files = sorted(
    (p for p in TRAJECTORIES.glob("*.csv") if p.stem.isdigit()),
    key=lambda p: int(p.stem)
)

if not files:
    raise FileNotFoundError(f"No trajectory CSV files found in {TRAJECTORIES}")

plt.figure(figsize=(13, 7))

for file in files:
    df = pd.read_csv(file)

    t = df["time"] - df["time"].iloc[0]
    speed = np.hypot(df["vx"], df["vy"])

    plt.plot(t, speed, label=file.stem, alpha=0.7)

plt.xlabel("Time (s)")
plt.ylabel("Ball speed (m/s)")
plt.title(f"Tennis Ball Velocity Profiles - {len(files)} Videos")
plt.legend(
    ncol=4,
    fontsize=7,
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(RESULTS / "ball_velocity_profiles.png", dpi=300)
plt.show()

print(f"Plotted {len(files)} trajectories.")
print("Saved: results/ball_velocity_profiles.png")
