
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
VIDEOS = BASE / "data"
TRAJECTORIES = BASE / "data" / "trajectories"

TRAJECTORIES.mkdir(parents=True, exist_ok=True)

videos = sorted(
    (f for f in VIDEOS.glob("*.mp4") if f.stem.isdigit()),
    key=lambda f: int(f.stem)
)

print(f"Found {len(videos)} videos.")

for file in videos:
    cap = cv2.VideoCapture(str(file))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()

    if not ok or fps <= 0:
        print(f"Video {file.stem}: Cannot open")
        cap.release()
        continue

    # Calibrate using the 80 cm reference
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f"Video {file.stem}: Click both ends of the 80 cm reference")
    points = np.array(plt.ginput(2, timeout=0))
    plt.close()

    if len(points) != 2 or np.linalg.norm(points[1] - points[0]) == 0:
        print(f"Video {file.stem}: Calibration skipped")
        cap.release()
        continue

    scale = 0.8 / np.linalg.norm(points[1] - points[0])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    rows = []
    frame_id = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Track the ball using HSV colour filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (25, 50, 50), (85, 255, 255))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            c = max(contours, key=cv2.contourArea)

            if cv2.contourArea(c) > 30:
                (x, y), radius = cv2.minEnclosingCircle(c)
                rows.append([frame_id / fps, x * scale, y * scale])

        frame_id += 1

    cap.release()

    if len(rows) < 4:
        print(f"Video {file.stem}: Tracking failed")
        continue

    raw = np.array(rows)

    # Resample the recorded positions at 30 Hz
    times = np.arange(
        0, raw[-1, 0] - raw[0, 0] + 1e-9, 1 / 30
    )

    if len(times) < 3:
        print(f"Video {file.stem}: Too short")
        continue

    df = pd.DataFrame({
        "time": times,
        "x": np.interp(times + raw[0, 0], raw[:, 0], raw[:, 1]),
        "y": np.interp(times + raw[0, 0], raw[:, 0], raw[:, 2])
    })

    # Velocity estimates for analysis
    df["vx"] = np.gradient(df.x, df.time)
    df["vy"] = np.gradient(df.y, df.time)

    output = TRAJECTORIES / f"{file.stem}.csv"
    df.to_csv(output, index=False)

    # Check the extracted path
    plt.plot(df.x, -df.y)
    plt.axis("equal")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title(f"Video {file.stem} trajectory")
    plt.grid()
    plt.show()

    print(f"Video {file.stem}: {len(df)} samples saved at 30 Hz")

print(f"\nFinished processing {len(videos)} videos!")
