"""Compare original and imitation-corrected Panda grasps on all 30 videos.

    python evaluate_hybrid_il.py --headless
    python evaluate_hybrid_il.py --videos 8 10 18 --headless
    python evaluate_hybrid_il.py --mode hybrid --videos 8

The IL correction is applied at GRASP initiation without pausing physics.
"""
import argparse
import json
import pickle
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
import time
from collections import deque
from contextlib import nullcontext
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from scipy.spatial.transform import Rotation
parser = argparse.ArgumentParser()
parser.add_argument("--videos", type=int, nargs="+", default=None)
parser.add_argument("--model", default="variable_anticipation_mlp_6frames_30.pt")
parser.add_argument("--il-model", default="imitation_grasp_policy_30.pt")
parser.add_argument("--mode", choices=("compare", "baseline", "hybrid"), default="compare")
parser.add_argument("--max-xy-correction", type=float, default=0.04)
parser.add_argument("--max-z-correction", type=float, default=0.03)
parser.add_argument("--il-min-correction", type=float, default=0.0, help="Optional correction deadband in metres (0 disables it)")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--pause", type=float, default=0.5)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--robot-speed", type=float, default=0.35)
parser.add_argument("--margin", type=float, default=0.10)
parser.add_argument("--start-grasp-z", type=float, default=0.275)
parser.add_argument("--pose-tolerance", type=float, default=0.018)
parser.add_argument("--original-start", action="store_true")
args = parser.parse_args()
BASE = Path(__file__).resolve().parent
TRAJECTORIES = BASE / "data" / "trajectories"
SPLIT_FILE = BASE / "data" / "variable_anticipation_split_30.json"
IL_SPLIT_FILE = BASE / "data" / "imitation_learning_split_30.json"
MODELS = BASE / "models"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
# ------------------------------------------------------------
# 1. Automatically locate or download Franka Panda
# ------------------------------------------------------------
def ensure_panda():
    local = BASE / "franka_emika_panda"
    menagerie = BASE / "mujoco_menagerie" / "franka_emika_panda"
    for folder in (local, menagerie):
        if (folder / "scene.xml").is_file():
            print("Using Panda model:", folder)
            return folder
    print("Downloading Franka Panda model automatically...")
    repo = BASE / "mujoco_menagerie"
    git = shutil.which("git")
    if git and not repo.exists():
        try:
            subprocess.run([
                git, "clone", "--depth", "1",
                "--filter=blob:none", "--sparse",
                "https://github.com/google-deepmind/mujoco_menagerie.git",
                str(repo)
            ], check=True)
            subprocess.run([
                git, "-C", str(repo), "sparse-checkout",
                "set", "franka_emika_panda"
            ], check=True)
        except (subprocess.CalledProcessError, OSError):
            print("Git download failed. Trying ZIP download.")
        if (menagerie / "scene.xml").is_file():
            return menagerie
    url = (
        "https://github.com/google-deepmind/"
        "mujoco_menagerie/archive/refs/heads/main.zip"
    )
    try:
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "menagerie.zip"
            urllib.request.urlretrieve(url, archive)
            local.mkdir(exist_ok=True)
            with zipfile.ZipFile(archive) as z:
                for member in z.infolist():
                    if "/franka_emika_panda/" not in member.filename:
                        continue
                    if member.is_dir():
                        continue
                    relative = member.filename.split(
                        "/franka_emika_panda/", 1
                    )[1]
                    dest = local / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member) as src, dest.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    except Exception as exc:
        raise RuntimeError(
            "Could not download Panda model. Check internet connection."
        ) from exc
    if not (local / "scene.xml").is_file():
        raise RuntimeError("Panda scene.xml missing after download")
    return local
PANDA = ensure_panda()
# ------------------------------------------------------------
# 2. Load dataset split and configuration
# ------------------------------------------------------------
split_file = SPLIT_FILE
if not split_file.exists():
    raise FileNotFoundError(
        "Missing variable_anticipation_split_30.json"
    )
splits = json.loads(split_file.read_text())
TRAIN_VIDEOS = set(splits["train"])
VALID_VIDEOS = set(splits["validation"])
TEST_VIDEOS = set(splits["test"])
ALL_VIDEOS = sorted(TRAIN_VIDEOS | VALID_VIDEOS | TEST_VIDEOS)
if len(ALL_VIDEOS) != 30:
    raise ValueError(f"Expected 30-video split containing 21, got {ALL_VIDEOS}")
# Use the IL split for reporting, but keep the anticipation split unchanged.
if not IL_SPLIT_FILE.exists():
    raise FileNotFoundError(f"Missing imitation-learning split: {IL_SPLIT_FILE}")
il_splits = json.loads(IL_SPLIT_FILE.read_text())
IL_GROUPS = {name: set(map(int, il_splits.get(name, [])))
             for name in ("train", "validation", "test", "challenge")}
if set().union(*IL_GROUPS.values()) != set(ALL_VIDEOS):
    raise ValueError("IL split does not cover the same 30 videos")
for name, vids in IL_GROUPS.items():
    if any(vids & other for other_name, other in IL_GROUPS.items() if other_name != name):
        raise ValueError("IL split groups overlap")
print(f"Dataset: {split_file.name} ({len(ALL_VIDEOS)} videos)")
VIDEOS = args.videos if args.videos is not None else ALL_VIDEOS
if any(v not in ALL_VIDEOS for v in VIDEOS):
    parser.error("Requested video is not in dataset split")
RADIUS = 0.033
BALL_Z = 0.233
HORIZONS = np.arange(0.10, 0.601, 0.04)
STEPS = 10
CLOSE_TIME = 0.30
CONTACT_TIME = 0.12
LIFT_HEIGHT = 0.12
GRASP_OFFSET = np.array([0.0, 0.0, 0.098])
# ------------------------------------------------------------
# 3. Load and map recorded ball trajectories
# ------------------------------------------------------------
def load_track(video):
    df = pd.read_csv(TRAJECTORIES / f"{video}.csv")
    df = df.sort_values("time").drop_duplicates("time")
    t = df.time.to_numpy(dtype=float, copy=True)
    p = df[["x", "y"]].to_numpy(dtype=float, copy=True)
    if len(t) < 6:
        raise ValueError(f"Video {video}: insufficient samples")
    t -= t[0]
    relative = p - p[0]
    movement = relative[-1]
    distance = np.linalg.norm(movement)
    if distance < 0.15 or t[-1] <= 0:
        raise ValueError(f"Video {video}: invalid trajectory")
    direction = movement / distance
    side = np.array([-direction[1], direction[0]])
    path_x = 0.55 + relative @ side
    path_y = relative @ direction - distance / 2
    return t, path_x, path_y, side, direction
t, path_x, path_y, track_side, track_direction = load_track(VIDEOS[0])
# ------------------------------------------------------------
# 4. Create table and tennis ball in MuJoCo
# ------------------------------------------------------------
scene = ET.parse(PANDA / "scene.xml")
world = scene.getroot().find("worldbody")
ET.SubElement(
    world, "geom",
    name="intercept_table",
    type="box",
    pos="0.60 0 0.10",
    size="0.48 0.75 0.10",
    rgba="0.65 0.70 0.75 1",
    friction="1 0.02 0.01"
)
ball = ET.SubElement(
    world, "body",
    name="tennis_ball",
    pos=f"{path_x[0]} {path_y[0]} {BALL_Z}"
)
ET.SubElement(ball, "freejoint", name="ball_joint")
ET.SubElement(
    ball, "geom",
    name="ball_geom",
    type="sphere",
    size=str(RADIUS),
    mass="0.058",
    friction="2.5 0.05 0.01",
    condim="4",
    rgba="0.85 0.90 0.15 1"
)
scene_file = PANDA / "nn_only_low_start_scene_6frames.xml"
scene_file.write_text(
    ET.tostring(scene.getroot(), encoding="unicode"),
    encoding="utf-8"
)
model = mujoco.MjModel.from_xml_path(str(scene_file))
data = mujoco.MjData(model)
def obj(kind, name):
    ident = mujoco.mj_name2id(model, kind, name)
    if ident < 0:
        raise ValueError(f"Missing MuJoCo object: {name}")
    return ident
B = mujoco.mjtObj.mjOBJ_BODY
J = mujoco.mjtObj.mjOBJ_JOINT
G = mujoco.mjtObj.mjOBJ_GEOM
A = mujoco.mjtObj.mjOBJ_ACTUATOR
hand = obj(B, "hand")
left = obj(B, "left_finger")
right = obj(B, "right_finger")
ball_body = obj(B, "tennis_ball")
ball_geom = obj(G, "ball_geom")
joints = [obj(J, f"joint{i}") for i in range(1, 8)]
q_ids = np.array([model.jnt_qposadr[j] for j in joints])
v_ids = np.array([model.jnt_dofadr[j] for j in joints])
arm_act = [obj(A, f"actuator{i}") for i in range(1, 8)]
grip_act = obj(A, "actuator8")
ball_joint = obj(J, "ball_joint")
ball_q = model.jnt_qposadr[ball_joint]
ball_v = model.jnt_dofadr[ball_joint]
left_geoms = {
    i for i in range(model.ngeom)
    if model.geom_bodyid[i] == left
}
right_geoms = {
    i for i in range(model.ngeom)
    if model.geom_bodyid[i] == right
}
# ------------------------------------------------------------
# 5. Robot and ball state
# ------------------------------------------------------------
def ball_position():
    return data.qpos[ball_q:ball_q + 3].copy()
def ball_velocity():
    return data.qvel[ball_v:ball_v + 3].copy()
def hand_rotation():
    return data.xmat[hand].reshape(3, 3).copy()
def grasp_position():
    return data.xpos[hand] + hand_rotation() @ GRASP_OFFSET
def ball_in_hand():
    return hand_rotation().T @ (
        ball_position() - data.xpos[hand]
    )
def finger_contacts():
    l = r = False
    for c in data.contact[:data.ncon]:
        pair = {c.geom1, c.geom2}
        if ball_geom in pair:
            l |= bool(pair & left_geoms)
            r |= bool(pair & right_geoms)
    return l, r
# ------------------------------------------------------------
# 6. Replay real recorded movement at original speed
# ------------------------------------------------------------
def recorded_position(s):
    s = np.clip(s, 0, t[-1])
    return np.array([
        np.interp(s, t, path_x),
        np.interp(s, t, path_y),
        BALL_Z
    ])
def recorded_velocity(s):
    a = max(0, s - 0.025)
    b = min(t[-1], s + 0.025)
    if b <= a:
        return np.zeros(3)
    return (recorded_position(b) - recorded_position(a)) / (b - a)
def guide_ball(s):
    force = (
        40 * (recorded_position(s) - ball_position())
        + 1.2 * (recorded_velocity(s) - ball_velocity())
    )
    force[2] += 0.058 * 9.81
    data.xfrc_applied[ball_body, :3] = np.clip(force, -5, 5)
# ------------------------------------------------------------
# 7. Jacobian robot controller
# ------------------------------------------------------------
def move_to(goal, feedforward=None):
    global q_target
    if feedforward is None:
        feedforward = np.zeros(3)
    point = grasp_position()
    pos_error = goal - point
    rot_error = Rotation.from_matrix(
        DESIRED_R @ hand_rotation().T
    ).as_rotvec()
    linear = np.clip(
        3.0 * pos_error + feedforward,
        -0.85, 0.85
    )
    angular = np.clip(3.0 * rot_error, -1.0, 1.0)
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, point, hand)
    jac = np.vstack([
        jp[:, v_ids],
        jr[:, v_ids]
    ])
    dq = jac.T @ np.linalg.solve(
        jac @ jac.T + 0.08**2 * np.eye(6),
        np.r_[linear, angular]
    )
    dq = np.clip(dq, -1.5, 1.5)
    q_target += dq * model.opt.timestep * STEPS
    q_target = np.clip(
        q_target,
        data.qpos[q_ids] - 0.15,
        data.qpos[q_ids] + 0.15
    )
    limits = model.jnt_range[joints]
    q_target = np.clip(
        q_target,
        limits[:, 0] + 0.005,
        limits[:, 1] - 0.005
    )
    data.ctrl[arm_act] = q_target
# ------------------------------------------------------------
# 8. Load trained 6-frame anticipation neural network
# ------------------------------------------------------------
observations = deque(maxlen=256)
def estimate_velocity():
    if len(observations) < 3:
        return np.zeros(3)
    recent = list(observations)[-5:]
    dt = recent[-1][0] - recent[0][0]
    if dt <= 0:
        return np.zeros(3)
    return (recent[-1][1] - recent[0][1]) / dt
class AnticipationMLP(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2)
        )
    def forward(self, x):
        return self.net(x)
checkpoint = MODELS / args.model
if not checkpoint.exists():
    raise FileNotFoundError(
        f"Missing trained anticipation model: {checkpoint}"
    )
# Checkpoint generated locally by the training script; load only trusted files.
ck = torch.load(checkpoint, map_location="cpu", weights_only=False)

history = int(ck["history"])
history_dt = float(ck["dt"])
x_mean = np.asarray(ck["x_mean"], dtype=np.float32)
x_std = np.asarray(ck["x_std"], dtype=np.float32)
y_scale = float(ck["y_scale"])
supported = np.asarray(ck["horizons"], dtype=float)
if history != 6 or abs(history_dt - 1 / 30) > 1e-5:
    raise ValueError("Expected a 6-frame, 30 Hz checkpoint")
if len(x_mean) != 2 * history + 3:
    raise ValueError("MLP input dimensions do not match")
if (
    min(HORIZONS) < min(supported) - 1e-6
    or max(HORIZONS) > max(supported) + 1e-6
):
    raise ValueError("Requested horizon not supported by MLP")
predictor = AnticipationMLP(
    len(x_mean),
    int(ck["hidden"])
)
predictor.load_state_dict(ck["state_dict"])
predictor.eval()
print(
    f"Loaded MLP: {checkpoint.name}\n"
    f"History: {history} frames\n"
    f"First prediction after: "
    f"{(history - 1) * history_dt:.3f} seconds"
)
# ------------------------------------------------------------
# 8b. Load trained imitation policy (frozen during rollout)
# ------------------------------------------------------------
class CorrectionMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3)
        )

    def forward(self, x):
        return self.net(x)

il_checkpoint = MODELS / args.il_model
il_policy = None
if args.mode in ("hybrid", "compare"):
    if not il_checkpoint.exists():
        raise FileNotFoundError(f"Missing imitation-learning model: {il_checkpoint}")
    il_ck = torch.load(il_checkpoint, map_location="cpu", weights_only=False)
    if il_ck.get("policy_stage") != "GRASP_INIT":
        raise ValueError("Imitation model was not trained for grasp initiation")
    if list(il_ck.get("feature_names", [])) != [
        "ball_minus_grasp_x", "ball_minus_grasp_y", "ball_minus_grasp_z",
        "ball_velocity_x", "ball_velocity_y", "ball_velocity_z"
    ]:
        raise ValueError("Imitation feature names do not match simulation inputs")
    il_mean = np.asarray(il_ck["feature_mean"], dtype=np.float32)
    il_std = np.asarray(il_ck["feature_std"], dtype=np.float32)
    if il_mean.shape != (6,) or il_std.shape != (6,) or np.any(il_std <= 0):
        raise ValueError("Imitation normalization shape or std is invalid")
    il_scale = float(il_ck["output_scale"])
    il_policy = CorrectionMLP(int(il_ck["hidden"]))
    il_policy.load_state_dict(il_ck["state_dict"])
    il_policy.eval()
    print(f"Loaded IL model: {il_checkpoint.name}")


def imitation_correction(ball_p, ball_vel):
    """Predict a one-time world-XYZ residual at original GRASP_INIT."""
    # Same six inputs used when training the residual policy.
    state = np.r_[ball_p - grasp_position(), ball_vel].astype(np.float32)
    with torch.no_grad():
        normalized = torch.from_numpy((state - il_mean) / il_std)
        delta = il_policy(normalized).numpy().astype(float) / il_scale
    if not np.isfinite(delta).all():
        raise ValueError("IL model produced non-finite correction")
    raw = delta.copy()
    delta[:2] = np.clip(delta[:2], -args.max_xy_correction, args.max_xy_correction)
    delta[2] = np.clip(delta[2], -args.max_z_correction, args.max_z_correction)
    if np.linalg.norm(delta) < args.il_min_correction:
        delta[:] = 0.0
    return delta, raw

# ------------------------------------------------------------
# 9. Predict future ball positions
# ------------------------------------------------------------
def predict_candidates():
    if len(observations) < 3:
        return None
    now = observations[-1][0]
    sample_times = now - (
        np.arange(history - 1, -1, -1) * history_dt
    )
    times = np.array([o[0] for o in observations])
    if sample_times[0] < times[0]:
        return None
    observed = np.array([
        o[1][:2] for o in observations
    ])
    sim_past = np.column_stack([
        np.interp(sample_times, times, observed[:, k])
        for k in range(2)
    ])
    relative = sim_past - sim_past[-1]
    # Convert simulation coordinates back to camera coordinates.
    csv_relative = (
        relative[:, 0, None] * track_side
        + relative[:, 1, None] * track_direction
    )
    # Same five-observation velocity estimator used in training.
    centered = (np.arange(5) - 2) * history_dt
    csv_vel = np.einsum(
        "ki,k->i",
        csv_relative[-5:],
        centered
    ) / (centered @ centered)
    common = np.r_[
        csv_relative.reshape(-1),
        csv_vel
    ].astype(np.float32)
    xs = np.column_stack([
        np.repeat(common[None, :], len(HORIZONS), axis=0),
        HORIZONS
    ]).astype(np.float32)
    with torch.no_grad():
        pred = predictor(
            torch.from_numpy((xs - x_mean) / x_std)
        ).numpy() * y_scale
    # Transform predicted displacement back to MuJoCo.
    sim_delta = np.column_stack([
        pred @ track_side,
        pred @ track_direction
    ])
    current = observations[-1][1]
    future = np.tile(current, (len(HORIZONS), 1))
    future[:, :2] += sim_delta
    future[:, 2] = BALL_Z
    return list(zip(HORIZONS, future))
# ------------------------------------------------------------
# 10. Select interception point
# ------------------------------------------------------------
def predict_interception():
    choices = predict_candidates()
    if choices is None:
        return None
    robot = grasp_position()
    candidates = []
    for eta, candidate in choices:
        # Preserve the previous workspace restriction.
        if not (
            0.40 <= candidate[0] <= 0.70
            and -0.22 <= candidate[1] <= 0.22
        ):
            continue
        robot_eta = (
            np.linalg.norm(candidate - robot)
            / max(args.robot_speed, 0.05)
            + args.margin
        )
        slack = eta - robot_eta
        candidates.append((
            slack, eta, robot_eta, candidate.copy()
        ))
    if not candidates:
        return None
    feasible = [c for c in candidates if c[0] >= 0]
    if feasible:
        choice = min(feasible, key=lambda c: c[1])
    else:
        choice = max(candidates, key=lambda c: c[0])
    slack, ball_eta, robot_eta, point = choice
    return point, ball_eta, robot_eta, slack
# ------------------------------------------------------------
# 11. Compute low-start initial joint configuration
# ------------------------------------------------------------
def find_low_initial_q(desired_grasp_z):
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    initial_grasp = grasp_position().copy()
    reference_R = hand_rotation()
    desired = initial_grasp.copy()
    desired[2] = desired_grasp_z
    lower_q = data.qpos[q_ids].copy()
    limits = model.jnt_range[joints]
    for _ in range(400):
        data.qpos[q_ids] = lower_q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        point = grasp_position()
        dp = desired - point
        dr = Rotation.from_matrix(
            reference_R @ hand_rotation().T
        ).as_rotvec()
        if np.linalg.norm(dp) < 0.002 and np.linalg.norm(dr) < 0.025:
            break
        jp = np.zeros((3, model.nv))
        jr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jp, jr, point, hand)
        J6 = np.vstack([
            jp[:, v_ids],
            jr[:, v_ids]
        ])
        dx = np.r_[
            np.clip(dp, -0.03, 0.03),
            np.clip(dr, -0.07, 0.07)
        ]
        dq = J6.T @ np.linalg.solve(
            J6 @ J6.T + 0.06**2 * np.eye(6),
            dx
        )
        lower_q = np.clip(
            lower_q + np.clip(dq, -0.08, 0.08),
            limits[:, 0] + 0.01,
            limits[:, 1] - 0.01
        )
    data.qpos[q_ids] = lower_q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    actual = grasp_position()
    err = np.linalg.norm(actual - desired)
    rot_err = np.linalg.norm(
        Rotation.from_matrix(
            reference_R @ hand_rotation().T
        ).as_rotvec()
    )
    print(
        f"LOW-START IK: requested Z={desired_grasp_z:.4f}, "
        f"achieved Z={actual[2]:.4f}, "
        f"error={err * 100:.2f} cm"
    )
    if err > args.pose_tolerance or rot_err > 0.12:
        raise RuntimeError(
            "Low-start IK failed. Try --start-grasp-z 0.30"
        )
    table_id = obj(G, "intercept_table")
    for contact in data.contact[:data.ncon]:
        if table_id not in (contact.geom1, contact.geom2):
            continue
        if contact.dist >= -0.002:
            continue
        other = (
            contact.geom2
            if contact.geom1 == table_id
            else contact.geom1
        )
        if model.geom_bodyid[other] != ball_body:
            raise RuntimeError(
                "Initial robot pose penetrates the table. "
                "Try --start-grasp-z 0.30"
            )
    return lower_q.copy()
LOW_Q = find_low_initial_q(args.start_grasp_z)
# ------------------------------------------------------------
# 12. Reset environment for each video
# ------------------------------------------------------------
def reset_trial(video):
    global t, path_x, path_y
    global track_side, track_direction
    global q_target, DESIRED_R
    t, path_x, path_y, track_side, track_direction = load_track(video)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    if not args.original_start:
        data.qpos[q_ids] = LOW_Q
    data.qpos[ball_q:ball_q + 3] = [
        path_x[0], path_y[0], BALL_Z
    ]
    data.qpos[ball_q + 3:ball_q + 7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    q_target = data.qpos[q_ids].copy()
    DESIRED_R = hand_rotation()
    data.ctrl[arm_act] = q_target
    data.ctrl[grip_act] = 255
class NoViewer:
    def is_running(self):
        return True
    def sync(self):
        pass
# ------------------------------------------------------------
# 13. Full interception, grasp and lift trial
# ------------------------------------------------------------
def run_trial(video, viewer, use_il=False):
    reset_trial(video)
    observations.clear()
    viewer.sync()
    phase = "OBSERVE"
    grasp_start = None
    lift_start = None
    lift_target = None
    released = False
    contact_duration = 0.0
    max_contact_duration = 0.0
    max_ball_lift = 0.0
    hold_duration = 0.0
    max_hold_duration = 0.0
    ever_both = False
    il_delta = np.zeros(3)
    il_raw = np.zeros(3)
    print(
        f"\nVideo {video}: 6-frame MLP, "
        f"initial grasp Z={grasp_position()[2]:.4f} m"
    )
    while (
        data.time < max(12.0, t[-1] + 6.0)
        and viewer.is_running()
    ):
        elapsed = float(data.time)
        data.xfrc_applied[ball_body] = 0
        if not released and elapsed <= t[-1]:
            guide_ball(elapsed)
        ball_p = ball_position()
        observations.append((elapsed, ball_p.copy()))
        vel = estimate_velocity()
        prediction = predict_interception()
        if phase in ("OBSERVE", "APPROACH"):
            if prediction is not None:
                phase = "APPROACH"
                move_to(prediction[0])
            else:
                data.ctrl[arm_act] = q_target
            data.ctrl[grip_act] = 255
            local = ball_in_hand()
            inside = (
                abs(local[0]) < 0.030
                and abs(local[1]) < 0.020
                and abs(local[2] - 0.103) < 0.035
            )
            if inside and phase == "APPROACH":
                released = True
                data.xfrc_applied[ball_body] = 0
                phase = "GRASP"
                grasp_start = elapsed
                # Compute one correction at grasp initiation, without pausing physics.
                if use_il:
                    il_delta, il_raw = imitation_correction(ball_p, ball_velocity())
                    print(f"  IL correction video {video}: {np.round(il_delta * 100, 2)} cm", flush=True)
            elif (
                elapsed > t[-1] + 0.5
                or (phase == "APPROACH" and ball_p[1] > 0.20)
            ):
                phase = "FAILED"
        elif phase == "GRASP":
            move_to(ball_p + il_delta, feedforward=vel)
            progress = np.clip(
                (elapsed - grasp_start) / CLOSE_TIME,
                0, 1
            )
            data.ctrl[grip_act] = 255 * (1 - progress)
            if elapsed - grasp_start > 1.0:
                phase = "FAILED"
        elif phase == "LIFT":
            move_to(lift_target)
            data.ctrl[grip_act] = 0
            if elapsed - lift_start >= 3.0:
                phase = "DONE"
        else:
            data.ctrl[arm_act] = q_target
            data.ctrl[grip_act] = 0
        for _ in range(STEPS):
            mujoco.mj_step(model, data)
            left_contact, right_contact = finger_contacts()
            both = left_contact and right_contact
            ever_both |= both
            if phase == "GRASP" and both:
                contact_duration += model.opt.timestep
                max_contact_duration = max(
                    max_contact_duration,
                    contact_duration
                )
            elif phase == "GRASP":
                contact_duration = 0.0
            lift = ball_position()[2] - BALL_Z
            max_ball_lift = max(max_ball_lift, lift)
            gap = np.linalg.norm(
                ball_position() - grasp_position()
            )
            if lift > 0.06 and both and gap < 0.08:
                hold_duration += model.opt.timestep
                max_hold_duration = max(
                    max_hold_duration,
                    hold_duration
                )
            else:
                hold_duration = 0.0
        if phase == "GRASP" and contact_duration >= CONTACT_TIME:
            phase = "LIFT"
            lift_start = float(data.time)
            lift_target = grasp_position() + np.array([
                0.0, 0.0, LIFT_HEIGHT
            ])
        if phase in ("FAILED", "DONE"):
            break
        viewer.sync()
        if not args.headless:
            time.sleep(
                model.opt.timestep * STEPS
                / max(0.05, args.speed)
            )
    split = next(name for name, ids in IL_GROUPS.items() if video in ids)
    return {
        "video": video,
        "split": split,
        "predictor": "mlp_6frames",
        "mode": "hybrid" if use_il else "baseline",
        "il_dx_cm": round(float(il_delta[0] * 100), 3),
        "il_dy_cm": round(float(il_delta[1] * 100), 3),
        "il_dz_cm": round(float(il_delta[2] * 100), 3),
        "il_raw_magnitude_cm": round(float(np.linalg.norm(il_raw) * 100), 3),
        "phase": phase,
        "success": bool(max_hold_duration >= 0.20),
        "both_fingers_contacted": bool(ever_both),
        "max_lift_m": round(float(max_ball_lift), 4),
        "hold_s": round(float(max_hold_duration), 3),
        "contact_s": round(float(max_contact_duration), 3),
        "sim_time_s": round(float(data.time), 3),
        "status": "OK"
    }
# ------------------------------------------------------------
# 14. Run and save evaluation
# ------------------------------------------------------------

# Report actual rollout success; training-fit errors are not used here.
def save_comparison_plots(frame, mode):
    groups = ["train", "validation", "test", "challenge"]
    methods = [m for m in ("baseline", "hybrid") if m in set(frame["mode"])]
    positions = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.36 if len(methods) == 2 else 0.55
    for idx, method in enumerate(methods):
        values = []
        labels = []
        for group in groups:
            subset = frame[(frame["mode"] == method) & (frame["split"] == group)]
            passed = int(subset["success"].sum())
            count = len(subset)
            values.append(100 * passed / count if count else np.nan)
            labels.append(f"{passed}/{count}" if count else "")
        xs = positions + (idx - (len(methods) - 1) / 2) * width
        bars = ax.bar(xs, values, width=width, label=method.capitalize())
        for bar, label in zip(bars, labels):
            if label:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        label, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(positions, [g.capitalize() for g in groups])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Successful grasps (%)")
    ax.set_title("Autonomous grasp success by IL data split")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    dest = RESULTS / f"hybrid_success_by_split_{mode}.png"
    fig.savefig(dest, dpi=180)
    plt.close(fig)
    print("Saved:", dest)

    # The paired plot identifies recoveries and regressions per video.
    if {"baseline", "hybrid"}.issubset(set(frame["mode"])):
        paired = frame.pivot(index="video", columns="mode", values="success")
        paired = paired.dropna(subset=["baseline", "hybrid"]).sort_index()
        if len(paired):
            codes = []
            labels = []
            for video, row in paired.iterrows():
                base, hybrid = bool(row["baseline"]), bool(row["hybrid"])
                codes.append(3 if not base and hybrid else 2 if base and not hybrid
                             else 1 if base and hybrid else 0)
                labels.append(str(video))
            from matplotlib.colors import ListedColormap, BoundaryNorm
            cmap = ListedColormap(["#b8bdc4", "#78b68e", "#e7a86d", "#5a90bd"])
            fig, ax = plt.subplots(figsize=(12, 3.0))
            ax.imshow(np.array(codes, dtype=int)[None, :], aspect="auto",
                      cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 4.5), 4))
            ax.set_xticks(range(len(labels)), labels)
            ax.set_yticks([])
            ax.set_xlabel("Video")
            ax.set_title("Per-video outcome: gray = both fail, green = both pass, orange = regression, blue = recovery")
            fig.tight_layout()
            dest = RESULTS / "hybrid_per_video_comparison.png"
            fig.savefig(dest, dpi=180)
            plt.close(fig)
            print("Saved:", dest)

        # Compare lift heights for the same recorded movements.
        lift = frame.pivot(index="video", columns="mode", values="max_lift_m")
        lift = lift.dropna(subset=["baseline", "hybrid"]).sort_index()
        if len(lift):
            xs = np.arange(len(lift))
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.bar(xs - 0.2, lift["baseline"].to_numpy() * 100, 0.4, label="Baseline")
            ax.bar(xs + 0.2, lift["hybrid"].to_numpy() * 100, 0.4, label="Hybrid")
            ax.axhline(6, ls="--", linewidth=1, label="6 cm lift threshold")
            ax.set_xticks(xs, lift.index.astype(str))
            ax.set_xlabel("Video")
            ax.set_ylabel("Maximum ball lift (cm)")
            ax.set_title("Maximum lift per trial (hold requirement assessed separately)")
            ax.legend()
            fig.tight_layout()
            dest = RESULTS / "hybrid_max_lift_comparison.png"
            fig.savefig(dest, dpi=180)
            plt.close(fig)
            print("Saved:", dest)

def main():
    results = []
    output = RESULTS / f"evaluation_hybrid_il_30_{args.mode}.csv"
    print("\nVideos to evaluate:", VIDEOS)
    # Reset the simulation before each baseline and hybrid rollout.
    runs = ([False, True] if args.mode == "compare"
            else [args.mode == "hybrid"])
    viewer_context = (
        nullcontext(NoViewer())
        if args.headless
        else mujoco.viewer.launch_passive(model, data)
    )
    with viewer_context as viewer:
        for video in VIDEOS:
            for use_il in runs:
                if not viewer.is_running():
                    break
                result = run_trial(video, viewer, use_il=use_il)
                results.append(result)
                print(result, flush=True)
                pd.DataFrame(results).to_csv(output, index=False)
                if viewer.is_running() and not args.headless:
                    viewer.sync()
                    time.sleep(max(0, args.pause))
            if not viewer.is_running():
                break

    if not results:
        return
    frame = pd.DataFrame(results)
    print("\nFINAL RESULTS")
    print(frame.to_string(index=False))
    print("\nSUCCESS BY MODE")
    for mode_name, group in frame.groupby("mode", sort=False):
        print(f"{mode_name}: {int(group.success.sum())}/{len(group)}")
        for split_name in ("train", "validation", "test", "challenge"):
            sub = group[group.split == split_name]
            if len(sub):
                print(f"  {split_name}: {int(sub.success.sum())}/{len(sub)}")
    if args.mode == "compare":
        wide = frame.pivot(index="video", columns="mode", values="success")
        if {"baseline", "hybrid"}.issubset(wide.columns):
            gained = wide.index[(~wide.baseline) & wide.hybrid].tolist()
            lost = wide.index[wide.baseline & (~wide.hybrid)].tolist()
            print("\nRecovered videos:", gained)
            print("Regressed videos:", lost)
    print("\nSaved:", output)
    save_comparison_plots(frame, args.mode)


if __name__ == "__main__":
    main()
