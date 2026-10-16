"""Collect Panda grasp corrections across the 30 recorded trajectories.

Run with: python teleop_il_train.py
Only IL-training trials are paused and saved as demonstrations.
Validation, test, and challenge videos run without manual edits.
"""
import argparse
import json
import uuid
import tkinter as tk
from tkinter import ttk
import time
from collections import deque
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import torch
from torch import nn
from scipy.spatial.transform import Rotation

parser=argparse.ArgumentParser()
parser.add_argument('--pause', type=float, default=1.0)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--demo-dir', default='data/il_demonstrations')
parser.add_argument('--pause-at', choices=['grasp', 'bilateral', 'approach'], default='grasp',
                    help='Pause at GRASP initiation (recommended) or bilateral contact')
parser.add_argument('--approach-pause-time', type=float, default=0.55, help='Simulation time (seconds) to pause during APPROACH; useful for video 21')
parser.add_argument('--diagnostic-interval', type=float, default=0.20)
parser.add_argument('--xy-range', type=float, default=0.04)
parser.add_argument('--speed', type=float, default=1., help='Playback rate only; no changes to physics')
parser.add_argument('--videos', type=int, nargs='+', default=None)
parser.add_argument('--model', default='models/variable_anticipation_mlp_6frames_30.pt')
parser.add_argument('--robot-speed', type=float, default=.35)
parser.add_argument('--margin', type=float, default=.10)
parser.add_argument('--start-grasp-z', type=float, default=.275)
parser.add_argument('--pose-tolerance', type=float, default=.018)
args=parser.parse_args()
BASE=Path(__file__).resolve().parent
PANDA = next((p for p in (BASE/'mujoco_menagerie'/'franka_emika_panda', BASE/'franka_emika_panda') if (p/'scene.xml').exists()), None)
if PANDA is None:
    raise FileNotFoundError('Franka Panda scene.xml missing from project directories')
# Keep the anticipation split and pretrained checkpoint unchanged.
source_split_file = BASE / 'data' / 'variable_anticipation_split_30.json'
if not source_split_file.is_file():
    raise FileNotFoundError(f'Missing anticipation split: {source_split_file}')
source_split = json.loads(source_split_file.read_text(encoding='utf-8'))
base_train = set(map(int, source_split['train']))
base_valid = set(map(int, source_split['validation']))
base_test = set(map(int, source_split['test']))
all_ids = set(range(1, 31))
if (base_train | base_valid | base_test) != all_ids or any((base_train & base_valid, base_train & base_test, base_valid & base_test)):
    raise ValueError('Expected disjoint train/validation/test groups covering videos 1–30')

# IL split: move 10 and 14 to training; reserve 11 as a separate challenge.
if not {8, 11}.issubset(base_train) or not {10, 14}.issubset(base_valid) or not {3, 9}.issubset(base_test):
    raise ValueError('Anticipation split does not match the final 30-video experiment')
TRAIN_VIDEOS = tuple(sorted((base_train - {11}) | {10, 14}))
VALID_VIDEOS = tuple(sorted(base_valid - {10, 14}))
TEST_VIDEOS = tuple(sorted(base_test))
CHALLENGE_VIDEOS = (11,)
ALL_VIDEOS = tuple(range(1, 31))
assert tuple(map(len, (TRAIN_VIDEOS, VALID_VIDEOS, TEST_VIDEOS, CHALLENGE_VIDEOS))) == (20, 3, 6, 1)

il_split_file = BASE / 'data' / 'imitation_learning_split_30.json'
il_split = {'train': list(TRAIN_VIDEOS), 'validation': list(VALID_VIDEOS),
            'test': list(TEST_VIDEOS), 'challenge': list(CHALLENGE_VIDEOS)}
if il_split_file.exists():
    existing = json.loads(il_split_file.read_text(encoding='utf-8'))
    if existing != il_split:
        raise ValueError('Existing IL split differs from the intended final split')
else:
    il_split_file.write_text(json.dumps(il_split, indent=2), encoding='utf-8')

# Default: go through all 30 videos in numerical order.
VIDEOS = args.videos if args.videos is not None else list(ALL_VIDEOS)
if any(v not in all_ids for v in VIDEOS):
    parser.error('Video numbers must be between 1 and 30')
if args.repeats < 1:
    parser.error('--repeats must be >= 1')
RADIUS=.033
# Prediction horizons used by the interception planner.
HORIZONS=np.arange(0.10,0.601,0.04)
BALL_Z=.233
STEPS=10
CLOSE_TIME=.30
CONTACT_TIME=.12
LIFT_HEIGHT=.12
GRASP_OFFSET=np.array([0.,0.,.098])
def load_track(video):
    df = pd.read_csv(BASE / 'data' / 'trajectories' / f'{video}.csv')
    df = df.sort_values('time').drop_duplicates('time')
    t = df.time.to_numpy(dtype=float, copy=True)
    p = df[['x', 'y']].to_numpy(dtype=float, copy=True)
    if len(t) < 6:
        raise ValueError(f'Video {video}: not enough samples')
    t -= t[0]
    relative = p - p[0]
    movement = relative[-1]
    distance = np.linalg.norm(movement)
    if distance < 0.15 or t[-1] <= 0:
        raise ValueError(f'Video {video}: invalid trajectory')
    direction = movement / distance
    side = np.array([-direction[1], direction[0]])
    path_x = 0.55 + relative @ side
    path_y = relative @ direction - distance / 2
    return t, path_x, path_y, side, direction

t, path_x, path_y, track_side, track_direction = load_track(VIDEOS[0])

scene = ET.parse(PANDA / 'scene.xml')
world = scene.getroot().find('worldbody')
ET.SubElement(world, 'geom', name='intercept_table', type='box',
              pos='0.60 0 0.10', size='0.48 0.75 0.10',
              rgba='0.65 0.70 0.75 1', friction='1 0.02 0.01')
ball = ET.SubElement(world, 'body', name='tennis_ball',
                     pos=f'{path_x[0]} {path_y[0]} {BALL_Z}')
ET.SubElement(ball, 'freejoint', name='ball_joint')
ET.SubElement(ball, 'geom', name='ball_geom', type='sphere',
              size=str(RADIUS), mass='0.058', friction='2.5 0.05 0.01',
              condim='4', rgba='0.85 0.90 0.15 1')
# Save the scene beside the Panda assets so mesh paths resolve.
xml = ET.tostring(scene.getroot(), encoding='unicode')
scene_file = PANDA / 'nn_contact_low_start_scene.xml'
scene_file.write_text(xml, encoding='utf-8')
model = mujoco.MjModel.from_xml_path(str(scene_file))
data = mujoco.MjData(model)

def obj(kind, name):
    ident = mujoco.mj_name2id(model, kind, name)
    if ident < 0:
        raise ValueError(f'Missing model object: {name}')
    return ident

B = mujoco.mjtObj.mjOBJ_BODY
J = mujoco.mjtObj.mjOBJ_JOINT
G = mujoco.mjtObj.mjOBJ_GEOM
A = mujoco.mjtObj.mjOBJ_ACTUATOR
hand = obj(B, 'hand')
left = obj(B, 'left_finger')
right = obj(B, 'right_finger')
ball_body = obj(B, 'tennis_ball')
ball_geom = obj(G, 'ball_geom')
joints = [obj(J, f'joint{i}') for i in range(1, 8)]
q_ids = np.array([model.jnt_qposadr[j] for j in joints])
v_ids = np.array([model.jnt_dofadr[j] for j in joints])
arm_act = [obj(A, f'actuator{i}') for i in range(1, 8)]
grip_act = obj(A, 'actuator8')
ball_joint = obj(J, 'ball_joint')
ball_q = model.jnt_qposadr[ball_joint]
ball_v = model.jnt_dofadr[ball_joint]
left_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == left}
right_geoms = {i for i in range(model.ngeom) if model.geom_bodyid[i] == right}

def ball_position():
    return data.qpos[ball_q:ball_q + 3].copy()

def ball_velocity():
    return data.qvel[ball_v:ball_v + 3].copy()

def hand_rotation():
    return data.xmat[hand].reshape(3, 3).copy()

def grasp_position():
    return data.xpos[hand] + hand_rotation() @ GRASP_OFFSET

def ball_in_hand():
    return hand_rotation().T @ (ball_position() - data.xpos[hand])

def finger_contacts():
    l = r = False
    for c in data.contact[:data.ncon]:
        pair = {c.geom1, c.geom2}
        if ball_geom in pair:
            l |= bool(pair & left_geoms)
            r |= bool(pair & right_geoms)
    return l, r

# Trajectory and future points are only used by environment replay.

def finger_contact_forces():
    """Simulated normal contact forces (N) on left and right fingers."""
    lf = rf = 0.0
    wrench = np.zeros(6)
    for i, contact in enumerate(data.contact[:data.ncon]):
        pair = {contact.geom1, contact.geom2}
        if ball_geom not in pair:
            continue
        mujoco.mj_contactForce(model, data, i, wrench)
        normal = abs(float(wrench[0]))
        if pair & left_geoms:
            lf += normal
        if pair & right_geoms:
            rf += normal
    return lf, rf

def compact_vector(v):
    return np.array2string(np.asarray(v), precision=4, suppress_small=True)

def log_grasp_state(label, video):
    left_touch, right_touch = finger_contacts()
    lf, rf = finger_contact_forces()
    print(f'\n[{label}] video={video} time={data.time:.4f}s', flush=True)
    print(f'  ball_xyz={compact_vector(ball_position())} ball_vel={compact_vector(ball_velocity())} m/s', flush=True)
    print(f'  grasp_xyz={compact_vector(grasp_position())} ball_minus_grasp={compact_vector(ball_position()-grasp_position())}', flush=True)
    print(f'  fingers L={int(left_touch)} R={int(right_touch)} forces L={lf:.3f}N R={rf:.3f}N, grip_cmd={data.ctrl[grip_act]:.1f}', flush=True)

def recorded_position(s):
    s = np.clip(s, 0, t[-1])
    return np.array([np.interp(s, t, path_x), np.interp(s, t, path_y), BALL_Z])

def recorded_velocity(s):
    a, b = max(0, s - 0.025), min(t[-1], s + 0.025)
    return (recorded_position(b) - recorded_position(a)) / (b - a) if b > a else np.zeros(3)

def guide_ball(s):
    force = (40 * (recorded_position(s) - ball_position())
             + 1.2 * (recorded_velocity(s) - ball_velocity()))
    force[2] += 0.058 * 9.81
    data.xfrc_applied[ball_body, :3] = np.clip(force, -5, 5)

def move_to(goal, feedforward=None):
    global q_target
    if feedforward is None:
        feedforward = np.zeros(3)
    point = grasp_position()
    pos_error = goal - point
    rot_error = Rotation.from_matrix(DESIRED_R @ hand_rotation().T).as_rotvec()
    linear = np.clip(3.0 * pos_error + feedforward, -0.85, 0.85)
    angular = np.clip(3.0 * rot_error, -1.0, 1.0)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, point, hand)
    jac = np.vstack([jp[:, v_ids], jr[:, v_ids]])
    dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.08**2 * np.eye(6), np.r_[linear, angular])
    dq = np.clip(dq, -1.5, 1.5)
    q_target += dq * model.opt.timestep * STEPS
    q_target = np.clip(q_target, data.qpos[q_ids] - 0.15, data.qpos[q_ids] + 0.15)
    limits = model.jnt_range[joints]
    q_target = np.clip(q_target, limits[:, 0] + 0.005, limits[:, 1] - 0.005)
    data.ctrl[arm_act] = q_target

# Observed positions only. The CSV replay is not accessed by the predictor.
observations = deque(maxlen=256)

def estimate_velocity():
    if len(observations)<3: return np.zeros(3)
    dt=observations[-1][0]-observations[0][0] if len(observations)<5 else observations[-1][0]-observations[-5][0]
    old=observations[0][1] if len(observations)<5 else observations[-5][1]
    return (observations[-1][1]-old)/dt if dt>0 else np.zeros(3)


class AnticipationMLP(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n,hidden),nn.ReLU(),
                               nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,2))
    def forward(self,x): return self.net(x)


predictor=None
if True:
    checkpoint=BASE/args.model
    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing anticipation checkpoint: {checkpoint}')
    # Load your own trusted checkpoint, which contains NumPy arrays.
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    history=int(ck['history'])
    history_dt=float(ck['dt'])
    x_mean=np.asarray(ck['x_mean'],dtype=np.float32)
    x_std=np.asarray(ck['x_std'],dtype=np.float32)
    y_scale=float(ck['y_scale'])
    supported=np.asarray(ck['horizons'],dtype=float)
    if history!=6 or abs(history_dt-(1/30))>1e-5:
        raise ValueError('Expected filtered six-frame / 30 Hz checkpoint')
    if len(x_mean)!=2*history+3 or len(x_std)!=len(x_mean):
        raise ValueError('Anticipation checkpoint feature dimension mismatch')
    if min(HORIZONS)<min(supported)-1e-6 or max(HORIZONS)>max(supported)+1e-6:
        raise ValueError('Planner requests a horizon outside model training range')
    predictor=AnticipationMLP(len(x_mean),int(ck['hidden']))
    predictor.load_state_dict(ck['state_dict'])
    predictor.eval()
    for param in predictor.parameters(): param.requires_grad_(False)
    print(f'Anticipation MLP loaded: {checkpoint.name}; history {history} x {history_dt}s',flush=True)


def predict_candidates():
    if len(observations)<3: return None
    current=observations[-1][1]
    horizons=HORIZONS
    future=None
    if predictor is not None:
        now=observations[-1][0]
        sample_times=now-np.arange(history-1,-1,-1)*history_dt
        times=np.asarray([o[0] for o in observations],dtype=float)
        if sample_times[0]>=times[0]:
            observed=np.asarray([o[1][:2] for o in observations])
            # Undo the simulation's rotation: model inputs were trained in
            # original CSV camera coordinates (relative displacements).
            sim_past=np.column_stack([np.interp(sample_times,times,observed[:,k]) for k in range(2)])
            relative=sim_past-sim_past[-1]
            csv_relative=relative[:, 0, None]*track_side + relative[:, 1, None]*track_direction
            centered=(np.arange(5)-2)*history_dt
            csv_vel=np.einsum('ki,k->i',csv_relative[-5:],centered)/(centered@centered)
            common=np.r_[csv_relative.reshape(-1),csv_vel].astype(np.float32)
            xs=np.column_stack([np.repeat(common[None,:],len(horizons),axis=0),horizons]).astype(np.float32)
            with torch.no_grad():
                pred=predictor(torch.from_numpy((xs-x_mean)/x_std)).numpy()*y_scale
            # Transform predicted displacement back to the rotated Panda frame.
            sim_delta=np.column_stack([pred@track_side,pred@track_direction])
            future=np.tile(current,(len(horizons),1))
            future[:,:2]+=sim_delta
    if future is None:
        return None  # Wait for 6 historical observations; never fall back to CV.
    future[:,2]=BALL_Z
    return list(zip(horizons,future))


def predict_interception():
    choices=predict_candidates()
    if choices is None: return None
    robot=grasp_position()
    candidates=[]
    for eta,candidate in choices:
        if not (.40<=candidate[0]<=.70 and -.22<=candidate[1]<=.22):
            continue
        robot_eta=np.linalg.norm(candidate-robot)/max(args.robot_speed,.05)+args.margin
        slack=eta-robot_eta
        candidates.append((slack,eta,robot_eta,candidate.copy()))
    if not candidates: return None
    feasible=[c for c in candidates if c[0]>=0]
    choice=min(feasible,key=lambda c:c[1]) if feasible else max(candidates,key=lambda c:c[0])
    slack,ball_eta,robot_eta,point=choice
    return point,ball_eta,robot_eta,slack


# Find the low starting pose without moving the robot during playback.
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
    initial_q = data.qpos[q_ids].copy()
    lower_q = initial_q.copy()
    limits = model.jnt_range[joints]
    for iteration in range(400):
        data.qpos[q_ids] = lower_q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        point=grasp_position()
        dp=desired-point
        dr=Rotation.from_matrix(reference_R@hand_rotation().T).as_rotvec()
        if np.linalg.norm(dp)<.002 and np.linalg.norm(dr)<.025:
            break
        jp,jr=np.zeros((3,model.nv)),np.zeros((3,model.nv))
        mujoco.mj_jac(model,data,jp,jr,point,hand)
        J6=np.vstack([jp[:,v_ids],jr[:,v_ids]])
        dx=np.r_[np.clip(dp,-.03,.03),np.clip(dr,-.07,.07)]
        dq=J6.T@np.linalg.solve(J6@J6.T + .06**2*np.eye(6),dx)
        lower_q=np.clip(lower_q+np.clip(dq,-.08,.08),limits[:,0]+.01,limits[:,1]-.01)
    data.qpos[q_ids]=lower_q
    data.qvel[:]=0
    mujoco.mj_forward(model,data)
    actual=grasp_position()
    err=np.linalg.norm(actual-desired)
    rot_err=np.linalg.norm(Rotation.from_matrix(reference_R@hand_rotation().T).as_rotvec())
    print(f'LOW-START IK: default Z={initial_grasp[2]:.4f}, requested Z={desired_grasp_z:.4f}, '
          f'achieved Z={actual[2]:.4f}, position error={err*100:.2f}cm, '
          f'orientation error={np.degrees(rot_err):.2f}deg',flush=True)
    if err>args.pose_tolerance or rot_err>.12:
        raise RuntimeError('Low initial pose IK did not reach the requested grasp pose. '
                           'Try --start-grasp-z 0.30 or 0.32; do not silently compare a bad IK pose.')
    table_id=obj(G,'intercept_table')
    penetrating=[]
    for contact in data.contact[:data.ncon]:
        if table_id in (contact.geom1,contact.geom2) and contact.dist < -.002:
            other=contact.geom2 if contact.geom1==table_id else contact.geom1
            if model.geom_bodyid[other]!=ball_body:
                penetrating.append(float(contact.dist))
    if penetrating:
        raise RuntimeError(f'Low starting pose penetrates table by up to {-min(penetrating)*100:.1f} cm. '
                           'Raise --start-grasp-z and retry.')
    return lower_q.copy(),initial_grasp

LOW_Q, BASE_GRASP = find_low_initial_q(args.start_grasp_z)

def reset_trial(video):
    global t, path_x, path_y, track_side, track_direction, q_target, DESIRED_R
    t, path_x, path_y, track_side, track_direction = load_track(video)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)
    data.qpos[q_ids] = LOW_Q
    data.qpos[ball_q:ball_q+3] = [path_x[0], path_y[0], BALL_Z]
    data.qpos[ball_q+3:ball_q+7] = [1, 0, 0, 0]
    data.qvel[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    q_target = data.qpos[q_ids].copy()
    DESIRED_R = data.xmat[hand].reshape(3, 3).copy()
    data.ctrl[arm_act] = q_target
    data.ctrl[grip_act] = 255




class ControlPanel:
    """Tk controls: separate window; MuJoCo viewer is passive and uses no mouse hooks."""
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('NN bilateral-contact paused grasp teleoperation')
        self.root.geometry('570x610')
        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.mode = tk.StringVar(value='ASSIST')
        self.info = tk.StringVar(value='Ready')
        self.status = tk.StringVar(value=f'Pause trigger: {args.pause_at.upper()} — use mouse/W/X, then Space')
        self.pressed = set()
        self.close_grip = False
        self.open_grip = False
        self.grip_latched = False
        self.pause = False
        self.reset = False
        self.next = False
        self.save = False
        self.step_once = False
        self.contact_editing = False
        self.restore_contact = False
        self.resume_contact = False
        self.stop = False
        self.x_corr = self.y_corr = 0.0
        self.z_corr = 0.0
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill='x')
        ttk.Label(top, text='Mode:').pack(side='left')
        ttk.Radiobutton(top, text='Assisted (recommended)', variable=self.mode, value='ASSIST').pack(side='left')
        # Manual interception deliberately disabled: only post-contact grasp adjustments are supported.
        ttk.Label(self.root, textvariable=self.info, padding=6, font=('Segoe UI', 10, 'bold')).pack()
        self.size = 310
        self.canvas = tk.Canvas(self.root, height=self.size, width=self.size, bg='#eaf0f5', highlightthickness=2, highlightbackground='#8293a5', takefocus=1)
        self.canvas.pack(pady=4)
        k=self.size//2
        self.canvas.create_line(0,k,self.size,k, fill='#9daab5')
        self.canvas.create_line(k,0,k,self.size, fill='#9daab5')
        self.canvas.create_oval(k-6,k-6,k+6,k+6, fill='#444', tags='cursor')
        self.canvas.create_text(k, 12, text='Forward (+Y)', fill='#444')
        self.canvas.create_text(k, self.size-12, text='Backward (-Y)', fill='#444')
        self.canvas.create_text(55,k-12, text='-X', fill='#444')
        self.canvas.create_text(self.size-55,k-12, text='+X', fill='#444')
        self.canvas.bind('<Motion>', self.mouse_move)
        self.canvas.bind('<ButtonPress-1>', self.grip_down)
        self.canvas.bind('<ButtonRelease-1>', self.grip_up)
        self.canvas.bind('<ButtonPress-3>', self.release_down)
        self.canvas.bind('<ButtonRelease-3>', self.release_up)
        self.root.bind_all('<KeyPress>', self.key_down)
        self.root.bind_all('<KeyRelease>', self.key_up)
        ttk.Label(self.root, textvariable=self.status).pack()
        buttons = ttk.Frame(self.root)
        buttons.pack(pady=5)
        for label, fn in [('Resume Space', self.request_resume), ('Step N', self.request_step), ('Restore C', self.request_restore), ('Restart T', self.request_reset), ('Save S', self.request_save)]:
            ttk.Button(buttons,text=label,command=fn).pack(side='left',padx=3)
        height_buttons = ttk.Frame(self.root)
        height_buttons.pack(pady=3)
        ttk.Button(height_buttons, text='UP +1 cm (W)', command=lambda: self.change_z(.01)).pack(side='left',padx=5)
        ttk.Button(height_buttons, text='DOWN -1 cm (X)', command=lambda: self.change_z(-.01)).pack(side='left',padx=5)
        ttk.Button(height_buttons, text='CENTER XY', command=self.center_xy).pack(side='left',padx=5)
        self.z_readout = tk.StringVar(value='Height correction: +0.0 cm')
        ttk.Label(self.root, textvariable=self.z_readout).pack()
        ttk.Label(self.root, text='Mouse location = XY correction; W/X = gripper height\n'
                                     'Left click = close (latched); right click = open\n'
                                     'Static pause at GRASP, bilateral contact, or APPROACH (--pause-at approach). Ball is pinned.\n'
                                     'N: extra preview steps; Space: resume; C: restore; T: restart.\n'
                                     'Paused preview steps are NOT saved as training data.', justify='center').pack()
        self.root.update()

    def mouse_move(self, event):
        k=self.size/2
        self.x_corr = np.clip((event.x-k)/k,-1,1)*args.xy_range
        self.y_corr = np.clip((k-event.y)/k,-1,1)*args.xy_range
        self.canvas.coords('cursor',event.x-6,event.y-6,event.x+6,event.y+6)
        self.contact_editing = True

    def grip_down(self, _):
        self.canvas.focus_set()
        self.close_grip=True
        self.grip_latched=True
        self.contact_editing = True
    def grip_up(self, _):
        self.close_grip=False
    def release_down(self, _):
        self.canvas.focus_set()
        self.open_grip=True
        self.grip_latched=False
        self.contact_editing = True
    def release_up(self, _):
        self.open_grip=False
    def request_step(self): self.step_once=True
    def request_restore(self): self.restore_contact=True
    def request_resume(self): self.resume_contact=True
    def request_save(self): self.save=True
    def request_reset(self): self.reset=True
    def request_next(self): self.next=True
    def toggle_pause(self): self.pause=not self.pause
    def quit(self): self.stop=True

    def change_z(self, amount):
        self.z_corr = float(np.clip(self.z_corr + amount,-.15,.15))
        self.z_readout.set(f'Height correction: {self.z_corr*100:+.1f} cm')
        self.contact_editing = True

    def center_xy(self):
        self.x_corr = self.y_corr = 0.
        k = self.size/2
        self.canvas.coords('cursor',k-6,k-6,k+6,k+6)
        self.contact_editing = True

    def key_down(self,event):
        key=event.keysym.lower()
        if key in self.pressed: return
        self.pressed.add(key)
        if key=='w': self.change_z(.01)
        elif key=='x': self.change_z(-.01)
        elif key=='s': self.request_save()
        elif key=='t': self.request_reset()
        elif key=='c': self.request_restore()
        elif key=='space': self.request_resume()
        elif key=='n': self.request_step()
        elif key=='p': self.toggle_pause()
        elif key=='1': self.mode.set('ASSIST')
        elif key=='2': self.mode.set('MANUAL')
    def key_up(self,event): self.pressed.discard(event.keysym.lower())
    def tick(self,dt):
        pass  # Height is changed directly by W/X or the UI buttons.
    def correction(self):
        return np.array([self.x_corr,self.y_corr,self.z_corr],dtype=float)
    def poll(self):
        try:
            self.root.update_idletasks()
            self.root.update()
            return not self.stop
        except tk.TclError:
            return False

# Use the same mouse and keyboard controls as the original collector.
panel=ControlPanel()
DEMO_DIR=BASE/args.demo_dir
DEMO_DIR.mkdir(parents=True,exist_ok=True)
RESULTS_DIR=BASE/'results'
RESULTS_DIR.mkdir(parents=True,exist_ok=True)

def save_demo(video, rows, result, manual_steps, kind, pose_info=None):
    if video not in TRAIN_VIDEOS:
        return
    if not rows:
        return
    stamp=time.strftime('%Y%m%d_%H%M%S')
    dest=DEMO_DIR/f'video_{video}_{stamp}_{uuid.uuid4().hex[:8]}_{kind}.npz'
    # All rows are aligned observation-action pairs at the time command was issued.
    arr={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    if pose_info is not None:
        arr.update(pose_info)
    np.savez_compressed(dest,**arr,video=np.int32(video),success=np.bool_(result['success']),
                        manual_steps=np.int32(manual_steps),mode='frozen_6frame_mlp_grasp_pose_collector',
                        il_dataset_split=np.str_('train'),
                        initial_grasp_z=np.float32(args.start_grasp_z),ball_friction=np.float32(2.5),
                        pause_at=np.str_(args.pause_at),
                        counterfactual_contact_edit=np.bool_(result.get('counterfactual_contact_edit', False)),
                        record_schema_version=np.int32(2))
    print(f'Demonstration saved: {dest.name} ({len(rows)} samples; manual edits: {manual_steps})',flush=True)


def run_trial(video, viewer):
    # Enable corrections and demonstration recording only for IL training videos.
    collect_demo = video in TRAIN_VIDEOS
    reset_trial(video)
    observations.clear()
    panel.x_corr = panel.y_corr = panel.z_corr = 0.
    panel.z_readout.set('Height correction: +0.0 cm')
    panel.contact_editing = False
    panel.grip_latched = False
    panel.close_grip = panel.open_grip = False
    panel.reset = panel.next = panel.save = False
    panel.step_once = panel.restore_contact = panel.resume_contact = False
    panel.pressed.clear()
    phase = 'OBSERVE'
    grasp_start = lift_start = None
    lift_target = None
    released = False
    contact_duration = max_contact_duration = 0.
    max_ball_lift = hold_duration = max_hold_duration = 0.
    ever_both = False
    manual_steps = 0
    rows = []
    hand_target = None
    contact_snapshot = None
    contact_paused = False
    manual_goal = None
    paused_steps = 0
    manual_grasp_offset = None
    contact_ball_initial = None
    contact_grasp_initial = None
    resumed_grasp_target = None
    grasp_start_snapshot = None
    pause_ball_initial = None
    pause_grasp_initial = None
    pause_ball_velocity = None
    pause_time_initial = None
    pause_finger_command = None
    pause_joint_positions = None
    pause_joint_velocities = None
    pause_contacts = None
    pause_contact_forces = None
    pause_trigger = None
    last_diag_time = -1e9
    first_any_contact_time = None
    first_bilateral_contact_time = None

    def capture_pause(trigger):
        nonlocal contact_snapshot, contact_paused, manual_goal
        nonlocal contact_ball_initial, contact_grasp_initial
        nonlocal pause_ball_initial, pause_grasp_initial, pause_ball_velocity
        nonlocal pause_time_initial, pause_finger_command, pause_joint_positions
        nonlocal pause_joint_velocities, pause_contacts, pause_contact_forces, pause_trigger
        if not collect_demo or contact_snapshot is not None:
            return
        contact_snapshot = {
            'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(),
            'ctrl': data.ctrl.copy(), 'act': data.act.copy(),
            'time': float(data.time), 'q_target': q_target.copy(),
        }
        contact_paused = True
        manual_goal = grasp_position().copy()
        pause_ball_initial = ball_position()
        pause_grasp_initial = grasp_position()
        pause_ball_velocity = ball_velocity()
        pause_time_initial = float(data.time)
        pause_finger_command = float(data.ctrl[grip_act])
        pause_joint_positions = data.qpos[q_ids].copy()
        pause_joint_velocities = data.qvel[v_ids].copy()
        pause_contacts = np.asarray(finger_contacts(), np.int8)
        pause_contact_forces = np.asarray(finger_contact_forces(), np.float32)
        pause_trigger = trigger
        contact_ball_initial = pause_ball_initial.copy()
        contact_grasp_initial = pause_grasp_initial.copy()
        panel.status.set(f'{trigger} — STATIC CORRECTION: mouse XY, W/X Z; Space to resume')
        log_grasp_state('PAUSED '+trigger, video)
        print(f'  contact/GRASP input 6D: {compact_vector(np.r_[pause_ball_initial-pause_grasp_initial,pause_ball_velocity])}', flush=True)

    print(f'Video {video}: NN anticipation, low initial grasp z={grasp_position()[2]:.4f}m, friction=2.5, speed=1.0x', flush=True)
    viewer.sync()
    while viewer.is_running() and not panel.stop and data.time < max(12., t[-1] + 6.):
        if not panel.poll():
            break
        panel.info.set(f'Video {video} | {"CONTACT PAUSED" if contact_paused else phase} | t={data.time:.2f}s | {"IL TRAIN — MANUAL" if collect_demo else "AUTONOMOUS — NOT SAVED"}')
        if contact_paused:
            if panel.reset:
                panel.reset = False
                return {'video': video, 'phase': 'RESTART', 'success': False, 'manual_steps': manual_steps}
            # Freeze physics until a user step. Snapshot restores both ball velocity
            # and robot configuration; changes made while paused are only previews.
            if panel.restore_contact:
                panel.restore_contact = False
                data.qpos[:] = contact_snapshot['qpos']
                data.qvel[:] = contact_snapshot['qvel']
                data.ctrl[:] = contact_snapshot['ctrl']
                data.time = contact_snapshot['time']
                data.act[:] = contact_snapshot['act']
                data.xfrc_applied[:] = 0
                q_target[:] = contact_snapshot['q_target']
                mujoco.mj_forward(model, data)
                manual_goal = grasp_position().copy()
                panel.x_corr = panel.y_corr = panel.z_corr = 0.
                panel.z_readout.set('Height correction: +0.0 cm')
                panel.contact_editing = False
                panel.grip_latched = False
                panel.close_grip = panel.open_grip = False
                viewer.sync()
            # Any edit moves the robot immediately, with the ball fixed in place.
            # N remains available to advance extra preview steps.
            edit_goal = manual_goal + panel.correction()
            needs_motion = np.linalg.norm(grasp_position() - edit_goal) > .002
            should_advance = (panel.step_once or panel.contact_editing or needs_motion)
            if should_advance:
                panel.step_once = False
                panel.contact_editing = False
                # Step the robot while pinning the ball to its first-contact pose.
                # This is a planning aid, NOT a physically valid moving-ball demonstration.
                ball_fixed = contact_snapshot['qpos'][ball_q:ball_q+7].copy()
                correction = panel.correction()
                move_to(manual_goal + correction)
                if panel.grip_latched: data.ctrl[grip_act] = 0
                elif panel.open_grip: data.ctrl[grip_act] = 255
                for _ in range(STEPS):
                    data.qpos[ball_q:ball_q+7] = ball_fixed
                    data.qvel[ball_v:ball_v+6] = 0
                    data.xfrc_applied[ball_body] = 0
                    mujoco.mj_step(model, data)
                data.qpos[ball_q:ball_q+7] = ball_fixed
                data.qvel[ball_v:ball_v+6] = 0
                mujoco.mj_forward(model, data)
                paused_steps += 1
                panel.status.set(f'CONTACT PAUSED — Z {panel.z_corr*100:+.1f} cm | robot moves automatically; Space to resume')
                viewer.sync()
            if panel.resume_contact:
                panel.resume_contact = False
                contact_paused = False
                # Keep the manually positioned robot, but restore the original
                # recorded ball velocity and contact timestamp.
                data.qpos[ball_q:ball_q+7] = contact_snapshot['qpos'][ball_q:ball_q+7]
                data.qvel[ball_v:ball_v+6] = contact_snapshot['qvel'][ball_v:ball_v+6]
                data.time = contact_snapshot['time']
                data.xfrc_applied[ball_body] = 0
                mujoco.mj_forward(model, data)
                observations.clear()
                observations.append((float(data.time), ball_position()))
                # Preserve the demonstrated robot-to-ball offset after resuming.
                manual_grasp_offset = grasp_position() - ball_position()
                resumed_grasp_target = grasp_position().copy()
                print(f'[RESUME] video={video} trigger={pause_trigger} preview_steps={paused_steps}', flush=True)
                print(f'  absolute pose delta={compact_vector(resumed_grasp_target-pause_grasp_initial)} m', flush=True)
                print(f'  corrected ball-relative pose={compact_vector(resumed_grasp_target-ball_position())} m', flush=True)
                print(f'  gripper command={float(data.ctrl[grip_act]):.1f} counterfactual={paused_steps>0}',flush=True)
                panel.x_corr = panel.y_corr = panel.z_corr = 0.
                panel.z_readout.set('Height correction: +0.0 cm')
                panel.grip_latched = True if data.ctrl[grip_act] < 128 else panel.grip_latched
                # For early APPROACH demonstrations keep interception active rather than forcing a grasp.
                if pause_trigger == 'APPROACH_EARLY':
                    phase = 'APPROACH'
                    released = False
                    grasp_start = None
                    hand_target = grasp_position().copy()
                else:
                    phase = 'GRASP'
                    released = True
                    grasp_start = float(data.time)
                contact_duration = 0.
                manual_steps += paused_steps
                panel.status.set('Normal-speed grasp running; paused preview steps excluded from BC data')
                viewer.sync()
            time.sleep(.015)
            continue
        if panel.reset:
            panel.reset = False
            return {'video': video, 'phase': 'RESTART', 'success': False, 'manual_steps': manual_steps}
        if panel.next:
            panel.next = False
            return {'video': video, 'phase': 'SKIPPED', 'success': False, 'manual_steps': manual_steps}
        if panel.pause:
            time.sleep(.015)
            continue
        dt = model.opt.timestep * STEPS
        panel.tick(dt)
        elapsed = float(data.time)
        data.xfrc_applied[ball_body] = 0
        if not released and elapsed <= t[-1]:
            guide_ball(elapsed)
        ball_p = ball_position()
        observations.append((elapsed, ball_p.copy()))
        vel = estimate_velocity()
        prediction = predict_interception()  # NN-only, no CV fallback
        position_before = grasp_position().copy()
        finger_before = float(data.ctrl[grip_act])
        old_phase = phase
        correction = np.zeros(3) if not collect_demo or phase in ('OBSERVE','APPROACH') else panel.correction()
        manual_mode = False
        manual_input = (collect_demo and phase in ('GRASP','LIFT') and (np.linalg.norm(correction) > .003 or panel.grip_latched or panel.open_grip))
        if manual_input:
            manual_steps += 1
        goal = position_before.copy()

        if phase in ('OBSERVE', 'APPROACH'):
            if prediction is not None:
                phase = 'APPROACH'
                if hand_target is None:
                    hand_target = position_before.copy()
                goal = (prediction[0] + ((resumed_grasp_target - pause_grasp_initial) if pause_trigger == 'APPROACH_EARLY' and resumed_grasp_target is not None else np.zeros(3))) + correction
                move_to(goal)
            elif manual_mode:
                if hand_target is None:
                    hand_target = position_before.copy()
                goal = hand_target + correction
                move_to(goal)
            else:
                data.ctrl[arm_act] = q_target
            data.ctrl[grip_act] = 255
            local = ball_in_hand()
            inside = (abs(local[0]) < .030 and abs(local[1]) < .020
                      and abs(local[2] - .103) < .035)
            if args.pause_at == 'approach' and phase == 'APPROACH' and contact_snapshot is None and elapsed >= args.approach_pause_time:
                capture_pause('APPROACH_EARLY')
            if inside and phase == 'APPROACH' and not contact_paused:
                released = True
                data.xfrc_applied[ball_body] = 0
                phase = 'GRASP'
                grasp_start = elapsed
                # Record first GRASP-phase state before either pause or human edits.
                grasp_start_snapshot = {
                    'time': float(data.time),
                    'ball_xyz': ball_position(),
                    'grasp_xyz': grasp_position(),
                    'ball_velocity': ball_velocity(),
                    'ball_velocity_est': vel.copy(),
                    'gripper_command': float(data.ctrl[grip_act]),
                    'joint_positions': data.qpos[q_ids].copy(),
                    'joint_velocities': data.qvel[v_ids].copy(),
                }
                log_grasp_state('GRASP INITIAL STATE', video)
                if args.pause_at == 'grasp':
                    capture_pause('GRASP_INIT')
            elif elapsed > t[-1] + .5 or (phase == 'APPROACH' and ball_p[1] > .20):
                phase = 'FAILED'
        elif phase == 'GRASP':
            if manual_grasp_offset is not None:
                goal = ball_p + manual_grasp_offset + correction
            else:
                goal = (position_before if manual_mode else ball_p) + correction
            move_to(goal, feedforward=np.zeros(3) if manual_mode else vel)
            progress = np.clip((elapsed - grasp_start) / CLOSE_TIME, 0, 1)
            data.ctrl[grip_act] = 255 * (1-progress)
            if elapsed - grasp_start > 1.:
                phase = 'FAILED'
        elif phase == 'LIFT':
            goal = lift_target + correction
            move_to(goal)
            data.ctrl[grip_act] = 0
            if elapsed - lift_start >= 3.:
                phase = 'DONE'
        else:
            data.ctrl[arm_act] = q_target
            data.ctrl[grip_act] = 0

        if collect_demo and phase in ('GRASP', 'LIFT'):
            if panel.open_grip:
                data.ctrl[grip_act] = 255
            elif panel.grip_latched:
                data.ctrl[grip_act] = 0

        # Matches the original BC demonstration feature/label schema.
        rows.append({
            'time': elapsed,
            'ball_xyz': ball_p.copy(),
            'ball_velocity_est': vel.copy(),
            'grasp_xyz': position_before,
            'grasp_to_ball': ball_p - position_before,
            'joint_positions': data.qpos[q_ids].copy(),
            'joint_velocities': data.qvel[v_ids].copy(),
            'finger_command_before': finger_before,
            'joint_position_targets': data.ctrl[arm_act].copy(),
            'finger_command': float(data.ctrl[grip_act]),
            'goal_xyz': goal.copy(),
            'user_correction_xyz': correction.copy(),
            'user_grip_close': int(panel.grip_latched),
            'user_grip_open': int(panel.open_grip),
            'manual_input': int(manual_input),
            'phase': old_phase,
            'left_contact': np.int8(finger_contacts()[0]),
            'right_contact': np.int8(finger_contacts()[1]),
            'contact_forces_N': np.asarray(finger_contact_forces(), np.float32),
            'after_paused_edit': np.int8(paused_steps > 0),
        })
        if contact_paused:
            viewer.sync()
            continue
        for _ in range(STEPS):
            mujoco.mj_step(model, data)
            l, r = finger_contacts()
            if first_any_contact_time is None and (l or r):
                first_any_contact_time = float(data.time)
                log_grasp_state('FIRST FINGER CONTACT', video)
            if first_bilateral_contact_time is None and l and r:
                first_bilateral_contact_time = float(data.time)
                log_grasp_state('FIRST BILATERAL CONTACT', video)
            if args.pause_at == 'bilateral' and contact_snapshot is None and l and r:
                capture_pause('BILATERAL_CONTACT')
                break
            both = l and r
            ever_both |= both
            if phase == 'GRASP' and both:
                contact_duration += model.opt.timestep
                max_contact_duration = max(max_contact_duration, contact_duration)
            elif phase == 'GRASP':
                contact_duration = 0.
            lift = ball_position()[2] - BALL_Z
            max_ball_lift = max(max_ball_lift, lift)
            gap = np.linalg.norm(ball_position() - grasp_position())
            if lift > .06 and both and gap < .08:
                hold_duration += model.opt.timestep
                max_hold_duration = max(max_hold_duration, hold_duration)
            else:
                hold_duration = 0.
        if not contact_paused and phase == 'GRASP' and contact_duration >= CONTACT_TIME and not manual_mode:
            phase = 'LIFT'
            lift_start = float(data.time)
            lift_target = grasp_position() + np.array([0.,0.,LIFT_HEIGHT])
        if phase == 'GRASP' and manual_mode and panel.z_corr >= .07:
            phase = 'LIFT'
            lift_start = float(data.time)
            lift_target = grasp_position().copy()
        result = {
            'video': video, 'phase': phase,
            'il_split': ('train' if collect_demo else 'validation' if video in VALID_VIDEOS else 'test' if video in TEST_VIDEOS else 'challenge'),
            'success': bool(max_hold_duration >= .20),
            'both_fingers_contacted': bool(ever_both),
            'max_lift_m': round(float(max_ball_lift),4),
            'hold_s': round(float(max_hold_duration),3),
            'contact_s': round(float(max_contact_duration),3),
            'sim_time_s': round(float(data.time),3),
            'manual_steps': manual_steps,
            'pause_preview_steps': paused_steps,
            'counterfactual_contact_edit': bool(paused_steps > 0),
            'initial_grasp_z': round(float(BASE_GRASP[2]),4),
            'low_start_grasp_z': round(float(args.start_grasp_z),4),
            'first_any_contact_time': first_any_contact_time,
            'first_bilateral_contact_time': first_bilateral_contact_time,
            'pause_trigger': pause_trigger,
        }
        if collect_demo and panel.save:
            panel.save = False
            save_demo(video, rows, result, manual_steps, 'snapshot')
        viewer.sync()
        if contact_paused:
            continue
        time.sleep(dt/max(.05,args.speed))
        if phase in ('FAILED','DONE'):
            break
    result.update(phase=phase, success=bool(max_hold_duration >= .20),
                  sim_time_s=round(float(data.time),3), manual_steps=manual_steps)
    if collect_demo and phase in ('DONE','FAILED'):
        pose_info = {
            'pause_trigger': np.str_(pause_trigger if pause_trigger else 'NONE'),
            'pause_preview_steps': np.int32(paused_steps),
            'first_any_contact_time': np.float32(first_any_contact_time if first_any_contact_time is not None else np.nan),
            'first_bilateral_contact_time': np.float32(first_bilateral_contact_time if first_bilateral_contact_time is not None else np.nan),
        }
        if grasp_start_snapshot is not None:
            gs = grasp_start_snapshot
            pose_info.update({
                'grasp_start_time': np.float32(gs['time']),
                'grasp_start_ball_xyz': gs['ball_xyz'].astype(np.float32),
                'grasp_start_grasp_xyz': gs['grasp_xyz'].astype(np.float32),
                'grasp_start_ball_velocity': gs['ball_velocity'].astype(np.float32),
                'grasp_start_ball_velocity_est': gs['ball_velocity_est'].astype(np.float32),
                'grasp_start_gripper_command': np.float32(gs['gripper_command']),
                'grasp_start_joint_positions': gs['joint_positions'].astype(np.float32),
                'grasp_start_joint_velocities': gs['joint_velocities'].astype(np.float32),
                'grasp_start_feature_vector': np.r_[gs['ball_xyz']-gs['grasp_xyz'], gs['ball_velocity']].astype(np.float32),
            })
        if pause_ball_initial is not None:
            pose_info.update({
                'contact_ball_xyz': pause_ball_initial.astype(np.float32),
                'contact_grasp_xyz': pause_grasp_initial.astype(np.float32),
                'contact_ball_velocity': pause_ball_velocity.astype(np.float32),
                'contact_feature_vector': np.r_[pause_ball_initial-pause_grasp_initial,pause_ball_velocity].astype(np.float32),
                'pause_time': np.float32(pause_time_initial),
                'pause_gripper_command': np.float32(pause_finger_command),
                'pause_joint_positions': pause_joint_positions.astype(np.float32),
                'pause_joint_velocities': pause_joint_velocities.astype(np.float32),
                'pause_finger_contacts': pause_contacts.astype(np.int8),
                'pause_contact_forces_N': pause_contact_forces.astype(np.float32),
            })
        if resumed_grasp_target is not None:
            pose_info.update({
                'resume_grasp_xyz': resumed_grasp_target.astype(np.float32),
                'resume_grasp_offset_xyz': (resumed_grasp_target - pause_grasp_initial).astype(np.float32),
                'resume_ball_relative_xyz': (resumed_grasp_target - pause_ball_initial).astype(np.float32),
                'corrected_relative_pose_xyz': (resumed_grasp_target - pause_ball_initial).astype(np.float32),
                'pose_label_valid': np.bool_(bool(result['success']) and paused_steps > 0 and pause_trigger != 'APPROACH_EARLY'),
            })
            if grasp_start_snapshot is not None:
                pose_info['grasp_start_to_corrected_pose_xyz'] = (resumed_grasp_target-grasp_start_snapshot['grasp_xyz']).astype(np.float32)
            print(f'[POSE LABEL] video {video} success={result["success"]} pause_at={args.pause_at}', flush=True)
            print(f'  feature={compact_vector(pose_info["contact_feature_vector"])}', flush=True)
            print(f'  correction={compact_vector(pose_info["resume_grasp_offset_xyz"])} m', flush=True)
            print(f'  corrected_relative_pose={compact_vector(pose_info["corrected_relative_pose_xyz"])} m', flush=True)
        else:
            pose_info['pose_label_valid'] = np.bool_(False)
        print(f'[FINAL] video={video} phase={phase} success={result["success"]} lift={max_ball_lift:.4f}m hold={max_hold_duration:.3f}s preview={paused_steps} steps',flush=True)
        save_demo(video, rows, result, manual_steps,
                  'success' if result['success'] else 'failure', pose_info=pose_info)
    return result


def main():
    results = []
    # Repeat training demonstrations; evaluate other splits only once.
    schedule = [
        (video, attempt)
        for attempt in range(1, args.repeats + 1)
        for video in VIDEOS
        if attempt == 1 or video in TRAIN_VIDEOS
    ]
    try:
        with mujoco.viewer.launch_passive(model,data) as viewer:
            print('NN-ANTICIPATION GRASP POSE DEMONSTRATIONS | pause_at:', args.pause_at)
            print('IL training videos pause for correction; other videos run autonomously.')
            print('All requested videos:',VIDEOS,'| IL train:',TRAIN_VIDEOS,'| IL validation:',VALID_VIDEOS,'| IL test:',TEST_VIDEOS,'| Challenge:',CHALLENGE_VIDEOS)
            print('Anticipation model unchanged:',args.model,'| IL split:',il_split_file.name)
            i = 0
            while i < len(schedule) and viewer.is_running() and not panel.stop:
                video, attempt = schedule[i]
                print(f'\n=== {i+1}/{len(schedule)} VIDEO {video}, attempt {attempt} ===', flush=True)
                result = run_trial(video, viewer)
                print(result, flush=True)
                if result['phase'] == 'RESTART':
                    continue
                if result['phase'] != 'SKIPPED':
                    results.append(result)
                    pd.DataFrame(results).to_csv(RESULTS_DIR/'teleop_trials.csv',index=False)
                i += 1
                deadline = time.monotonic()+max(0,args.pause)
                while time.monotonic() < deadline and viewer.is_running() and not panel.stop:
                    if not panel.poll():
                        break
                    time.sleep(.01)
    finally:
        try:
            panel.root.destroy()
        except tk.TclError:
            pass
    if results:
        print('\nFINAL RESULTS\n'+pd.DataFrame(results).to_string(index=False),flush=True)
    print('Saved training demonstrations:', DEMO_DIR, flush=True)
    print('Saved trial results:', RESULTS_DIR/'teleop_trials.csv', flush=True)

if __name__ == '__main__':
    main()
