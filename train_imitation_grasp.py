"""Train a residual grasp-correction MLP from local teleoperation .npz files.

Run from the project folder:
    python train_imitation_grasp_30.py

Trains only on the IL training videos in data/imitation_learning_split_30.json.
The anticipation MLP and the autonomous MuJoCo controller are untouched.
"""
import argparse
import csv
import json
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
P = argparse.ArgumentParser()
P.add_argument('--demo-dir', default='data/il_demonstrations')
P.add_argument('--split', default='data/imitation_learning_split_30.json')
P.add_argument('--output', default='models/imitation_grasp_policy_30.pt')
P.add_argument('--epochs', type=int, default=400)
P.add_argument('--hidden', type=int, default=32)
P.add_argument('--lr', type=float, default=0.001)
P.add_argument('--seed', type=int, default=42)
P.add_argument('--recovery-weight', type=float, default=0.5,
               help='Fraction of training loss assigned to recovered originally-failed grasps')
args = P.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

with (ROOT / args.split).open(encoding='utf-8') as f:
    split = json.load(f)
training_ids = set(map(int, split['train']))
nontraining_ids = set(map(int, split['validation'])) | set(map(int, split['test'])) | set(map(int, split.get('challenge', [])))
if training_ids & nontraining_ids:
    raise ValueError('IL dataset split leaks between train and held-out groups')
if not {8, 10, 18}.issubset(training_ids):
    raise ValueError('Expected Videos 8, 10 and 18 in imitation training')
if 11 in training_ids:
    raise ValueError('Challenge Video 11 must remain outside imitation training')

# Baseline failures from the 24/30 autonomous MuJoCo evaluation.
# Video 11 also failed but remains outside the IL training set.
BASELINE_FAILED_GRASP = {8, 10, 18}
BASELINE_SUCCESS = training_ids - BASELINE_FAILED_GRASP
if len(training_ids) != 20 or len(BASELINE_SUCCESS) != 17:
    raise ValueError('Unexpected IL split / baseline-success classification')

FEATURE_NAMES = [
    'ball_minus_grasp_x', 'ball_minus_grasp_y', 'ball_minus_grasp_z',
    'ball_velocity_x', 'ball_velocity_y', 'ball_velocity_z'
]
OUTPUT_NAMES = ['correction_x_m', 'correction_y_m', 'correction_z_m']
DEMO_DIR = ROOT / args.demo_dir
files = sorted(DEMO_DIR.glob('video_*.npz'))
if not files:
    raise FileNotFoundError(f'No video_*.npz files in {DEMO_DIR}')

rows = []
skips = Counter()
all_trials = Counter()
for path in files:
    try:
        with np.load(path, allow_pickle=False) as d:
            v = int(d['video'])
            if v not in training_ids:
                skips['outside training split'] += 1
                continue
            all_trials[v] += 1
            success = bool(d['success'])
            if not success:
                skips['failed assisted trial: outcome recorded, not an imitation target'] += 1
                continue
            if str(d['pause_trigger']) != 'GRASP_INIT':
                skips['not grasp-initiation data'] += 1
                continue
            if 'pose_label_valid' in d and not bool(d['pose_label_valid']):
                skips['invalid correction label'] += 1
                continue
            if 'grasp_start_feature_vector' not in d or 'grasp_start_to_corrected_pose_xyz' not in d:
                skips['missing feature or correction field'] += 1
                continue
            state = np.asarray(d['grasp_start_feature_vector'], dtype=np.float32).reshape(-1)
            correction = np.asarray(d['grasp_start_to_corrected_pose_xyz'], dtype=np.float32).reshape(-1)
            if state.shape != (6,) or correction.shape != (3,):
                skips['incorrect array shape'] += 1
                continue
            if not (np.isfinite(state).all() and np.isfinite(correction).all()):
                skips['nonfinite state or correction'] += 1
                continue
            if np.max(np.abs(correction)) > 0.15:
                skips['correction > 15 cm: possible corrupted demonstration'] += 1
                continue
            # Successful manual alternatives on already-successful baselines are
            # retained for audit, but NOT taught as a preferred nonzero action:
            # success does not establish that an intervention was better.
            rows.append({'video': v, 'x': state, 'y': correction,
                         'recovery': v in BASELINE_FAILED_GRASP,
                         'preferred': v in BASELINE_FAILED_GRASP,
                         'file': path.name})
    except (OSError, KeyError, ValueError) as exc:
        print(f'Skipping unreadable {path.name}: {exc}')
        skips['unreadable demonstration'] += 1

# Add ONE genuine zero-action target for each successful baseline video.
# Take the autonomous grasp-start state from one corresponding saved trial;
# the pre-pause feature is captured before any operator edits.
zero_rows = []
for v in sorted(BASELINE_SUCCESS):
    candidates = [r for r in rows if r['video'] == v]
    if not candidates:
        # A failed assisted trial can still contain a valid PRE-intervention state.
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as d:
                    if int(d['video']) != v or 'grasp_start_feature_vector' not in d:
                        continue
                    state = np.asarray(d['grasp_start_feature_vector'], np.float32).reshape(-1)
                    if state.shape == (6,) and np.isfinite(state).all():
                        candidates = [{'x': state, 'file': path.name}]
                        break
            except (OSError, KeyError, ValueError):
                continue
    if candidates:
        zero_rows.append({'video': v, 'x': candidates[0]['x'],
                          'y': np.zeros(3, np.float32),
                          'recovery': False, 'preferred': True,
                          'file': 'BASELINE_ZERO:' + candidates[0]['file']})
    else:
        print(f'WARNING: no captured grasp-start state for baseline-success Video {v}')

recovery_rows = [r for r in rows if r['preferred']]
if not recovery_rows:
    raise RuntimeError('No successful recovery demonstrations found for Videos 8, 10 or 18')
if not zero_rows:
    raise RuntimeError('No baseline-success states found for zero-correction targets')

# Prevent any one video with many repeats from dominating its entire class.
# Give the recovery class and zero class configurable proportions.
train_rows = recovery_rows + zero_rows
rec_counts = Counter(r['video'] for r in recovery_rows)
zero_counts = Counter(r['video'] for r in zero_rows)
rec_videos = len(rec_counts)
zero_videos = len(zero_counts)
weights = []
for r in train_rows:
    if r['recovery']:
        weights.append(args.recovery_weight / (rec_videos * rec_counts[r['video']]))
    else:
        weights.append((1 - args.recovery_weight) / (zero_videos * zero_counts[r['video']]))
weights = np.asarray(weights, np.float32)
weights /= weights.sum()

X = np.stack([r['x'] for r in train_rows])
Y = np.stack([r['y'] for r in train_rows]) * 100.0  # supervised labels in cm
x_mean = np.average(X, axis=0, weights=weights).astype(np.float32)
x_var = np.average((X - x_mean)**2, axis=0, weights=weights)
x_std = np.maximum(np.sqrt(x_var), np.array([.005,.005,.005,.05,.05,.05], np.float32)).astype(np.float32)
xt = torch.tensor((X - x_mean) / x_std, dtype=torch.float32)
yt = torch.tensor(Y, dtype=torch.float32)
wt = torch.tensor(weights, dtype=torch.float32)

class CorrectionMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 3))
    def forward(self, x):
        return self.net(x)

net = CorrectionMLP(args.hidden)
optim = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)
print('\nTRAINING DATA')
print('NPZ files found:', len(files))
print('Training demonstrations per video:', dict(sorted(all_trials.items())))
print('Assisted successes:', len(rows), '| assisted failures/other skipped:', dict(skips))
print('Recovered grasp examples:', len(recovery_rows), 'from videos:', sorted(rec_counts))
print('Zero-correction references:', len(zero_rows), 'from videos:', sorted(zero_counts))
print('Loss allocation: recovery {:.0%}, baseline zeros {:.0%}'.format(args.recovery_weight, 1-args.recovery_weight))
print('NOTE: Successful altered grasps on already-successful videos are not preferred corrections.')
print('NOTE: Training error is NOT held-out grasp success. Test physics separately.')

loss_history = []
for epoch in range(1, args.epochs + 1):
    net.train()
    prediction = net(xt)
    mse = ((prediction - yt)**2).mean(dim=1)
    loss = (wt * mse).sum()
    optim.zero_grad()
    loss.backward()
    optim.step()
    loss_history.append(float(loss.item()))
    if epoch == 1 or epoch % 50 == 0 or epoch == args.epochs:
        print(f'Epoch {epoch:3d} | weighted correction MSE {loss.item():.4f} cm^2')

net.eval()
with torch.no_grad():
    fit = net(xt).numpy()
errors = np.linalg.norm(fit - Y, axis=1)
print('\nTRAIN-FIT DIAGNOSTICS (not an independent validation)')
print('Recovery action error: {:.2f} cm'.format(float(np.mean(errors[:len(recovery_rows)]))))
print('Zero-action output magnitude: {:.2f} cm'.format(float(np.mean(np.linalg.norm(fit[len(recovery_rows):], axis=1)))))

ckpt = dict(
    state_dict=net.state_dict(), hidden=args.hidden,
    feature_mean=x_mean, feature_std=x_std,
    feature_names=FEATURE_NAMES, output_names=OUTPUT_NAMES,
    output_scale=100.0, # model outputs centimeters; divide by 100 for metres
    target='grasp_start_to_corrected_pose_xyz',
    label_definition='corrected_grasp_xyz - original_grasp_start_xyz (world axes, m)',
    policy_stage='GRASP_INIT', training_videos=sorted(training_ids),
    recovery_videos=sorted(rec_counts), baseline_success_videos=sorted(zero_counts),
    blocked_videos=sorted(nontraining_ids),
    recovery_weight=args.recovery_weight, n_recovery=len(recovery_rows),
    n_baseline_zero=len(zero_rows), epochs=args.epochs, seed=args.seed,
    caution='Paused counterfactual data. Must evaluate continuous-physics rollout; no guarantee of success.'
)
out = ROOT / args.output
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, out)
results_dir = ROOT / 'results'
results_dir.mkdir(parents=True, exist_ok=True)
report = results_dir / 'imitation_training_targets_30.csv'
with report.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['video','source','target_type','dx_cm','dy_cm','dz_cm','weight','pred_dx_cm','pred_dy_cm','pred_dz_cm'])
    for r, w, pred in zip(train_rows, weights, fit):
        writer.writerow([r['video'],r['file'], 'recovery' if r['recovery'] else 'baseline_zero',
                         *np.round(r['y'] * 100,4), round(float(w),7), *np.round(pred,4)])
print('\nSaved:', out)
print('Saved:', report)
print('Next: evaluate this checkpoint under uninterrupted MuJoCo physics, without teleoperation.')


# Save training diagnostics; these are not held-out simulation results.
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(1, len(loss_history) + 1), loss_history)
ax.set(xlabel='Epoch', ylabel='Weighted MSE (cm²)', title='Residual grasp policy training loss')
ax.grid(alpha=0.25)
fig.tight_layout()
loss_plot = results_dir / 'imitation_training_loss.png'
fig.savefig(loss_plot, dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
groups = [errors[:len(recovery_rows)], errors[len(recovery_rows):]]
ax.boxplot(groups, tick_labels=['Recovery', 'Zero correction'], showmeans=True)
ax.set(ylabel='Prediction error (cm)', title='Training-set correction errors')
ax.grid(axis='y', alpha=0.25)
fig.tight_layout()
error_plot = results_dir / 'imitation_training_errors.png'
fig.savefig(error_plot, dpi=180)
plt.close(fig)
print('Saved:', loss_plot)
print('Saved:', error_plot)
