# Real-World Ball Motion Anticipation and Human-Guided Robotic Interception

**Humanoid Robotics Internship Challenge | October 2026**

A robotic manipulation project that uses **self-collected, real-world ball-motion data** to drive a **Franka Emika Panda** manipulator in **MuJoCo**. A neural motion predictor anticipates the moving ball's position, a Jacobian-based controller executes an interception, and a residual imitation-learning policy improves grasping using human-guided corrections.

## Demonstrations

Two recordings are provided in the [`Demo Videos/`] folder:

- **Teleoperation:** pausing at grasp initiation and adjusting the robot's grasp to collect correction demonstrations.
- **Autonomous interception:** the trained controller intercepting, grasping, and lifting a moving ball.

## Demo Videos

### 1. Human-Guided Teleoperation
[▶ Watch Teleoperation Demo](Demo%20Videos/Demo_Imitation_Learning_Teleoperation.mp4)

### 2. Autonomous Ball Interception
[▶ Watch Autonomous Grasping Demo](Demo%20Videos/Demo_Video_Anticipation_MLP_Without_Imitation_Intervention.mp4)

## Results at a glance

| Controller | Successful grasps | Success rate |
|---|---:|---:|
| Anticipation + Jacobian control (baseline without Imitation-learning correction) | 24 / 30 | 80.0% |
| **Anticipation + residual imitation learning (hybrid)** | **26 / 30** | **86.7%** |

The hybrid recovered **Videos 9 and 18**, with **no regressions** among the 24 baseline-successful trials. Video 9 belongs to the **held-out imitation-learning test set**. These figures describe evaluations on the 30 recorded trajectories; they are not estimates of performance in arbitrary new environments.

| Imitation-learning split | Baseline | Hybrid |
|---|---:|---:|
| Training (20 videos) | 17/20 | 18/20 |
| Validation (3 videos) | 3/3 | 3/3 |
| Test (6 videos) | 4/6 | 5/6 |
| Challenge (1 video) | 0/1 | 0/1 |

The complete per-video evaluation and generated plots are in `results/`, including `evaluation_hybrid_il_30_compare.csv`, `hybrid_success_by_split_compare.png`, `hybrid_per_video_comparison.png`, and `hybrid_max_lift_comparison.png`.

## System overview

```text
Recorded real-world tennis-ball videos
                 |
                 v
    HSV-based tracking and calibration
                 |
                 v
    30 Hz XY trajectories (30 videos)
                 |
                 v
       6-frame anticipation MLP
        (future XY displacement)
                 |
                 v
      Interception-point selection
                 |
                 v
     Damped least-squares Jacobian
         Franka Panda controller
                 |
                 v
        MuJoCo grasp and lift
                 |
          Human correction
        at grasp initiation
                 |
                 v
   Residual grasp-correction MLP
                 |
                 v
    Autonomous hybrid evaluation
```

### 1. Real-world data acquisition

The input data consists of **30 recorded tennis-ball motion sequences**. `ExtractTrajectories.py` uses OpenCV HSV colour segmentation and contour tracking to estimate the ball centre. Each recording is spatially calibrated using an **80 cm reference** and resampled to **30 Hz**. The resulting CSV files provide time, position, and velocity estimates (`time`, `x`, `y`, `vx`, `vy`).

The extracted trajectories are stored in `data/trajectories/`. `visualize_velocities.py` plots their speed profiles. During simulation, recorded trajectories are mapped into the Panda's workspace and used to drive a simulated ball; the anticipation network itself receives a history of **observed ball positions**, not future ground-truth trajectory points.

### 2. Neural motion anticipation

`train_anticipation.py` trains a feed-forward MLP to predict the ball's **future planar displacement** from six observed position frames, an estimated velocity, and a requested prediction horizon (how far into the future the model is predicting).

- Observation history: **6 frames at 30 Hz**, covering approximately **0.167 s**.
- Network: fully connected **64 → 64 → 2** with ReLU activations.
- Inputs: relative XY position history, velocity estimate, prediction horizon.
- Outputs: future XY displacement, transformed into the simulator's reference frame.
- Training: video-level train/validation/test separation, Adam optimisation, validation-based checkpoint selection.

The constant-velocity (CV) model is used as a benchmark to evaluate the anticipation MLP's prediction performance. The MLP outperforms the CV baseline by learning variations in ball motion rather than assuming constant velocity. Prediction errors for both methods are reported in `results/variable_anticipation_errors_6frames_30.csv`.

### 3. Interception and robotic control

`mujoco_baseline.py` integrates the trained predictor with a Franka Panda manipulator simulated in MuJoCo. Candidate future ball positions are checked against workspace limits and estimated robot travel time to select an interception point. The robot follows the target with **damped least-squares differential inverse kinematics**, using the end-effector Jacobian:

\[
\dot{q} = J^{\mathsf{T}}(JJ^{\mathsf{T}} + \lambda^2 I)^{-1} v_{\mathrm{task}}
\]

The controller executes an **observe → approach → grasp → lift** sequence. Grasp success is assessed through simulated bilateral finger contact and sustained ball lifting, not merely end-effector proximity.

### 4. Human-guided demonstration collection

`Teleop_train.py` provides a MuJoCo viewer and a Tkinter teleoperation panel. For imitation-learning **training videos only**, a trial can pause at the start of grasping. The operator adjusts the gripper's **XY position** with the mouse and **Z height** with the W/X keys, previews the adjustment, and resumes the simulation.

The collector records the original grasp-initiation state, the corrected grasp pose, and the ensuing trial outcome as `.npz` files under `data/il_demonstrations/`. Validation, test, and challenge trajectories are not used as imitation-learning training targets.

For the final experiment, the demonstration set contained **60 trials**: three per training video. Five successful correction examples from originally unsuccessful training videos (8, 10, and 18) were selected along with 17 zero-correction references from baseline-successful training videos.

**Important methodological detail:** During teleoperation, the simulation is paused and the ball is temporarily held in place while the operator adjusts the gripper's position. These corrections are used to train the residual imitation-learning policy. The trained policy is then evaluated autonomously under continuous MuJoCo physics, without pauses or human intervention.

The residual grasp-correction MLP is trained using a weighted mean squared error (MSE) loss:

\[
\mathcal{L}_{IL} = 0.5\mathcal{L}_{recovery} + 0.5\mathcal{L}_{zero}
\]

**The loss assigns equal overall weight to successful human-guided grasp corrections (50%) and zero-correction references from baseline-successful trajectories (50%). This encourages the model to recover unsuccessful grasps while avoiding unnecessary adjustments to already successful ones.**

### 5. Residual imitation learning

`train_imitation_grasp.py` trains a small residual MLP (**6 → 32 → 32 → 3**, Tanh activations) to predict a one-time XYZ correction at grasp initiation. Its inputs are the ball-to-grasp relative position and the simulated ball's XYZ velocity. The model learns corrected grasp positions from successful demonstrations while using zero-correction targets to discourage unnecessary interventions on already-successful cases.

The trained policy is stored at `Models/imitation_grasp_policy_30.pt`. `evaluate_hybrid_il.py` combines this frozen policy with the frozen anticipation network and evaluates baseline and hybrid controllers on the same 30 trajectories, without pausing or teleoperation.

### 6. Hybrid control architecture: prediction, control, and learned correction

The **hybrid approach** combines a model-based robot controller with two learned components, each responsible for a different part of the task:

1. **Motion anticipation (learned):** A six-frame MLP predicts the ball's future XY displacement over candidate time horizons, allowing the robot to plan where to intercept it.
2. **Interception and motion execution (model-based):** A workspace-aware planner selects an interception point, and a damped least-squares Jacobian controller moves the Panda toward it. The robot's approach, gripper closing, and lift phases remain controlled by the original simulation logic.
3. **Residual grasp correction (learned from teleoperation):** **Only when the robot reaches `GRASP_INIT`**, a second MLP takes the current ball-to-grasp relative XYZ position and ball XYZ velocity and predicts a *one-time* grasp-target offset, `Δp = [Δx, Δy, Δz]`. The grasp-phase controller then tracks the moving ball with that offset:

   ```text
   grasp_target(t) = ball_position(t) + Δp
   ```

   The deployed correction is clipped to **±4 cm in X/Y** and **±3 cm in Z**. The residual model does not replace the trajectory predictor, inverse kinematics, or gripper controller.

```text
Observed ball positions ──> 6-frame anticipation MLP ──> Future ball positions
                                                        │
                                                        v
                                               Interception planner
                                                        │
                                                        v
                                               Jacobian Panda control
                                                        │
                                             On reaching GRASP_INIT
                                                        │
            Ball–grasp XYZ + ball velocity ──> Residual correction MLP
                                                        │
                                                        v
                                      Ball-following grasp target + Δp
                                                        │
                                                        v
                                               Gripper close and lift
```

The residual network is trained on **successful human-guided grasp corrections from baseline-failure training videos**, balanced against **zero-correction references** for already-successful training videos. In autonomous evaluation, both networks are frozen and no human intervention or simulation pause occurs.

This division of responsibilities is the main design choice: the learned anticipation model handles *when and where to intercept*, conventional Jacobian control handles *how to move*, and residual imitation learning refines *the final grasp*. Because the residual policy activates only at grasp initiation, it cannot fix missed approaches that never reach that phase (such as Video 11).


## Installation

Tested with the package versions recorded in `requirements.txt` on Windows. Use a compatible Python environment with Tkinter support for teleoperation.

```powershell
python -m pip install -r requirements.txt
```

### Franka Panda simulation assets

The repository includes the Franka Panda model assets from [Google DeepMind's MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), under `mujoco_menagerie/franka_emika_panda/`. **No separate robot-model download is needed** when these assets are included in the checkout.

The scripts look for `mujoco_menagerie/franka_emika_panda/scene.xml` or a standalone `franka_emika_panda/scene.xml`. The MuJoCo **Python package** is installed through `requirements.txt`; the bundled Menagerie files supply the robot description and meshes, not the MuJoCo simulator itself.


## Reproduce the evaluation

**Fastest way to inspect the supplied trained models and results:**

```powershell
# Baseline controller, 30 videos, no viewer
python mujoco_baseline.py --headless

# Paired baseline-versus-hybrid evaluation, 30 videos
python evaluate_hybrid_il.py --headless

# Optional: visually inspect a held-out recovered video
python evaluate_hybrid_il.py --videos 9
```

The last command runs both baseline and hybrid rollouts for Video 9 with the MuJoCo viewer. Evaluation outputs are saved under `results/`.

### Re-train from the provided data

```powershell
# Re-train anticipation MLP from trajectory CSVs
python train_anticipation.py

# Re-train residual policy from the supplied selected demonstration set
python train_imitation_grasp.py

# Re-evaluate the resulting checkpoints
python evaluate_hybrid_il.py --headless
```

**Caution:** Training scripts overwrite their default checkpoint paths. Back up the provided `Models/` and `results/` files before retraining. Re-running teleoperation in the default demonstration directory adds new `.npz` files; it does **not** overwrite existing demonstrations.


## Limitations

- **Limited evaluation set:** Results reflect 30 recorded ball trajectories, with six held-out imitation-learning test videos.
- **Unresolved failures:** Videos **3, 8, 10, and 11** remain unsuccessful under the final hybrid controller. Challenge Video 11 is not recovered by the grasp-correction model.
- **Correction timing:** The residual policy acts only at grasp initiation, so trajectories that never reach the grasp phase cannot be corrected through this mechanism.
- **Simulation gap:** Success is measured in MuJoCo, not verified on a physical Panda. Contact mechanics, ball tracking noise, and camera calibration may differ on real hardware.
- **Human supervision:** Corrected demonstrations are collected through temporary paused-scene edits; they are not full real-time teleoperated dynamic trajectories.


This repository was developed as a submission for the Humanoid Robotics internship challenge.
