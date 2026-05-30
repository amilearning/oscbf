"""Genesis + OSCBF on the AgileX Piper (lite CBF).

Same pattern as genesis_franka_oscbf.py but for the 6-DOF Piper:
  - Joint limit barriers (2 × 6 = 12)
  - EE workspace-box barriers (6)
No collision-sphere model yet — add piper_collision_model.py later to unlock
arm-vs-obstacle CBFs.

Run:
    conda activate oscbf
    python genesis_piper_oscbf.py
    python genesis_piper_oscbf.py --no_filter   # comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import imageio.v3 as iio
import torch

import genesis as gs

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator
from oscbf.core.oscbf_configs import OSCBFVelocityConfig

from oscbf.core.manipulator import load_piper
from oscbf.assets import ASSETS_DIR
PIPER_URDF = ASSETS_DIR / "piper" / "piper.urdf"


# ── CBF: joint limits + EE workspace box ────────────────────────────
@jax.tree_util.register_static
class PiperLiteConfig(OSCBFVelocityConfig):
    """Joint-limit + EE-workspace-box barriers (all rel-degree-1)."""

    def __init__(self, robot: Manipulator, ee_min, ee_max, margin=0.02):
        # margin: the CBF enforces a box `margin` m tighter than (ee_min, ee_max)
        # so the actual EE stays inside the visualized box despite tracking lag.
        self.ee_min = np.asarray(ee_min, dtype=np.float64) + margin
        self.ee_max = np.asarray(ee_max, dtype=np.float64) - margin
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        ee = self.robot.ee_position(q)
        return jnp.concatenate([
            q - jnp.asarray(self.robot.joint_lower_limits),    # ≥ 0
            jnp.asarray(self.robot.joint_upper_limits) - q,    # ≥ 0
            ee - jnp.asarray(self.ee_min),                     # ≥ 0
            jnp.asarray(self.ee_max) - ee,                     # ≥ 0
        ])

    def alpha(self, h):
        # Higher gain helps because Genesis's velocity-PD tracking lags the
        # commanded qdot, so the barrier needs to react earlier.
        return 20.0 * h


# ── "RL policy" stand-in ────────────────────────────────────────────
def fake_rl_policy(t: float) -> np.ndarray:
    """6-DOF joint-velocity command. Pushes joint1 toward its upper limit and
    rocks joint2/joint4 so the EE will also walk out of the workspace box if
    the filter doesn't intervene."""
    qdot = np.zeros(6)
    qdot[0] = 1.2                         # constant push on q1
    qdot[1] = 0.6 * np.sin(0.5 * t)       # shoulder rock
    qdot[3] = 0.5 * np.cos(0.7 * t)       # wrist swing
    qdot[4] = 0.3 * np.sin(0.6 * t)
    return qdot


def _grab(cam):
    rgb = cam.render()[0]
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.cpu().numpy()
    if rgb.ndim == 4:
        rgb = rgb[0]
    if rgb.dtype in (np.float32, np.float64):
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    return rgb[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)   # ~12 s sim at 10 ms
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--out", default="/home/erl/safety_filter/piper_oscbf.mp4")
    args = ap.parse_args()

    # Workspace box sized for Piper-with-gripper's ~0.8 m reach
    EE_MIN = (0.15, -0.30, 0.10)
    EE_MAX = (0.65,  0.30, 0.75)

    # ── Genesis setup ───────────────────────────────────────────────
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(
            background_color=(0.9, 0.9, 0.95),
            ambient_light=(0.7, 0.7, 0.7),
            shadow=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    # Workspace-box wireframe (12 thin edges)
    edge_thickness = 0.008
    box_color = (0.0, 0.8, 0.0)
    def add_box_edge(p0, p1):
        cx = (p0[0] + p1[0]) / 2; cy = (p0[1] + p1[1]) / 2; cz = (p0[2] + p1[2]) / 2
        sx = max(abs(p1[0] - p0[0]), edge_thickness)
        sy = max(abs(p1[1] - p0[1]), edge_thickness)
        sz = max(abs(p1[2] - p0[2]), edge_thickness)
        scene.add_entity(
            gs.morphs.Box(size=(sx, sy, sz), pos=(cx, cy, cz), fixed=True, collision=False),
            surface=gs.surfaces.Default(color=box_color, roughness=0.5),
        )
    bx0, by0, bz0 = EE_MIN
    bx1, by1, bz1 = EE_MAX
    corners = [(bx0, by0, bz0), (bx1, by0, bz0), (bx1, by1, bz0), (bx0, by1, bz0),
               (bx0, by0, bz1), (bx1, by0, bz1), (bx1, by1, bz1), (bx0, by1, bz1)]
    for a, b in [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]:
        add_box_edge(corners[a], corners[b])

    piper = scene.add_entity(gs.morphs.URDF(file=str(PIPER_URDF), fixed=True))

    cam = scene.add_camera(res=(1280, 720), pos=(1.1, -1.1, 0.9),
                           lookat=(0.25, 0.0, 0.35), fov=55, GUI=False)
    scene.build()

    n = 6
    # Home pose: arm raised forward (matches Piper joint limits)
    q_init = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0], dtype=np.float32)
    piper.set_dofs_position(torch.tensor(q_init, device=gs.device))
    piper.set_dofs_velocity(torch.zeros(n, device=gs.device))
    piper.set_dofs_kp([300.0] * n)   # stiffer than v1 → cleaner velocity tracking
    piper.set_dofs_kv([30.0] * n)
    for _ in range(20):
        piper.control_dofs_velocity(torch.zeros(n, device=gs.device))
        scene.step()

    # ── OSCBF setup ─────────────────────────────────────────────────
    robot = load_piper()
    cbf = CBF.from_config(PiperLiteConfig(robot, EE_MIN, EE_MAX))
    print(f"[boot] CBF built with {2 * robot.num_joints + 6} barriers "
          f"({2 * robot.num_joints} joint + 6 workspace)")
    _ = cbf.safety_filter(q_init.astype(np.float64), np.zeros(n, dtype=np.float64))

    # ── Drive loop ──────────────────────────────────────────────────
    frames = []
    n_active = 0
    n_violations = 0
    t = 0.0
    SIM_DT = 0.01
    for k in range(args.steps):
        q = piper.get_dofs_position().cpu().numpy().astype(np.float64)
        qdot_des = fake_rl_policy(t).astype(np.float64)

        if args.no_filter:
            qdot_cmd = qdot_des
        else:
            qdot_safe = np.asarray(cbf.safety_filter(q, qdot_des))
            qdot_cmd = qdot_safe
            if np.linalg.norm(qdot_safe - qdot_des) > 1e-3:
                n_active += 1

        piper.control_dofs_velocity(torch.tensor(qdot_cmd, dtype=torch.float32, device=gs.device))
        scene.step()
        t += SIM_DT

        q_now = piper.get_dofs_position().cpu().numpy()
        ee_now = np.asarray(robot.ee_position(jnp.asarray(q_now)))
        out_of_box = (np.any(ee_now < np.asarray(EE_MIN) - 1e-3) or
                      np.any(ee_now > np.asarray(EE_MAX) + 1e-3))
        out_of_joint = (np.any(q_now < np.asarray(robot.joint_lower_limits) - 1e-3) or
                        np.any(q_now > np.asarray(robot.joint_upper_limits) + 1e-3))
        if out_of_box or out_of_joint:
            n_violations += 1

        if k % 4 == 0:
            frames.append(_grab(cam))

        if k % 100 == 0:
            tag = "(filt off)" if args.no_filter else f"filt {n_active}/{k+1}"
            print(f"  step {k:4d}  t={t:5.2f}s  q1={q[0]:+.2f}  "
                  f"EE=[{ee_now[0]:+.2f},{ee_now[1]:+.2f},{ee_now[2]:+.2f}]  {tag}")

    iio.imwrite(args.out, np.stack(frames), fps=25, codec="libx264")
    print(f"\n[done] wrote {args.out} ({len(frames)} frames)")
    if not args.no_filter:
        print(f"        safety filter active on {n_active}/{args.steps} steps "
              f"({100 * n_active / args.steps:.0f}%)")
    print(f"        constraint violations: {n_violations}/{args.steps}")


if __name__ == "__main__":
    main()
