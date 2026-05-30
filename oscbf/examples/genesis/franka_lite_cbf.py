"""Genesis + OSCBF integration demo.

Loads a Franka Panda into Genesis. A fake RL policy emits joint velocities;
OSCBF filters them through a CBF that enforces joint limits + an EE
workspace box. Renders a chase mp4.

Run:
    conda activate oscbf
    python genesis_franka_oscbf.py

The full pipeline runs in a single Python process: Genesis (PyTorch/Taichi
on the GPU) does physics + rendering; JAX (CPU) does the safety-filter QP.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import imageio.v3 as iio
import torch

import genesis as gs

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.oscbf_configs import OSCBFVelocityConfig
from oscbf.assets import ASSETS_DIR


URDF_PATH = ASSETS_DIR / "franka_panda" / "panda.urdf"


# ── CBF: joint limits + EE workspace box ────────────────────────────
@jax.tree_util.register_static
class JointLimitsAndBoxConfig(OSCBFVelocityConfig):
    """Enforces:
      - joint angle limits (2 × n_joints)
      - end-effector position inside a workspace box (6 = 2 × 3)
    All barriers are relative-degree-1 (velocity control) → h_1.
    """

    def __init__(self, robot: Manipulator, ee_min, ee_max):
        self.ee_min = np.asarray(ee_min)
        self.ee_max = np.asarray(ee_max)
        super().__init__(robot)  # default weights so the QP is well-posed

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        ee = self.robot.ee_position(q)
        return jnp.concatenate(
            [
                q - jnp.asarray(self.robot.joint_lower_limits),
                jnp.asarray(self.robot.joint_upper_limits) - q,
                ee - jnp.asarray(self.ee_min),
                jnp.asarray(self.ee_max) - ee,
            ]
        )

    def alpha(self, h):
        return 8.0 * h


# ── "RL policy" stand-in ────────────────────────────────────────────
def fake_rl_policy(t: float) -> np.ndarray:
    """Sinusoidal joint-velocity command. Intentionally pushes toward
    joint 1's upper limit so we can see the safety filter engage."""
    qdot = np.zeros(7)
    qdot[0] = 1.5                            # constant push on q1
    qdot[1] = 0.8 * np.sin(0.5 * t)
    qdot[3] = 0.4 * np.cos(0.7 * t)
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
    ap.add_argument("--steps", type=int, default=1500)   # 6 s sim at 4 ms dt → ~25 s wall
    ap.add_argument("--no_filter", action="store_true",
                     help="bypass the safety filter (for comparison)")
    ap.add_argument("--out", default="/home/erl/safety_filter/franka_oscbf.mp4")
    args = ap.parse_args()

    # Workspace box (the EE must stay inside)
    EE_MIN = (0.20, -0.40, 0.10)
    EE_MAX = (0.70, 0.40, 0.80)

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

    # Workspace-box visual (8 thin edges) so we can see what the CBF enforces
    edge_thickness = 0.01
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

    panda = scene.add_entity(gs.morphs.URDF(file=str(URDF_PATH), fixed=True))

    cam = scene.add_camera(res=(1280, 720), pos=(1.5, 1.5, 1.2),
                            lookat=(0.3, 0.0, 0.4), fov=55, GUI=False)
    scene.build()

    # Initial joint configuration matches OSCBF examples
    q_init = np.array([0, -np.pi/3, 0, -5*np.pi/6, 0, np.pi/2, 0])
    panda.set_dofs_position(torch.tensor(q_init, dtype=torch.float32, device=gs.device))
    panda.set_dofs_velocity(torch.zeros(7, dtype=torch.float32, device=gs.device))
    panda.set_dofs_kp([400.0] * 7)
    panda.set_dofs_kv([40.0] * 7)
    # Run a few settling steps
    for _ in range(20):
        panda.control_dofs_velocity(torch.zeros(7, dtype=torch.float32, device=gs.device))
        scene.step()

    # ── OSCBF setup ─────────────────────────────────────────────────
    robot = load_panda()
    cbf = CBF.from_config(JointLimitsAndBoxConfig(robot, EE_MIN, EE_MAX))
    print(f"[boot] CBF built with {2 * robot.num_joints + 6} barriers")
    # JIT warm-up (use float64 — JAX strict dtype matching)
    _ = cbf.safety_filter(q_init.astype(np.float64), np.zeros(7, dtype=np.float64))

    # ── Drive loop ──────────────────────────────────────────────────
    frames = []
    n_active = 0
    n_violations = 0
    t = 0.0
    SIM_DT = 0.01
    for k in range(args.steps):
        # Genesis returns float32; OSCBF/cbfpy expects float64 throughout
        # (jax.jvp dtype-strict). Cast explicitly.
        q = panda.get_dofs_position().cpu().numpy().astype(np.float64)
        qdot_des = fake_rl_policy(t).astype(np.float64)

        if args.no_filter:
            qdot_cmd = qdot_des
        else:
            qdot_safe = np.asarray(cbf.safety_filter(q, qdot_des))
            qdot_cmd = qdot_safe
            if np.linalg.norm(qdot_safe - qdot_des) > 1e-3:
                n_active += 1

        panda.control_dofs_velocity(torch.tensor(qdot_cmd, dtype=torch.float32, device=gs.device))
        scene.step()
        t += SIM_DT

        # Count violations: any joint outside its limits, or EE outside the box
        q_now = panda.get_dofs_position().cpu().numpy()
        ee_now = np.asarray(robot.ee_position(jnp.asarray(q_now)))
        out_of_box = np.any(ee_now < np.asarray(EE_MIN) - 1e-3) or np.any(ee_now > np.asarray(EE_MAX) + 1e-3)
        out_of_joint = (np.any(q_now < np.asarray(robot.joint_lower_limits) - 1e-3) or
                         np.any(q_now > np.asarray(robot.joint_upper_limits) + 1e-3))
        if out_of_box or out_of_joint:
            n_violations += 1

        if k % 4 == 0:
            frames.append(_grab(cam))

        if k % 100 == 0:
            print(f"  step {k:4d}  t={t:5.2f}s  q1={q[0]:+.2f}  "
                   f"EE=[{ee_now[0]:+.2f},{ee_now[1]:+.2f},{ee_now[2]:+.2f}]  "
                   f"{'(filt off)' if args.no_filter else 'filt_active' if n_active else ''}")

    iio.imwrite(args.out, np.stack(frames), fps=25, codec="libx264")
    print(f"\n[done] wrote {args.out} ({len(frames)} frames)")
    if not args.no_filter:
        print(f"        safety filter active on {n_active}/{args.steps} steps "
               f"({100 * n_active / args.steps:.0f}%)")
    print(f"        constraint violations: {n_violations}/{args.steps}")


if __name__ == "__main__":
    main()
