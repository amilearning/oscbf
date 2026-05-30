"""Genesis + OSCBF: Piper with sphere-vs-box collision avoidance.

CBF:
  - 12 joint-angle limits (2 × 6)
  - 12 sphere-vs-box distance barriers (1 per Piper collision sphere)

A fake policy drives the EE straight at a fixed box. With the filter on, the
arm contorts around it; without, it slams through.

Run:
    conda activate oscbf
    python genesis_piper_box_cbf.py            # filter on
    python genesis_piper_box_cbf.py --no_filter
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


# ── CBF: joint limits + sphere-vs-box distances ─────────────────────
@jax.tree_util.register_static
class BoxObstacleConfig(OSCBFVelocityConfig):
    """For every robot collision sphere at world position p with radius r:

        dist = ||p − clip(p, box_min, box_max)|| − r

    dist > 0 → sphere outside box. CBF wants dist ≥ 0 ∀ sphere.
    """

    def __init__(self, robot: Manipulator, box_center, box_half_extents, margin=0.01):
        # margin: tighten the box by this much in the CBF (vs. visualized box)
        # to absorb Genesis velocity-PD tracking lag.
        self.box_center = np.asarray(box_center, dtype=np.float64)
        self.box_half_extents = np.asarray(box_half_extents, dtype=np.float64) + margin
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        h_joint = jnp.concatenate([
            q - jnp.asarray(self.robot.joint_lower_limits),
            jnp.asarray(self.robot.joint_upper_limits) - q,
        ])
        col = self.robot.link_collision_data(q)          # (K, 4)
        sphere_pos = col[:, :3]
        sphere_rad = col[:, 3]
        delta = sphere_pos - jnp.asarray(self.box_center)
        clamped = jnp.clip(delta, -jnp.asarray(self.box_half_extents),
                                   jnp.asarray(self.box_half_extents))
        closest = jnp.asarray(self.box_center) + clamped
        dist = jnp.linalg.norm(sphere_pos - closest, axis=-1) - sphere_rad
        return jnp.concatenate([h_joint, dist])

    def alpha(self, h):
        return 15.0 * h


# ── Fake policy: Jacobian-IK toward a target inside the box ─────────
def fake_rl_policy(t: float, q: np.ndarray, robot, target_center) -> np.ndarray:
    """Drive EE straight at `target_center` via damped least-squares IK.
    With filter off → EE plows into the box. With filter on → EE stops at
    box surface (or wherever the safety set allows)."""
    target = np.asarray(target_center) + 0.02 * np.array([np.sin(0.4 * t),
                                                          np.cos(0.4 * t),
                                                          0.0])
    ee = np.asarray(robot.ee_position(jnp.asarray(q)))
    err = target - ee                           # (3,)
    J_full = np.asarray(robot.ee_jacobian(jnp.asarray(q)))   # (6, 6) (xyz + rpy)
    J = J_full[:3]                              # (3, 6) — position rows
    damping = 1e-2
    JJt = J @ J.T + damping * np.eye(3)
    qdot = J.T @ np.linalg.solve(JJt, err)
    # Scale to a reasonable max speed
    return np.clip(qdot * 4.0, -2.0, 2.0)


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
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--out", default="/home/erl/safety_filter/piper_box_cbf.mp4")
    args = ap.parse_args()

    # Box positioned with all 12 spheres ≥ 14 cm clear at start pose.
    # Wider in y than x/z so the arm can't trivially swing around it.
    BOX_CENTER = (0.40, 0.00, 0.15)
    BOX_HALF   = (0.06, 0.20, 0.10)

    # ── Genesis setup ───────────────────────────────────────────────
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(
            background_color=(0.92, 0.92, 0.96),
            ambient_light=(0.7, 0.7, 0.7),
            shadow=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    # Visual-only box (collision=False) so only the CBF stops the arm
    scene.add_entity(
        gs.morphs.Box(
            size=tuple(2 * h for h in BOX_HALF),
            pos=BOX_CENTER, fixed=True, collision=False,
        ),
        surface=gs.surfaces.Default(color=(0.85, 0.15, 0.15), roughness=0.6),
    )

    piper = scene.add_entity(gs.morphs.URDF(file=str(PIPER_URDF), fixed=True))

    cam = scene.add_camera(res=(1280, 720), pos=(1.2, -1.2, 0.95),
                           lookat=(0.30, 0.0, 0.35), fov=55, GUI=False)
    scene.build()

    n = 6
    # Home: arm extended forward, EE at (0.56, 0, 0.48) — clearly outside the
    # box at (0.40, 0, 0.45) ± (0.08, 0.08, 0.08).
    q_init = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0], dtype=np.float32)
    piper.set_dofs_position(torch.tensor(q_init, device=gs.device))
    piper.set_dofs_velocity(torch.zeros(n, device=gs.device))
    piper.set_dofs_kp([300.0] * n)
    piper.set_dofs_kv([30.0] * n)
    for _ in range(20):
        piper.control_dofs_velocity(torch.zeros(n, device=gs.device))
        scene.step()

    # ── OSCBF setup ─────────────────────────────────────────────────
    robot = load_piper(with_collision=True)
    K_col = robot.link_collision_data(jnp.asarray(q_init)).shape[0]
    config = BoxObstacleConfig(robot, BOX_CENTER, BOX_HALF)
    cbf = CBF.from_config(config)
    print(f"[boot] CBF: {2*robot.num_joints} joint barriers + "
          f"{K_col} sphere-vs-box barriers = {2*robot.num_joints + K_col} total")
    _ = cbf.safety_filter(q_init.astype(np.float64), np.zeros(n, dtype=np.float64))

    # ── Drive loop ──────────────────────────────────────────────────
    frames = []
    n_active = 0
    min_clr_log = []
    SIM_DT = 0.01
    t = 0.0
    for k in range(args.steps):
        q = piper.get_dofs_position().cpu().numpy().astype(np.float64)
        qdot_des = fake_rl_policy(t, q, robot, BOX_CENTER).astype(np.float64)

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

        # Diagnostic: min clearance from any sphere to the box (visualized box)
        col = np.asarray(robot.link_collision_data(jnp.asarray(q)))
        delta = col[:, :3] - np.asarray(BOX_CENTER)
        clamped = np.clip(delta, -np.asarray(BOX_HALF), np.asarray(BOX_HALF))
        closest = np.asarray(BOX_CENTER) + clamped
        dists = np.linalg.norm(col[:, :3] - closest, axis=-1) - col[:, 3]
        min_clr_log.append(float(dists.min()))

        if k % 4 == 0:
            frames.append(_grab(cam))

        if k % 100 == 0:
            ee = np.asarray(robot.ee_position(jnp.asarray(q)))
            tag = "(filt off)" if args.no_filter else f"filt {n_active}/{k+1}"
            print(f"  step {k:4d}  t={t:5.2f}s  EE=[{ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:+.2f}]  "
                  f"min_clr={min_clr_log[-1]:+.3f} m  {tag}")

    iio.imwrite(args.out, np.stack(frames), fps=25, codec="libx264")
    print(f"\n[done] wrote {args.out} ({len(frames)} frames)")
    n_pen = sum(1 for c in min_clr_log if c < 0)
    print(f"        sphere-in-box frames: {n_pen}/{args.steps} "
          f"(min clr over run: {min(min_clr_log):+.3f} m)")
    if not args.no_filter:
        print(f"        filter active on {n_active}/{args.steps} steps "
              f"({100 * n_active / args.steps:.0f}%)")


if __name__ == "__main__":
    main()
