"""Genesis + OSCBF: collision avoidance with a physical box obstacle.

Loads a Franka Panda + a physical box obstacle into Genesis. A fake "RL
policy" drives joint velocities aimed straight at the box. OSCBF
intercepts the command and corrects it so every robot collision sphere
stays out of the box.

Run:
    conda activate oscbf
    python genesis_franka_box_cbf.py            # with filter (collision avoided)
    python genesis_franka_box_cbf.py --no_filter  # raw policy (robot crashes into box)
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
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.oscbf_configs import OSCBFVelocityConfig
from oscbf.assets import ASSETS_DIR


URDF_PATH = ASSETS_DIR / "franka_panda" / "panda.urdf"


# ── CBF: box obstacle (sphere-vs-AABB distance) + joint limits ─────
@jax.tree_util.register_static
class BoxObstacleConfig(OSCBFVelocityConfig):
    """Enforces:
      - joint angle limits (2 × n_joints)
      - every robot collision sphere stays out of an axis-aligned box

    Robot collision spheres come from OSCBF's hand-coded Franka collision
    model (Manipulator.link_collision_data). For each sphere at world
    position p with radius r:

        delta       = p − box_center
        clamped     = clip(delta, -half_extents, +half_extents)
        closest_pt  = box_center + clamped
        dist        = ‖p − closest_pt‖ − r

    Outside box: dist > 0. Inside box: dist < 0. CBF wants dist ≥ 0.
    """

    def __init__(self, robot: Manipulator, box_center, box_half_extents):
        self.box_center = np.asarray(box_center, dtype=np.float64)
        self.box_half_extents = np.asarray(box_half_extents, dtype=np.float64)
        super().__init__(robot)  # default 1/1/1 task/rot/joint weights

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        # Joint limits
        h_joint = jnp.concatenate(
            [
                q - jnp.asarray(self.robot.joint_lower_limits),
                jnp.asarray(self.robot.joint_upper_limits) - q,
            ]
        )
        # Box-sphere distances for every robot collision sphere
        col = self.robot.link_collision_data(q)           # (K, 4) — xyz + radius
        sphere_pos = col[:, :3]
        sphere_rad = col[:, 3]
        delta = sphere_pos - jnp.asarray(self.box_center)             # (K, 3)
        clamped = jnp.clip(delta, -jnp.asarray(self.box_half_extents),
                                  jnp.asarray(self.box_half_extents))  # (K, 3)
        closest_pt_on_box = jnp.asarray(self.box_center) + clamped     # (K, 3)
        dist = jnp.linalg.norm(sphere_pos - closest_pt_on_box, axis=-1) - sphere_rad
        return jnp.concatenate([h_joint, dist])

    def alpha(self, h):
        return 5.0 * h


# ── "RL policy" stand-in: pushes the EE straight at the box ──────────
def fake_rl_policy(t: float, q: np.ndarray) -> np.ndarray:
    """Drive the EE toward (0.55, 0.0, 0.45) and through whatever's there."""
    # Move forward (+x) by raising joint 1 slightly and pulling joint 2 forward
    # and joint 4 to extend. Mostly hand-tuned to head into the box.
    qdot = np.zeros(7)
    qdot[1] = +0.5 * np.cos(0.3 * t)       # gentle vertical sway
    qdot[3] = +0.8                          # always extend the elbow (drives EE forward)
    qdot[5] = +0.3 * np.sin(0.4 * t)       # wrist wiggle
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
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--no_filter", action="store_true",
                     help="bypass OSCBF (for comparison)")
    ap.add_argument("--out", default="/home/erl/safety_filter/franka_box_cbf.mp4")
    args = ap.parse_args()

    # Place a 30 cm box right in the EE's natural reach
    BOX_CENTER = (0.55, 0.0, 0.45)
    BOX_HALF   = (0.15, 0.15, 0.15)

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

    # Visual-only box (collision=False) so Genesis's rigid-body solver does
    # NOT prevent the robot from passing through. The only thing keeping the
    # robot out of the box is the CBF — that's the comparison we want.
    box = scene.add_entity(
        gs.morphs.Box(
            size=tuple(2 * h for h in BOX_HALF),
            pos=BOX_CENTER,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(0.85, 0.15, 0.15), roughness=0.6),
    )

    panda = scene.add_entity(gs.morphs.URDF(file=str(URDF_PATH), fixed=True))
    cam = scene.add_camera(
        res=(1280, 720), pos=(1.4, -1.4, 1.2),
        lookat=(0.35, 0.0, 0.45), fov=60, GUI=False,
    )
    scene.build()

    q_init = np.array([0, -np.pi/3, 0, -5*np.pi/6, 0, np.pi/2, 0])
    panda.set_dofs_position(torch.tensor(q_init, dtype=torch.float32, device=gs.device))
    panda.set_dofs_velocity(torch.zeros(7, dtype=torch.float32, device=gs.device))
    panda.set_dofs_kp([400.0] * 7)
    panda.set_dofs_kv([40.0] * 7)
    for _ in range(20):
        panda.control_dofs_velocity(torch.zeros(7, dtype=torch.float32, device=gs.device))
        scene.step()

    # ── OSCBF setup ─────────────────────────────────────────────────
    robot = load_panda()
    config = BoxObstacleConfig(robot, BOX_CENTER, BOX_HALF)
    cbf = CBF.from_config(config)
    K_col = robot.link_collision_data(jnp.asarray(q_init)).shape[0]
    print(f"[boot] CBF: {2*robot.num_joints} joint barriers + "
           f"{K_col} sphere-vs-box barriers = "
           f"{2*robot.num_joints + K_col} total")
    _ = cbf.safety_filter(q_init.astype(np.float64), np.zeros(7, dtype=np.float64))

    # ── Drive loop ──────────────────────────────────────────────────
    frames, t = [], 0.0
    n_active = 0
    min_clearance_log = []
    SIM_DT = 0.01

    for k in range(args.steps):
        q = panda.get_dofs_position().cpu().numpy().astype(np.float64)
        qdot_des = fake_rl_policy(t, q).astype(np.float64)

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

        # Diagnostic: minimum clearance from any robot sphere to the box
        col = np.asarray(robot.link_collision_data(jnp.asarray(q)))
        delta = col[:, :3] - np.asarray(BOX_CENTER)
        clamped = np.clip(delta, -np.asarray(BOX_HALF), np.asarray(BOX_HALF))
        closest = np.asarray(BOX_CENTER) + clamped
        dists = np.linalg.norm(col[:, :3] - closest, axis=-1) - col[:, 3]
        min_clearance_log.append(float(dists.min()))

        if k % 4 == 0:
            frames.append(_grab(cam))

        if k % 100 == 0:
            ee = np.asarray(robot.ee_position(jnp.asarray(q)))
            print(f"  step {k:4d}  t={t:5.2f}s  EE=[{ee[0]:+.2f},{ee[1]:+.2f},{ee[2]:+.2f}]  "
                   f"min clearance to box={min_clearance_log[-1]:+.3f} m  "
                   f"{'(filt off)' if args.no_filter else 'filt_active' if n_active else ''}")

    iio.imwrite(args.out, np.stack(frames), fps=25, codec="libx264")
    print(f"\n[done] wrote {args.out} ({len(frames)} frames)")
    min_clr = min(min_clearance_log)
    print(f"        min clearance to box (over whole run): {min_clr:+.3f} m  "
           f"(negative = penetration)")
    n_violations = sum(1 for c in min_clearance_log if c < 0)
    print(f"        collision frames: {n_violations}/{args.steps}")
    if not args.no_filter:
        print(f"        safety filter active on {n_active}/{args.steps} steps "
               f"({100*n_active/args.steps:.0f}%)")


if __name__ == "__main__":
    main()
