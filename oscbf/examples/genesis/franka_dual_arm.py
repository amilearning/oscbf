"""Genesis + OSCBF: dual-arm Franka with joint-limit + inter-arm collision +
box-obstacle CBFs.

Two Pandas, each at y = ±0.5 m, facing +x. A red box sits between them.
A fake "policy" drives both arms toward the box (and each other). OSCBF
keeps the whole 14-DOF system out of the box AND apart from itself.

Run:
    conda activate oscbf
    python genesis_dual_franka_cbf.py            # filter on
    python genesis_dual_franka_cbf.py --no_filter
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

from cbfpy import CBF, CBFConfig
from oscbf.core.manipulator import load_panda
from oscbf.assets import ASSETS_DIR


URDF_PATH = ASSETS_DIR / "franka_panda" / "panda.urdf"

# Per-arm base offsets in world frame.
OFFSET_LEFT  = (0.0, -0.4, 0.0)
OFFSET_RIGHT = (0.0, +0.4, 0.0)
# List of (center, half_extents) for each box obstacle. The first is the
# small red box in front; the second is a tall wall behind the bases.
BOX_OBSTACLES = [
    ((+0.50, 0.00, 0.45), (0.10, 0.10, 0.10)),   # red box in front
    ((-0.35, 0.00, 0.60), (0.10, 0.90, 0.60)),   # wall behind both arms
]
INTER_ARM_SAFETY_MARGIN = 0.02   # extra clearance to require between arms


# ── Dual-arm CBF config ──────────────────────────────────────────────
@jax.tree_util.register_static
class DualArmConfig(CBFConfig):
    """14-DOF state (q_left, q_right). Control u = (q̇_left, q̇_right).

    Barriers (relative degree 1, all in h_1):
      - 28 joint-angle limits (2 × 7 per arm)
      - K_L × 1 left-arm sphere ↔ box distances
      - K_R × 1 right-arm sphere ↔ box distances
      - K_L × K_R left ↔ right sphere distances (inter-arm)

    The QP cost is identity-weighted: min ‖u − u_des‖². No task-space
    objective — caller is responsible for what u_des should be.
    """

    def __init__(self, left_robot, right_robot, base_offset_left,
                 base_offset_right, box_obstacles):
        """`box_obstacles`: iterable of (center, half_extents) tuples."""
        self.left = left_robot
        self.right = right_robot
        self.bL = np.asarray(base_offset_left,  dtype=np.float64)
        self.bR = np.asarray(base_offset_right, dtype=np.float64)
        centers = np.asarray([c for c, _ in box_obstacles], dtype=np.float64)
        halfs   = np.asarray([h for _, h in box_obstacles], dtype=np.float64)
        self.box_centers = tuple(map(tuple, centers))   # hashable for jax static
        self.box_halfs   = tuple(map(tuple, halfs))
        # Joint limits/velocities for both arms (concat in this order: L then R)
        u_max = np.concatenate([np.asarray(left_robot.joint_max_velocities),
                                  np.asarray(right_robot.joint_max_velocities)])
        super().__init__(n=14, m=14, u_min=-u_max, u_max=+u_max)

    # State-transition: u IS q̇, so ż = u
    def f(self, z, *args, **kwargs):
        return jnp.zeros(self.n)

    def g(self, z, *args, **kwargs):
        return jnp.eye(self.n)

    def h_1(self, z, **kwargs):
        qL, qR = z[:7], z[7:]
        # ── Joint angle limits (28)
        joint_h = jnp.concatenate([
            qL - jnp.asarray(self.left.joint_lower_limits),
            jnp.asarray(self.left.joint_upper_limits)  - qL,
            qR - jnp.asarray(self.right.joint_lower_limits),
            jnp.asarray(self.right.joint_upper_limits) - qR,
        ])
        # ── Collision spheres, shifted into world frame
        colL = self.left.link_collision_data(qL)          # (K_L, 4)  pos xyz + radius
        colR = self.right.link_collision_data(qR)         # (K_R, 4)
        pL = colL[:, :3] + jnp.asarray(self.bL)
        pR = colR[:, :3] + jnp.asarray(self.bR)
        rL = colL[:, 3]
        rR = colR[:, 3]
        # ── Sphere ↔ box (axis-aligned) distances for every box obstacle
        def sphere_to_box(p, r, box_c, box_h):
            delta   = p - box_c
            clamped = jnp.clip(delta, -box_h, box_h)
            closest = box_c + clamped
            return jnp.linalg.norm(p - closest, axis=-1) - r
        box_h_list = []
        for center, half in zip(self.box_centers, self.box_halfs):
            cj = jnp.asarray(center); hj = jnp.asarray(half)
            box_h_list.append(sphere_to_box(pL, rL, cj, hj))
            box_h_list.append(sphere_to_box(pR, rR, cj, hj))
        boxes_h = jnp.concatenate(box_h_list)
        # ── Pairwise inter-arm distances  ((K_L, K_R))
        d = jnp.linalg.norm(pL[:, None, :] - pR[None, :, :], axis=-1)
        inter_h = (d - (rL[:, None] + rR[None, :])
                      - INTER_ARM_SAFETY_MARGIN).reshape(-1)
        return jnp.concatenate([joint_h, boxes_h, inter_h])

    def alpha(self, h):
        return 5.0 * h

    # Identity-weighted QP cost: min ‖u − u_des‖²
    def P(self, z, u_des, *args, **kwargs):
        return jnp.eye(self.n)

    def q(self, z, u_des, *args, **kwargs):
        return -u_des


# ── Fake policies: random joint velocities per arm ──────────────────
# Each arm samples a fresh q̇ uniformly in [-AMP, +AMP]^7 every
# RESAMPLE_S seconds (held flat in between → no high-frequency noise).
_RNG_LEFT  = np.random.default_rng(1)
_RNG_RIGHT = np.random.default_rng(2)
_RANDOM_AMP        = 2.5        # rad/s peak (close to Panda's 2.17 joint vmax)
_RANDOM_RESAMPLE_S = 1.0        # hold each random command this many seconds
_qdot_left_cache  = {"t_next": 0.0, "val": np.zeros(7)}
_qdot_right_cache = {"t_next": 0.0, "val": np.zeros(7)}

def fake_policy_left(t: float, q: np.ndarray) -> np.ndarray:
    if t >= _qdot_left_cache["t_next"]:
        _qdot_left_cache["val"]    = _RNG_LEFT.uniform(-_RANDOM_AMP, +_RANDOM_AMP, size=7)
        _qdot_left_cache["t_next"] = t + _RANDOM_RESAMPLE_S
    return _qdot_left_cache["val"]

def fake_policy_right(t: float, q: np.ndarray) -> np.ndarray:
    if t >= _qdot_right_cache["t_next"]:
        _qdot_right_cache["val"]    = _RNG_RIGHT.uniform(-_RANDOM_AMP, +_RANDOM_AMP, size=7)
        _qdot_right_cache["t_next"] = t + _RANDOM_RESAMPLE_S
    return _qdot_right_cache["val"]


def _grab(cam):
    rgb = cam.render()[0]
    if isinstance(rgb, torch.Tensor): rgb = rgb.cpu().numpy()
    if rgb.ndim == 4: rgb = rgb[0]
    if rgb.dtype in (np.float32, np.float64):
        rgb = (rgb.clip(0, 1) * 255).astype(np.uint8)
    return rgb[..., :3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--out", default="/home/erl/safety_filter/dual_arm_cbf.mp4")
    args = ap.parse_args()

    # ── Genesis ─────────────────────────────────────────────────────
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        vis_options=gs.options.VisOptions(
            background_color=(0.9, 0.9, 0.95),
            ambient_light=(0.7, 0.7, 0.7), shadow=True,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())

    # Visual-only boxes (CBF is the only thing keeping the arms out of them)
    _COLORS = [(0.85, 0.15, 0.15), (0.30, 0.30, 0.85)]   # front=red, back=blue wall
    for (center, half), color in zip(BOX_OBSTACLES, _COLORS):
        scene.add_entity(
            gs.morphs.Box(size=tuple(2 * h for h in half), pos=center,
                            fixed=True, collision=False),
            surface=gs.surfaces.Default(color=color, roughness=0.6),
        )

    panda_L = scene.add_entity(gs.morphs.URDF(
        file=str(URDF_PATH), fixed=True, pos=OFFSET_LEFT))
    panda_R = scene.add_entity(gs.morphs.URDF(
        file=str(URDF_PATH), fixed=True, pos=OFFSET_RIGHT))

    cam = scene.add_camera(res=(1280, 720), pos=(1.8, -2.0, 1.5),
                            lookat=(0.4, 0.0, 0.5), fov=60, GUI=False)
    scene.build()

    q_init = np.array([0, -np.pi/3, 0, -5*np.pi/6, 0, np.pi/2, 0])
    q_init_t = torch.tensor(q_init, dtype=torch.float32, device=gs.device)
    for p in [panda_L, panda_R]:
        p.set_dofs_position(q_init_t)
        p.set_dofs_velocity(torch.zeros(7, dtype=torch.float32, device=gs.device))
        p.set_dofs_kp([400.0]*7); p.set_dofs_kv([40.0]*7)
    for _ in range(20):
        for p in [panda_L, panda_R]:
            p.control_dofs_velocity(torch.zeros(7, dtype=torch.float32, device=gs.device))
        scene.step()

    # ── OSCBF dual-arm ──────────────────────────────────────────────
    left = load_panda(); right = load_panda()
    config = DualArmConfig(left, right, OFFSET_LEFT, OFFSET_RIGHT,
                            BOX_OBSTACLES)
    cbf = CBF.from_config(config)
    K_L = left.link_collision_data(jnp.asarray(q_init)).shape[0]
    K_R = right.link_collision_data(jnp.asarray(q_init)).shape[0]
    n_boxes = len(BOX_OBSTACLES)
    print(f"[boot] DualArmCBF: 28 joint + {n_boxes*(K_L+K_R)} arm↔box ({n_boxes} boxes) + "
           f"{K_L*K_R} inter-arm = {28 + n_boxes*(K_L+K_R) + K_L*K_R} barriers")
    z0 = np.concatenate([q_init, q_init]).astype(np.float64)
    _ = cbf.safety_filter(z0, np.zeros(14, dtype=np.float64))

    # ── Drive loop ──────────────────────────────────────────────────
    frames, t = [], 0.0
    n_active = 0
    min_clr_box_log, min_clr_inter_log = [], []
    SIM_DT = 0.01

    for k in range(args.steps):
        qL = panda_L.get_dofs_position().cpu().numpy().astype(np.float64)
        qR = panda_R.get_dofs_position().cpu().numpy().astype(np.float64)
        z = np.concatenate([qL, qR])
        u_des = np.concatenate([fake_policy_left(t, qL), fake_policy_right(t, qR)]).astype(np.float64)

        if args.no_filter:
            u_cmd = u_des
        else:
            u_safe = np.asarray(cbf.safety_filter(z, u_des))
            u_cmd = u_safe
            if np.linalg.norm(u_safe - u_des) > 1e-3:
                n_active += 1

        panda_L.control_dofs_velocity(torch.tensor(u_cmd[:7],  dtype=torch.float32, device=gs.device))
        panda_R.control_dofs_velocity(torch.tensor(u_cmd[7:], dtype=torch.float32, device=gs.device))
        scene.step()
        t += SIM_DT

        # Diagnostics: closest box clearance (worst over all boxes) + inter-arm
        colL = np.array(left.link_collision_data(jnp.asarray(qL)));  colL[:, :3] += np.asarray(OFFSET_LEFT)
        colR = np.array(right.link_collision_data(jnp.asarray(qR))); colR[:, :3] += np.asarray(OFFSET_RIGHT)
        all_spheres = np.concatenate([colL, colR])
        d_box_min = np.inf
        for center, half in BOX_OBSTACLES:
            delta = all_spheres[:, :3] - np.asarray(center)
            clamped = np.clip(delta, -np.asarray(half), np.asarray(half))
            closest = np.asarray(center) + clamped
            d_this = np.linalg.norm(all_spheres[:, :3] - closest, axis=-1) - all_spheres[:, 3]
            d_box_min = min(d_box_min, float(d_this.min()))
        min_clr_box_log.append(d_box_min)
        # Inter-arm clearance
        d_inter = np.linalg.norm(colL[:, None, :3] - colR[None, :, :3], axis=-1) - (colL[:, 3, None] + colR[None, :, 3])
        min_clr_inter_log.append(float(d_inter.min()))

        if k % 4 == 0:
            frames.append(_grab(cam))
        if k % 100 == 0:
            print(f"  step {k:4d}  t={t:5.2f}s  "
                   f"min_box={min_clr_box_log[-1]:+.3f}m  min_inter={min_clr_inter_log[-1]:+.3f}m  "
                   f"{'(filt off)' if args.no_filter else 'filt_active' if n_active else ''}")

    iio.imwrite(args.out, np.stack(frames), fps=25, codec="libx264")
    print(f"\n[done] wrote {args.out} ({len(frames)} frames)")
    print(f"        min clearance to box   : {min(min_clr_box_log):+.3f} m")
    print(f"        min clearance inter-arm: {min(min_clr_inter_log):+.3f} m")
    print(f"        box collision frames   : {sum(1 for c in min_clr_box_log if c < 0)}/{args.steps}")
    print(f"        inter-arm collision frames: {sum(1 for c in min_clr_inter_log if c < 0)}/{args.steps}")
    if not args.no_filter:
        print(f"        safety filter active   : {n_active}/{args.steps} steps ({100*n_active/args.steps:.0f}%)")


if __name__ == "__main__":
    main()
