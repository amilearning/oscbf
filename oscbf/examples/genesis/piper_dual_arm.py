"""Genesis + OSCBF: dual-arm Piper with joint-limit + arm-vs-box + arm-vs-arm CBFs.

Two Pipers, 30 cm apart (y = ±0.15 m), facing +x. A 40 cm cube box sits
0.4 m behind them as an obstacle. Random policies on both arms.

Barriers:
  - 24 joint-angle limits (2 × 6 × 2 arms)
  - 24 sphere-vs-box (12 spheres × 2 arms × 1 box)
  - 144 inter-arm sphere↔sphere (12 × 12)
  = 192 total

Run:
    conda activate oscbf
    python genesis_dual_piper_cbf.py            # filter on
    python genesis_dual_piper_cbf.py --no_filter
"""

from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp
import imageio.v3 as iio
import torch

import genesis as gs

from cbfpy import CBF, CBFConfig

from oscbf.core.manipulator import load_piper
from oscbf.assets import ASSETS_DIR
PIPER_URDF = ASSETS_DIR / "piper" / "piper.urdf"


# Per-arm base offsets in world frame (30 cm apart along y)
OFFSET_LEFT  = (0.0, -0.15, 0.0)
OFFSET_RIGHT = (0.0, +0.15, 0.0)
# Box obstacle behind the arms: 40 cm cube centered 0.40 m behind bases
BOX_OBSTACLES = [
    ((-0.30, 0.00, 0.20), (0.20, 0.20, 0.20)),    # 30 cm behind bases (was 40)
]
INTER_ARM_SAFETY_MARGIN = 0.02

N_DOF = 6   # Piper


# ── Dual-arm CBF config ──────────────────────────────────────────────
@jax.tree_util.register_static
class DualPiperConfig(CBFConfig):
    """12-DOF state (q_left, q_right). Control u = (q̇_left, q̇_right).

    QP cost is identity-weighted: min ‖u − u_des‖².
    """

    def __init__(self, left_robot, right_robot, base_offset_left,
                 base_offset_right, box_obstacles, box_margin=0.02):
        self.left = left_robot
        self.right = right_robot
        self.bL = np.asarray(base_offset_left,  dtype=np.float64)
        self.bR = np.asarray(base_offset_right, dtype=np.float64)
        centers = np.asarray([c for c, _ in box_obstacles], dtype=np.float64)
        # Inflate box internally by box_margin so the actual arm stays clear
        # of the visualized box despite Genesis velocity-PD tracking lag.
        halfs   = np.asarray([h for _, h in box_obstacles], dtype=np.float64) + box_margin
        self.box_centers = tuple(map(tuple, centers))
        self.box_halfs   = tuple(map(tuple, halfs))
        u_max = np.concatenate([np.asarray(left_robot.joint_max_velocities),
                                np.asarray(right_robot.joint_max_velocities)])
        super().__init__(n=2 * N_DOF, m=2 * N_DOF, u_min=-u_max, u_max=+u_max)

    def f(self, z, *args, **kwargs):
        return jnp.zeros(self.n)

    def g(self, z, *args, **kwargs):
        return jnp.eye(self.n)

    def h_1(self, z, **kwargs):
        qL, qR = z[:N_DOF], z[N_DOF:]
        joint_h = jnp.concatenate([
            qL - jnp.asarray(self.left.joint_lower_limits),
            jnp.asarray(self.left.joint_upper_limits)  - qL,
            qR - jnp.asarray(self.right.joint_lower_limits),
            jnp.asarray(self.right.joint_upper_limits) - qR,
        ])
        colL = self.left.link_collision_data(qL)          # (K, 4)
        colR = self.right.link_collision_data(qR)
        pL = colL[:, :3] + jnp.asarray(self.bL)
        pR = colR[:, :3] + jnp.asarray(self.bR)
        rL = colL[:, 3]
        rR = colR[:, 3]

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

        d = jnp.linalg.norm(pL[:, None, :] - pR[None, :, :], axis=-1)
        inter_h = (d - (rL[:, None] + rR[None, :])
                      - INTER_ARM_SAFETY_MARGIN).reshape(-1)
        return jnp.concatenate([joint_h, boxes_h, inter_h])

    def alpha(self, h):
        return 15.0 * h

    def P(self, z, u_des, *args, **kwargs):
        return jnp.eye(self.n)

    def q(self, z, u_des, *args, **kwargs):
        return -u_des


# ── Random policies: held flat between resamples ────────────────────
_RNG_LEFT  = np.random.default_rng(1)
_RNG_RIGHT = np.random.default_rng(2)
_RANDOM_AMP        = 2.5         # rad/s peak (more aggressive — invites contact)
_RANDOM_RESAMPLE_S = 1.0
_qdot_left_cache  = {"t_next": 0.0, "val": np.zeros(N_DOF)}
_qdot_right_cache = {"t_next": 0.0, "val": np.zeros(N_DOF)}

def fake_policy_left(t: float, q: np.ndarray) -> np.ndarray:
    if t >= _qdot_left_cache["t_next"]:
        _qdot_left_cache["val"]    = _RNG_LEFT.uniform(-_RANDOM_AMP, +_RANDOM_AMP, size=N_DOF)
        _qdot_left_cache["t_next"] = t + _RANDOM_RESAMPLE_S
    return _qdot_left_cache["val"]

def fake_policy_right(t: float, q: np.ndarray) -> np.ndarray:
    if t >= _qdot_right_cache["t_next"]:
        _qdot_right_cache["val"]    = _RNG_RIGHT.uniform(-_RANDOM_AMP, +_RANDOM_AMP, size=N_DOF)
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
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--no_filter", action="store_true")
    ap.add_argument("--out", default="/home/erl/safety_filter/dual_piper_cbf.mp4")
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

    # Dynamic, 50 g, collidable box. If the arms touch it, it gets shoved —
    # a movable sentinel for whether the CBF actually keeps the arms clear.
    # For a 0.4 × 0.4 × 0.4 m cube, rho = 0.05 kg / 0.064 m³ ≈ 0.78 kg/m³
    box_entities = []
    for (center, half) in BOX_OBSTACLES:
        volume = float(np.prod([2 * h for h in half]))
        rho = 0.050 / volume                    # ≈ 0.78 kg/m³ for the 40 cm cube
        box_entities.append(scene.add_entity(
            gs.morphs.Box(size=tuple(2 * h for h in half), pos=center,
                          fixed=False, collision=True),
            material=gs.materials.Rigid(rho=rho, friction=0.6),
            surface=gs.surfaces.Default(color=(0.85, 0.15, 0.15), roughness=0.6),
        ))

    piper_L = scene.add_entity(gs.morphs.URDF(
        file=str(PIPER_URDF), fixed=True, pos=OFFSET_LEFT))
    piper_R = scene.add_entity(gs.morphs.URDF(
        file=str(PIPER_URDF), fixed=True, pos=OFFSET_RIGHT))

    cam = scene.add_camera(res=(1280, 720), pos=(1.6, -1.8, 1.3),
                           lookat=(0.0, 0.0, 0.4), fov=58, GUI=False)
    scene.build()

    q_init = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0])
    q_init_t = torch.tensor(q_init, dtype=torch.float32, device=gs.device)
    for p in [piper_L, piper_R]:
        p.set_dofs_position(q_init_t)
        p.set_dofs_velocity(torch.zeros(N_DOF, dtype=torch.float32, device=gs.device))
        p.set_dofs_kp([300.0] * N_DOF)
        p.set_dofs_kv([30.0] * N_DOF)
    for _ in range(20):
        for p in [piper_L, piper_R]:
            p.control_dofs_velocity(torch.zeros(N_DOF, dtype=torch.float32, device=gs.device))
        scene.step()

    # ── OSCBF dual-arm ──────────────────────────────────────────────
    left = load_piper(with_collision=True)
    right = load_piper(with_collision=True)
    config = DualPiperConfig(left, right, OFFSET_LEFT, OFFSET_RIGHT, BOX_OBSTACLES)
    cbf = CBF.from_config(config)
    K_L = left.link_collision_data(jnp.asarray(q_init)).shape[0]
    K_R = right.link_collision_data(jnp.asarray(q_init)).shape[0]
    n_boxes = len(BOX_OBSTACLES)
    n_joint = 4 * N_DOF
    n_box   = n_boxes * (K_L + K_R)
    n_inter = K_L * K_R
    print(f"[boot] DualPiperCBF: {n_joint} joint + {n_box} arm↔box ({n_boxes} box) + "
          f"{n_inter} inter-arm = {n_joint + n_box + n_inter} barriers")
    z0 = np.concatenate([q_init, q_init]).astype(np.float64)
    _ = cbf.safety_filter(z0, np.zeros(2 * N_DOF, dtype=np.float64))

    # ── Drive loop ──────────────────────────────────────────────────
    frames, t = [], 0.0
    n_active = 0
    min_clr_box_log, min_clr_inter_log = [], []
    box_disp_log = []
    initial_box_pos = np.asarray(BOX_OBSTACLES[0][0])
    SIM_DT = 0.01

    for k in range(args.steps):
        qL = piper_L.get_dofs_position().cpu().numpy().astype(np.float64)
        qR = piper_R.get_dofs_position().cpu().numpy().astype(np.float64)
        z = np.concatenate([qL, qR])
        u_des = np.concatenate([fake_policy_left(t, qL),
                                fake_policy_right(t, qR)]).astype(np.float64)

        if args.no_filter:
            u_cmd = u_des
        else:
            u_safe = np.asarray(cbf.safety_filter(z, u_des))
            u_cmd = u_safe
            if np.linalg.norm(u_safe - u_des) > 1e-3:
                n_active += 1

        piper_L.control_dofs_velocity(torch.tensor(u_cmd[:N_DOF],  dtype=torch.float32, device=gs.device))
        piper_R.control_dofs_velocity(torch.tensor(u_cmd[N_DOF:], dtype=torch.float32, device=gs.device))
        scene.step()
        t += SIM_DT

        # Diagnostics
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
        d_inter = np.linalg.norm(colL[:, None, :3] - colR[None, :, :3], axis=-1) \
                  - (colL[:, 3, None] + colR[None, :, 3])
        min_clr_inter_log.append(float(d_inter.min()))

        # Track box displacement from start
        box_pos = box_entities[0].get_pos().cpu().numpy()
        if box_pos.ndim == 2: box_pos = box_pos[0]
        box_disp_log.append(float(np.linalg.norm(box_pos - initial_box_pos)))

        if k % 4 == 0:
            frames.append(_grab(cam))
        if k % 100 == 0:
            tag = "(filt off)" if args.no_filter else f"filt {n_active}/{k+1}"
            print(f"  step {k:4d}  t={t:5.2f}s  "
                  f"min_box={min_clr_box_log[-1]:+.3f}m  min_inter={min_clr_inter_log[-1]:+.3f}m  "
                  f"box_disp={box_disp_log[-1]:.3f}m  {tag}")

    iio.imwrite(args.out, np.stack(frames), fps=25, codec="libx264")
    print(f"\n[done] wrote {args.out} ({len(frames)} frames)")
    print(f"        min clearance to box     : {min(min_clr_box_log):+.3f} m")
    print(f"        min clearance inter-arm  : {min(min_clr_inter_log):+.3f} m")
    print(f"        box-collision frames     : {sum(1 for c in min_clr_box_log if c < 0)}/{args.steps}")
    print(f"        inter-arm collision frames: {sum(1 for c in min_clr_inter_log if c < 0)}/{args.steps}")
    print(f"        max box displacement     : {max(box_disp_log):.3f} m  "
          f"(final: {box_disp_log[-1]:.3f} m)")
    if not args.no_filter:
        print(f"        filter active            : {n_active}/{args.steps} steps "
              f"({100 * n_active / args.steps:.0f}%)")


if __name__ == "__main__":
    main()
