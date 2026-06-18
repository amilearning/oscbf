"""Genesis + OSCBF: dual-Piper with a deliberately unsafe sinusoidal policy.

Counterpart to piper_safe_policy.py. Here the joint-space targets are large
amplitude sinusoids that constantly try to push past the joint limits, into
the rear box, and across into the partner arm. The safety filter must clip
the commands every step.

Result: filter materially intervenes on (close to) every step; the plot shows
the desired (blue) vs filtered (orange) commands diverging by full rad/s.

Run:
    conda activate oscbf
    python -m oscbf.examples.genesis.piper_dangerous_policy
    python -m oscbf.examples.genesis.piper_dangerous_policy --no_filter  # arms violently collide
"""

from __future__ import annotations

import argparse

import numpy as np
import jax
import jax.numpy as jnp
import imageio.v3 as iio
import matplotlib.pyplot as plt
import torch

import genesis as gs

from cbfpy import CBF, CBFConfig

from oscbf.core.manipulator import load_piper
from oscbf.assets import ASSETS_DIR


PIPER_URDF = ASSETS_DIR / "piper" / "piper.urdf"

OFFSET_LEFT  = (0.0, -0.15, 0.0)
OFFSET_RIGHT = (0.0, +0.15, 0.0)
BOX_OBSTACLES = [
    ((-0.30, 0.00, 0.20), (0.20, 0.20, 0.20)),
]
INTER_ARM_SAFETY_MARGIN = 0.02
N_DOF = 6

# ── Dangerous policy: amplitudes deliberately exceed joint limits ────
HOME = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0])

#  j1 lim [-2.62, 2.17]  → amp 4.0 pushes way past both bounds
#  j2 lim [0, 3.14]      → amp 3.0 around 1.5 pushes past 0 and 3.14
#  j3 lim [-2.97, 0]     → amp 3.0 around -1.5 pushes past both
#  j4 lim [-1.75, 1.75]  → amp 3.0
#  j5 lim [-1.22, 1.22]  → amp 2.5
#  j6 lim [-2.09, 2.09]  → amp 3.5
DANGER_AMP    = np.array([4.0, 3.0, 3.0, 3.0, 2.5, 3.5])
DANGER_FREQ_L = np.array([0.40, 0.30, 0.35, 0.50, 0.45, 0.40])
DANGER_FREQ_R = np.array([0.45, 0.32, 0.37, 0.47, 0.50, 0.42])
PHASE_L       = np.zeros(N_DOF)
PHASE_R       = np.array([2.0, 1.0, 2.5, 1.5, 2.0, 1.0])
K_TRACK       = 5.0
QDOT_LIMIT    = 5.0   # rad/s — same as Piper's joint max velocity


def dangerous_policy(t, q, phase, freq):
    omega = 2.0 * np.pi * freq
    q_target = HOME + DANGER_AMP * np.sin(omega * t + phase)
    qdot = K_TRACK * (q_target - q)
    return np.clip(qdot, -QDOT_LIMIT, +QDOT_LIMIT)


# ── Dual-arm CBF config (same as piper_dual_arm / safe_policy) ──────
@jax.tree_util.register_static
class DualPiperConfig(CBFConfig):
    def __init__(self, left, right, bL, bR, box_obstacles, box_margin=0.02):
        self.left = left
        self.right = right
        self.bL = np.asarray(bL, dtype=np.float64)
        self.bR = np.asarray(bR, dtype=np.float64)
        centers = np.asarray([c for c, _ in box_obstacles], dtype=np.float64)
        halfs   = np.asarray([h for _, h in box_obstacles], dtype=np.float64) + box_margin
        self.box_centers = tuple(map(tuple, centers))
        self.box_halfs   = tuple(map(tuple, halfs))
        u_max = np.concatenate([np.asarray(left.joint_max_velocities),
                                np.asarray(right.joint_max_velocities)])
        super().__init__(n=2 * N_DOF, m=2 * N_DOF, u_min=-u_max, u_max=+u_max)

    def f(self, z, *a, **k): return jnp.zeros(self.n)
    def g(self, z, *a, **k): return jnp.eye(self.n)
    def P(self, z, u, *a, **k): return jnp.eye(self.n)
    def q(self, z, u, *a, **k): return -u

    def h_1(self, z, **kwargs):
        qL, qR = z[:N_DOF], z[N_DOF:]
        joint_h = jnp.concatenate([
            qL - jnp.asarray(self.left.joint_lower_limits),
            jnp.asarray(self.left.joint_upper_limits)  - qL,
            qR - jnp.asarray(self.right.joint_lower_limits),
            jnp.asarray(self.right.joint_upper_limits) - qR,
        ])
        colL = self.left.link_collision_data(qL)
        colR = self.right.link_collision_data(qR)
        pL = colL[:, :3] + jnp.asarray(self.bL)
        pR = colR[:, :3] + jnp.asarray(self.bR)
        rL = colL[:, 3]; rR = colR[:, 3]
        def sphere_to_box(p, r, c, h):
            delta = p - c
            clamped = jnp.clip(delta, -h, h)
            closest = c + clamped
            return jnp.linalg.norm(p - closest, axis=-1) - r
        bxs = []
        for c, h in zip(self.box_centers, self.box_halfs):
            cj = jnp.asarray(c); hj = jnp.asarray(h)
            bxs.append(sphere_to_box(pL, rL, cj, hj))
            bxs.append(sphere_to_box(pR, rR, cj, hj))
        boxes_h = jnp.concatenate(bxs)
        d = jnp.linalg.norm(pL[:, None, :] - pR[None, :, :], axis=-1)
        inter_h = (d - (rL[:, None] + rR[None, :])
                      - INTER_ARM_SAFETY_MARGIN).reshape(-1)
        return jnp.concatenate([joint_h, boxes_h, inter_h])

    def alpha(self, h):
        return 15.0 * h


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
    ap.add_argument("--out", default="/home/erl/safety_filter/piper_dangerous_policy.mp4")
    ap.add_argument("--plot", default="/home/erl/safety_filter/piper_dangerous_policy_plot.png",
                    help="Path for the input-vs-filtered command plot (set to '' to skip)")
    args = ap.parse_args()

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

    for (center, half) in BOX_OBSTACLES:
        scene.add_entity(
            gs.morphs.Box(size=tuple(2 * h for h in half), pos=center,
                          fixed=True, collision=False),
            surface=gs.surfaces.Default(color=(0.85, 0.15, 0.15), roughness=0.6),
        )

    piper_L = scene.add_entity(gs.morphs.URDF(
        file=str(PIPER_URDF), fixed=True, pos=OFFSET_LEFT))
    piper_R = scene.add_entity(gs.morphs.URDF(
        file=str(PIPER_URDF), fixed=True, pos=OFFSET_RIGHT))

    cam = scene.add_camera(res=(1280, 720), pos=(1.6, -1.8, 1.3),
                           lookat=(0.0, 0.0, 0.4), fov=58, GUI=False)
    scene.build()

    q_init_t = torch.tensor(HOME, dtype=torch.float32, device=gs.device)
    for p in [piper_L, piper_R]:
        p.set_dofs_position(q_init_t)
        p.set_dofs_velocity(torch.zeros(N_DOF, dtype=torch.float32, device=gs.device))
        p.set_dofs_kp([300.0] * N_DOF)
        p.set_dofs_kv([30.0] * N_DOF)
    for _ in range(20):
        for p in [piper_L, piper_R]:
            p.control_dofs_velocity(torch.zeros(N_DOF, dtype=torch.float32, device=gs.device))
        scene.step()

    left = load_piper(with_collision=True)
    right = load_piper(with_collision=True)
    config = DualPiperConfig(left, right, OFFSET_LEFT, OFFSET_RIGHT, BOX_OBSTACLES)
    cbf = CBF.from_config(config)
    K_L = left.link_collision_data(jnp.asarray(HOME)).shape[0]
    K_R = right.link_collision_data(jnp.asarray(HOME)).shape[0]
    n_box = len(BOX_OBSTACLES) * (K_L + K_R)
    print(f"[boot] DualPiperCBF: {4*N_DOF} joint + {n_box} arm↔box + "
          f"{K_L*K_R} inter-arm = {4*N_DOF + n_box + K_L*K_R} barriers")
    z0 = np.concatenate([HOME, HOME]).astype(np.float64)
    _ = cbf.safety_filter(z0, np.zeros(2 * N_DOF, dtype=np.float64))

    frames, t = [], 0.0
    n_active = 0
    delta_log, u_des_log, u_safe_log, t_log = [], [], [], []
    SIM_DT = 0.01

    for k in range(args.steps):
        qL = piper_L.get_dofs_position().cpu().numpy().astype(np.float64)
        qR = piper_R.get_dofs_position().cpu().numpy().astype(np.float64)
        z = np.concatenate([qL, qR])
        uL = dangerous_policy(t, qL, PHASE_L, DANGER_FREQ_L)
        uR = dangerous_policy(t, qR, PHASE_R, DANGER_FREQ_R)
        u_des = np.concatenate([uL, uR]).astype(np.float64)

        if args.no_filter:
            u_cmd = u_des
            delta_log.append(0.0)
        else:
            u_safe = np.asarray(cbf.safety_filter(z, u_des))
            u_cmd  = u_safe
            delta  = np.linalg.norm(u_safe - u_des)
            delta_log.append(float(delta))
            if delta > 1e-2:
                n_active += 1

        piper_L.control_dofs_velocity(torch.tensor(u_cmd[:N_DOF],  dtype=torch.float32, device=gs.device))
        piper_R.control_dofs_velocity(torch.tensor(u_cmd[N_DOF:], dtype=torch.float32, device=gs.device))
        scene.step()
        u_des_log.append(u_des.copy())
        u_safe_log.append(u_cmd.copy())
        t_log.append(t)
        t += SIM_DT

        if k % 4 == 0:
            frames.append(_grab(cam))
        if k % 100 == 0:
            tag = "(filt off)" if args.no_filter else f"|Δu|={delta_log[-1]:.2f}  active {n_active}/{k+1}"
            print(f"  step {k:4d}  t={t:5.2f}s  {tag}")

    iio.imwrite(args.out, np.stack(frames), fps=25, codec="libx264")
    print(f"\n[done] wrote {args.out} ({len(frames)} frames)")
    if not args.no_filter:
        print(f"        material interventions (|Δu| > 1e-2 rad/s): {n_active}/{args.steps} "
              f"({100*n_active/args.steps:.1f}%)")
        print(f"        max  |qdot_safe - qdot_des|: {max(delta_log):.3f} rad/s")
        print(f"        mean |qdot_safe - qdot_des|: {np.mean(delta_log):.3f} rad/s")

    # ── Input-vs-filtered plot ─────────────────────────────────────
    if args.plot:
        t_arr  = np.asarray(t_log)
        u_des  = np.stack(u_des_log)
        u_safe = np.stack(u_safe_log)
        fig, axes = plt.subplots(N_DOF, 2, figsize=(13, 11), sharex=True)
        for j in range(N_DOF):
            for col, arm in enumerate(["LEFT arm", "RIGHT arm"]):
                ax = axes[j, col]
                idx = j if col == 0 else N_DOF + j
                ax.plot(t_arr, u_des[:,  idx], lw=1.4, color="C0",
                        label="desired (dangerous policy)")
                ax.plot(t_arr, u_safe[:, idx], lw=1.2, color="C1",
                        linestyle="--", label="filtered (safety_filter output)")
                ax.set_ylabel(f"j{j+1}  qdot\n(rad/s)", fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-6, 6)
                if j == 0:
                    ax.set_title(arm, fontsize=11)
                if j == N_DOF - 1:
                    ax.set_xlabel("time (s)")
                if j == 0 and col == 1:
                    ax.legend(loc="upper right", fontsize=8)
        if args.no_filter:
            title = "Dangerous policy, NO filter (commands sent unchanged → arms collide)"
        else:
            title = (f"Dangerous policy: filter clips/redirects commands every step  "
                     f"(max |Δu| = {max(delta_log):.2f} rad/s, "
                     f"{100*n_active/args.steps:.0f}% of steps materially modified)")
        fig.suptitle(title, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(args.plot, dpi=130)
        plt.close(fig)
        print(f"\n[plot] wrote {args.plot}")


if __name__ == "__main__":
    main()
