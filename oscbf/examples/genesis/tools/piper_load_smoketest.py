"""Quick sanity check: load AgileX Piper into Genesis and render a short clip.

Goal: confirm the URDF + STL meshes load, kinematics look right, no parser
errors. No safety filter yet — that's the next step.

Run:
    conda activate oscbf
    python genesis_piper_load.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import imageio.v3 as iio

import genesis as gs


PIPER_URDF = (
    ASSETS_DIR / "piper" / "piper_no_gripper.urdf"
).resolve()


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
    print(f"[boot] URDF: {PIPER_URDF}")
    assert PIPER_URDF.exists(), f"URDF not found at {PIPER_URDF}"

    gs.init(backend=gs.gpu, precision="32", logging_level="info")
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

    piper = scene.add_entity(gs.morphs.URDF(file=str(PIPER_URDF), fixed=True))

    cam = scene.add_camera(
        res=(1280, 720),
        pos=(1.1, -1.1, 0.9),
        lookat=(0.0, 0.0, 0.3),
        fov=55,
        GUI=False,
    )
    scene.build()

    n_dofs = piper.n_dofs
    print(f"[info] piper n_dofs = {n_dofs}")
    print(f"[info] dof joint names = {[j.name for j in piper.joints if j.n_dofs > 0]}")

    # Sweep each joint to confirm visuals + kinematics are sane.
    q_init = np.zeros(n_dofs, dtype=np.float32)
    piper.set_dofs_position(torch.tensor(q_init, device=gs.device))
    piper.set_dofs_velocity(torch.zeros(n_dofs, device=gs.device))
    piper.set_dofs_kp([150.0] * n_dofs)
    piper.set_dofs_kv([15.0] * n_dofs)

    # Settle
    for _ in range(20):
        piper.control_dofs_position(torch.tensor(q_init, device=gs.device))
        scene.step()

    frames = []
    n_steps = 400
    for k in range(n_steps):
        t = k * 0.01
        q_des = np.array([
            0.6 * np.sin(0.7 * t),       # joint1 yaw
            0.6 + 0.5 * np.sin(0.6 * t), # joint2 shoulder (within [0, 3.14])
            -1.0 + 0.6 * np.sin(0.5 * t),# joint3 elbow (within [-2.97, 0])
            0.8 * np.sin(0.9 * t),       # joint4
            0.5 * np.sin(0.8 * t),       # joint5
            0.7 * np.sin(1.0 * t),       # joint6
        ], dtype=np.float32)
        piper.control_dofs_position(torch.tensor(q_des, device=gs.device))
        scene.step()
        if k % 4 == 0:
            frames.append(_grab(cam))
        if k % 100 == 0:
            q_now = piper.get_dofs_position().cpu().numpy()
            print(f"  step {k:4d}  q = {np.round(q_now, 2)}")

    out = Path(__file__).parent / "piper_load.mp4"
    iio.imwrite(out, np.stack(frames), fps=25, codec="libx264")
    print(f"\n[done] wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
