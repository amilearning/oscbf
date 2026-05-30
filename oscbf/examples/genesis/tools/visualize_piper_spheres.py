"""Visualize Piper's hand-authored sphere collision model in Genesis.

Loads Piper at a home pose, queries `robot.link_collision_data(q)` for the 12
sphere world positions (with the gripper merged into link6), and renders the
arm with semi-transparent green spheres overlaid on the meshes.

Renders both a still PNG and a short MP4 sweeping through a few joint configs
so we can see whether the spheres track the body well at multiple poses.

Run:
    conda activate oscbf
    python visualize_piper_spheres.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import jax.numpy as jnp
import imageio.v3 as iio

import genesis as gs

from oscbf.core.manipulator import load_piper
from oscbf.assets import ASSETS_DIR
PIPER_URDF = ASSETS_DIR / "piper" / "piper.urdf"


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
    robot = load_piper(with_collision=True)
    n = robot.num_joints
    print(f"[boot] Piper has {n} DOF, {sum(len(p) for p in robot.collision_positions)} spheres")

    # Test poses to sweep
    poses = [
        np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0]),       # home (forward)
        np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0]),       # extended up
        np.array([0.0, 2.5, -2.0, 0.0, 0.0, 0.0]),       # folded down
        np.array([1.2, 1.5, -1.5, 0.0, 0.0, 0.0]),       # rotated yaw
        np.array([0.0, 1.5, -1.5, 1.0, 1.0, 0.0]),       # wrist twist
    ]

    # Compute sphere world positions for each pose (off-Genesis, via OSCBF FK)
    sphere_sets = []
    for q in poses:
        col = np.asarray(robot.link_collision_data(jnp.asarray(q)))   # (12, 4)
        sphere_sets.append(col)

    # ── Genesis: one scene per pose to render a still per pose ─────
    out_dir = Path(__file__).parent
    frames = []

    for idx, (q, spheres) in enumerate(zip(poses, sphere_sets)):
        gs.init(backend=gs.gpu, precision="32", logging_level="warning")
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            vis_options=gs.options.VisOptions(
                background_color=(0.95, 0.95, 0.97),
                ambient_light=(0.8, 0.8, 0.8),
                shadow=False,
            ),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        piper = scene.add_entity(gs.morphs.URDF(file=str(PIPER_URDF), fixed=True))

        # Add a translucent green sphere per collision sphere (fixed in world)
        for sx, sy, sz, sr in spheres:
            scene.add_entity(
                gs.morphs.Sphere(
                    radius=float(sr),
                    pos=(float(sx), float(sy), float(sz)),
                    fixed=True, collision=False,
                ),
                surface=gs.surfaces.Default(
                    color=(0.0, 1.0, 0.0, 0.35),    # semi-transparent green
                    roughness=0.4,
                ),
            )

        cam = scene.add_camera(res=(1280, 720), pos=(1.0, -1.0, 0.85),
                                lookat=(0.25, 0.0, 0.35), fov=55, GUI=False)
        scene.build()

        q32 = q.astype(np.float32)
        piper.set_dofs_position(torch.tensor(q32, device=gs.device))
        piper.set_dofs_velocity(torch.zeros(n, device=gs.device))
        piper.set_dofs_kp([300.0] * n)
        piper.set_dofs_kv([30.0] * n)
        for _ in range(15):
            piper.control_dofs_position(torch.tensor(q32, device=gs.device))
            scene.step()

        # Capture multiple frames per pose for the MP4
        for _ in range(20):
            piper.control_dofs_position(torch.tensor(q32, device=gs.device))
            scene.step()
            frames.append(_grab(cam))

        # Save a still per pose
        still_path = out_dir / f"piper_spheres_pose{idx}.png"
        iio.imwrite(still_path, frames[-1])
        print(f"  [pose {idx}] q = {np.round(q, 2).tolist()}  →  {still_path.name}")

        gs.destroy()

    mp4 = out_dir / "piper_spheres.mp4"
    iio.imwrite(mp4, np.stack(frames), fps=20, codec="libx264")
    print(f"\n[done] wrote {mp4} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
