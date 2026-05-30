# Genesis-based OSCBF examples

These examples run OSCBF safety filters on top of the
[Genesis](https://github.com/Genesis-Embodied-AI/Genesis) simulator (instead of
the PyBullet examples shipped alongside this folder).

Genesis (PyTorch + Taichi on GPU) handles physics + rendering; OSCBF (JAX on
CPU) runs the safety-filter QP inside the same Python process. Each sim step
the loop is:

```
q ← genesis.get_dofs_position()
qdot_des ← policy(q, t)
qdot_safe ← cbf.safety_filter(q, qdot_des)
genesis.control_dofs_velocity(qdot_safe)
genesis.step()
```

## Setup

```bash
# In your OSCBF environment
pip install genesis-world trimesh imageio
```

## Examples

| Script | Robot | What it shows |
|---|---|---|
| `franka_lite_cbf.py` | Franka Panda | Joint-limit + EE-workspace-box CBF |
| `franka_box_cbf.py` | Franka Panda | Sphere-vs-box obstacle CBF |
| `franka_dual_arm.py` | 2× Franka Panda | Joint + arm-vs-arm + box-obstacle CBFs |
| `piper_lite_cbf.py` | AgileX Piper | Joint-limit + EE-workspace-box CBF |
| `piper_box_cbf.py` | AgileX Piper | Sphere-vs-box obstacle CBF |
| `piper_dual_arm.py` | 2× AgileX Piper | Full CBF (joint + arm-vs-arm + dynamic-box obstacle) |

Each demo accepts `--no_filter` to bypass the safety filter for a side-by-side
comparison, and writes an `.mp4` to `~/safety_filter/` by default (override
with `--out path.mp4`).

## Tools

| Script | Purpose |
|---|---|
| `tools/piper_load_smoketest.py` | Loads Piper, sweeps every joint through its limits, renders MP4 |
| `tools/fit_piper_spheres.py` | Auto-fits collision spheres from the Piper STL meshes (used to generate `oscbf/core/piper_collision_model.py`) |
| `tools/visualize_piper_spheres.py` | Renders Piper with all collision spheres overlaid as translucent green spheres |
