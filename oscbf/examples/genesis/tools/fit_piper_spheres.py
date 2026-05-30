"""Auto-fit Piper collision spheres per Manipulator link, accounting for the
fixed-merged gripper (gripper_base + link7 + link8 → link6 frame).

Outputs ready-to-paste Python tuples for piper_collision_model.py.

Sphere placement: along each combined-body's longest aabb axis, with K=1/2/3
spheres depending on extent. Radius = half the larger of the two cross-axis
extents (conservative cover).
"""

from __future__ import annotations

import numpy as np
import trimesh

from oscbf.assets import ASSETS_DIR


MESH_DIR = ASSETS_DIR / "piper" / "meshes"


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _xform(xyz, rpy) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rpy(*rpy)
    T[:3, 3] = xyz
    return T


# Per-link config: a list of meshes (each with its transform into the link's frame).
# Format: link_name -> [(mesh_filename, T_link_from_mesh), ...]
# For link6: also fold in gripper_base + link7 + link8 (fixed-merged in URDF).
LINK_DEFS = {
    "link1": [("link1.STL", np.eye(4))],
    "link2": [("link2.STL", np.eye(4))],
    "link3": [("link3.STL", np.eye(4))],
    "link4": [("link4.STL", np.eye(4))],
    "link5": [("link5.STL", np.eye(4))],
    "link6": [
        ("link6.STL",        np.eye(4)),
        # joint6_to_gripper_base: xyz=0 rpy=0 → gripper_base is at link6 origin
        ("gripper_base.STL", np.eye(4)),
        # joint7: xyz=(0,0,0.1558) rpy=(1.5708,0,0) → link7 frame in link6 frame
        ("link7.STL",        _xform((0.0, 0.0, 0.1558), (1.5708, 0.0, 0.0))),
        # joint8: xyz=(0,0,0.1158) rpy=(1.5708,0,-3.1416) → link8 in link6 frame
        ("link8.STL",        _xform((0.0, 0.0, 0.1158), (1.5708, 0.0, -3.1416))),
    ],
}


def fit_axis_spheres(verts: np.ndarray, n_max: int = 3):
    """Cover the point cloud with K spheres along its longest aabb axis."""
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    extents = bb_max - bb_min
    long_axis = int(np.argmax(extents))
    others = [i for i in (0, 1, 2) if i != long_axis]
    radius = 0.5 * max(extents[others[0]], extents[others[1]])

    L = extents[long_axis]
    if L < 0.08:
        K = 1
    elif L < 0.20:
        K = 2
    else:
        K = min(n_max, 3)

    centers = []
    if K == 1:
        c = (bb_min + bb_max) / 2.0
        centers.append(c)
    else:
        ts = np.linspace(bb_min[long_axis] + radius,
                         bb_max[long_axis] - radius, K)
        for tval in ts:
            c = (bb_min + bb_max) / 2.0
            c[long_axis] = tval
            centers.append(c)
    return centers, radius, long_axis, extents


def main():
    print(f"{'link':10s}  {'#mesh':>5s}  {'#sph':>4s}  {'r(m)':>6s}  {'long':>5s}  extents (m)")
    print("-" * 70)
    results = {}
    for link, mesh_specs in LINK_DEFS.items():
        all_verts = []
        for fname, T in mesh_specs:
            mesh = trimesh.load_mesh(str(MESH_DIR / fname), force="mesh")
            v = mesh.vertices
            v_h = np.hstack([v, np.ones((v.shape[0], 1))])
            v_link = (T @ v_h.T).T[:, :3]
            all_verts.append(v_link)
        verts = np.vstack(all_verts)
        centers, radius, long_axis, extents = fit_axis_spheres(verts)
        axes = ["x", "y", "z"]
        print(f"{link:10s}  {len(mesh_specs):>5d}  {len(centers):>4d}  {radius:>6.3f}  "
              f"{axes[long_axis]:>5s}  {np.round(extents, 3).tolist()}")
        results[link] = (centers, radius)

    print("\n# ─── Paste into piper_collision_model.py ─────────────")
    for link, (centers, radius) in results.items():
        print(f"{link}_pos = (")
        for c in centers:
            print(f"    ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}),")
        print(f")")
        print(f"{link}_radii = (" + ", ".join([f"{radius:.3f}"] * len(centers)) + ",)")
        print()


if __name__ == "__main__":
    main()
