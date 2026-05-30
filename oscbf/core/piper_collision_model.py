"""Sphere collision model for the AgileX Piper.

Positions are in each link's kinematic frame (NOT COM frame), in metres.
Auto-fitted via fit_piper_spheres.py from the URDF's STL meshes. Link 6's
spheres cover the fixed-merged body: link6 + gripper_base + link7 + link8.

Total spheres: 12 (1 + 3 + 3 + 1 + 2 + 2).
"""

link1_pos = (
    (+0.0001, +0.0020, -0.0065),
)
link1_radii = (0.036,)

link2_pos = (
    (+0.0170, -0.0165, +0.0000),
    (+0.1425, -0.0165, +0.0000),
    (+0.2680, -0.0165, +0.0000),
)
link2_radii = (0.049, 0.049, 0.049)

link3_pos = (
    (-0.0131, -0.1717, +0.0004),
    (-0.0131, -0.0924, +0.0004),
    (-0.0131, -0.0131, +0.0004),
)
link3_radii = (0.043, 0.043, 0.043)

link4_pos = (
    (-0.0015, +0.0020, -0.0038),
)
link4_radii = (0.031,)

link5_pos = (
    (+0.0000, -0.0595, -0.0020),
    (+0.0000, -0.0070, -0.0020),
)
link5_radii = (0.030, 0.030)

link6_pos = (
    (-0.0047, +0.0000, +0.0685),
    (-0.0047, +0.0000, +0.0833),
)
link6_radii = (0.072, 0.072)


piper_collision_data = {
    "positions": (link1_pos, link2_pos, link3_pos, link4_pos, link5_pos, link6_pos),
    "radii":     (link1_radii, link2_radii, link3_radii, link4_radii, link5_radii, link6_radii),
}
