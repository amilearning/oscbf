# AgileX Piper assets

URDF and collision/visual meshes for the 6-DOF AgileX Piper arm.

## Files

- `piper.urdf` — 6 actuated joints + fixed-mounted two-finger gripper (joint7/joint8 set to `fixed` at a mid-open position so OSCBF treats them as part of link6's rigid body).
- `piper_no_gripper.urdf` — same arm without the gripper, for use cases that don't need fingers.
- `meshes/` — STL files (visual + collision) for `base_link`, `link1`–`link8`, `gripper_base`.

## Provenance

Adapted from [agilexrobotics/piper_ros](https://github.com/agilexrobotics/piper_ros) (`noetic` branch, `piper_description` package). Changes from upstream:

1. Mesh path prefix changed from `package://piper_description/meshes/` to `meshes/` (relative to URDF).
2. `joint7` and `joint8` (prismatic finger sliders) converted to `fixed` joints at midrange positions, since OSCBF requires that all non-chain joints be merged via `fixed` so kinematics/dynamics fold into the chain. The two-finger gripper is therefore visually present but not actuated.

License: see upstream piper_ros package metadata.
