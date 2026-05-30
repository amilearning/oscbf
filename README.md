![oscbf_gif](https://github.com/user-attachments/assets/43b615a3-dcec-4fc4-bc7f-2ba674562323)

# Operational Space Control Barrier Functions

Code for *"Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions"* -- Daniel Morton and Marco Pavone

Accepted to IROS 2025, Hangzhou

[![Paper](http://img.shields.io/badge/arXiv-2503.06736-B31B1B.svg)](https://arxiv.org/abs/2503.06736)

## What is OSCBF?

This is a safe, high-performance, and easy-to-use controller / safety filter for robotic manipulators.

With OSCBF, you can...
- Operate at kilohertz speed even with over 400 active safety constraints
- Design safety constraints (barrier functions) easily via CBFpy
- Enforce safety on both torque-controlled and velocity-controlled robots
- Either apply a safety filter on top of your existing controller, or use our provided controller

In general, this will be especialy useful for enforcing safety during **teleoperation** or while executing **learned policies**

For more details and videos, check out the [project webpage](https://stanfordasl.github.io/oscbf/) as well as the [CBFpy documentation](https://danielpmorton.github.io/cbfpy/).


## How do I use this?

Check out the `examples/` folder for interactive demos in Pybullet! This is the best place to start

If you're applying this to a different robot, you'll need to provide a URDF -- we can parse the kinematics and dynamics from that. Some notes:
- I haven't written an MJCF parser yet, but it should be feasible
- All joints that shouldn't be controlled as part of the kinematic chain should be set to `fixed` (gripper joints, for instance). Since you probably want to still be able to use the gripper joints in simulation, the best way to handle this is to make a copy of the URDF: load the non-fixed one in sim, and parse the fixed one with OSCBF
- I manually defined the collision model for the Franka. There are probably better ways to parse or generate this data from meshes, but I haven't done it yet. For now, I'd recommend doing the same for your robot.

## FAQ

- "I already have a well-tuned operational space controller! I don't want to replace that"
  - You can use OSCBF as a safety filter on top of your existing controller!
- "I'm working with a mobile manipulator. Does this still work?"
  - Yes! We support prismatic and revolute joints, so adding a mobile base just adds 3DOF (PPR) to the beginning of the kinematic chain.
- "I'm controlling joint-space motions -- does this still apply?"
  - Depending on the task, you will likely want to modify the objective function. But, the robot dynamics and CBF formulation will still be useful!


## Installation

A virtual environment is optional, but highly recommended. For `pyenv` installation instructions, see [here](https://danielpmorton.github.io/cbfpy/pyenv).

```
git clone https://github.com/stanfordasl/oscbf
cd oscbf
pip install -e .
```

To also run the Genesis-simulator examples (see [Piper + Genesis examples](#piper--genesis-examples) below), install with the `genesis` extra:

```
pip install -e .[genesis]
```

This pulls in `genesis-world` and `imageio[ffmpeg]` on top of the core deps.

Note: This code will work with most versions of `jax`, but there seems to have been a CPU slowdown introduced in version `0.4.32`. To avoid this, I use version `0.4.30`, which is the version indicated in this repo's `pyproject.toml` as well. However, feel free to use any version you like.

This has been tested on Python 3.10 and 3.11, on Ubuntu 22.04.


## Piper + Genesis examples

This fork adds support for the 6-DOF **AgileX Piper** arm and a set of
demos that run on the **Genesis** simulator (in addition to OSCBF's existing
PyBullet examples).

- `oscbf.core.manipulator.load_piper()` mirrors `load_panda()` — returns a
  `Manipulator` with kinematics/dynamics from the bundled URDF
  (`oscbf/assets/piper/`) and a 12-sphere collision model (`oscbf/core/piper_collision_model.py`).
- `oscbf/examples/genesis/` contains six demos (single + dual arm, Piper +
  Franka) and three tools (sphere auto-fit, sphere visualization, joint sweep).

After `pip install -e .[genesis]`, reproduce the headline result:

```
python -m oscbf.examples.genesis.piper_dual_arm                # with CBF
python -m oscbf.examples.genesis.piper_dual_arm --no_filter    # without
```

Two AgileX Pipers facing forward, 30 cm apart, with a 40 cm 50 g obstacle box behind them; both arms run a random joint-velocity policy. The CBF
enforces 192 barriers (24 joint + 24 sphere-vs-box + 144 inter-arm).

| Metric | filter on | filter off |
|---|---|---|
| Box-collision frames | 0 / 1000 | 671 / 1000 |
| Inter-arm collision frames | 0 / 1000 | 189 / 1000 |
| Box displacement | 0.000 m | 0.323 m |

See `oscbf/examples/genesis/README.md` for the full list of demos.


## Documentation

See the CBFpy documentation, available at [this link](https://danielpmorton.github.io/cbfpy)


## Hardware / ROS2 

The code used to run the Franka Panda hardware experiments is available at [this link](https://github.com/StanfordASL/oscbf_hardware_ws)


## Citation
```
@inproceedings{morton2025oscbf,
  author={Morton, Daniel and Pavone, Marco},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions}, 
  year={2025},
  pages={187-194},
  doi={10.1109/IROS60139.2025.11246389}
}
```

