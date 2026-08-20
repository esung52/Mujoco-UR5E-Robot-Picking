
# UR5e Conveyor Pick-and-Place Simulation

This project implements an autonomous pick-and-place system for a UR5e robotic arm operating on a moving conveyor belt. It integrates perception, planning, and control modules to detect, track, and grasp objects of varying sizes and orientations.

## Features

- Moving conveyor with configurable speed and direction
- Free-joint objects with variable size, colour, and yaw
- Overhead camera: depth → height-above-belt, colour-independent segmentation
- Optional YOLO (Ultralytics) for detection experiments
- Latency-aware position lock and constant-velocity tracking
- Grasp planning from height, length, width, aspect, yaw
- Intercept timing from arm travel estimate + belt velocity
- XY bias learning (EMA) and grasp XY offsets for TCP alignment
- Phase 4 multi-object evaluation with success metrics and failure tags
- Place → home cycle for a complete pick-and-place loop

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| Python | 3.10+ recommended |
| Git | Required to clone MuJoCo Menagerie |
| OS | Windows / Linux / macOS (paths below use Windows-style examples) |
| GPU (optional) | Speeds up YOLO if configured; CPU works |

## Installation

### 1. Clone or open this project

Open the project folder in PyCharm (or your IDE).

### 2. Install Python dependencies

```bash
pip install mujoco mujoco-python-viewer opencv-python ikpy glfw scipy numpy pillow PyOpenGL sympy
```

YOLO:

```bash
pip install ultralytics opencv-python
```

Install Git if needed, as required for Menagerie: https://git-scm.com

### 3. Download MuJoCo models

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

Models used in program:

- `mujoco_menagerie/universal_robots_ur5e/` — arm + scene
- `mujoco_menagerie/robotiq_2f85/` — gripper assets and model

## Path configuration

All absolute paths must be changed to match machine.

- **Main.py:** must update `UR5E_XML`, `xml_path` (scene)
- **scene.xml:** must update `<include file="...">` for `ur5e.xml` / `2f85.xml`; mesh paths if any
- **ur5e.xml:** must update `meshdir` / mesh `file=` if not relative
- **2f85.xml:** must update compiler `meshdir`, each `<mesh file="...">`

Example:

```python
UR5E_XML = r"C:\Users\<YOU>\...\mujoco_menagerie\universal_robots_ur5e\ur5e.xml"
xml_path = r"C:\Users\<YOU>\...\mujoco_menagerie\universal_robots_ur5e\scene.xml"
```

## Project structure

```text
YourProject/
├── Main.py                 # Simulation entry, vision, IK, Phase 4
├── README.md
└── (optional local copies of XML)

mujoco_menagerie/
├── universal_robots_ur5e/
│   ├── ur5e.xml
│   ├── scene.xml           # Your extended scene (belt, box, camera, gripper weld)
│   └── assets/
└── robotiq_2f85/
    ├── 2f85.xml
    └── assets/
```

## How the program works

### High-level pipeline

1. Reset arm to home, open gripper, spawn or start the box on the belt.
2. Phase 1 — perception: one-shot (or re-lock) overhead RGB + depth → object centre, height, L/W, yaw.
3. Track: predict position with known belt velocity: `p(t) = p_lock + v_belt * (t - t_lock)`.
4. Wait until a predicted intercept near mid-belt is reachable in time.
5. Phase 2 — plan: map features → approach/grasp/lift Z, gripper width, wrist yaw strategy.
6. Motion: approach above the line → tracking descend → polish XY → close → hold → lift → place → home.
7. Phase 4: repeat for a suite of sizes/colours with metrics.

### Conveyor

- Belt geom uses `surfacevel` set in Python: `model.geom_surfacevel[belt] = [direction * belt_speed, 0, 0, 0, 0, 0]`.
- Box is a free joint body that must collide with the belt (friction high enough to follow without slip).

### Control loop

- Simulation stepped with `mujoco.mj_step` inside `step_sim`.
- Arm actuators receive joint position targets from IK or joint-space moves (`move_to_q`).

## Phases (Phase 1 / 2 / 4)

These names appear in logs (`[PHASE1 ...]`, `[PHASE2 plan]`, `[PHASE4 ...]`).

### Phase 1 — Perception lock

- Capture overhead RGB + depth.
- Convert depth to height above belt.
- Segment object by height (colour-independent).
- Geometry: contour → centre UV → world XY, median height, min-area rect (L, W, yaw).
- Quality gates reject noise / off-belt blobs.
- Optional latency compensation: shift XY by `v * t_measure`.
- Re-lock again just before “go pick” for a fresh centre.

### Phase 2 — Grasp planning

From Phase 1 features:

- `grasp_z` / `approach_z` / `lift_z` from `BELT_Z + height + offsets`
- Gripper opening from `min(L, W) + margin` → actuator ctrl
- Strategy (`top_square` / `top_long` / …) and wrist yaw suggestion
- Does not move the arm by itself; `safe_pick_box_moving` consumes the plan.

### Phase 4 — Evaluation suite

- Spawns objects from `PHASE4_SUITE` (sizes, colours).
- Fixed spawn X (e.g. `0.725`), direction toward mid-belt, optional random yaw.
- Resets vision between trials; restores nominal `XY_BIAS` so trials stay independent.
- Logs success, XY error, time, failure tags; prints a success table.
- Phase 4 should not rewrite core IK/bias logic—only scenario setup and metrics.

## Perception: OpenCV and YOLO

### OpenCV (primary)

Main path for reliable, colour-agnostic belt objects in this project.

1. Depth → height map: metric height without fixed `GRASP_Z` only
2. Height segmentation: works for any colour
3. Contours / min-area rect: centre, L, W, yaw in belt plane
4. Display: bounding box / track marker in the camera window

### YOLO

1. Generic object proposals: research comparison, labelled detections
2. Throttled (`YOLO_EVERY_N`): avoid slowing the sim

## Inverse kinematics, bias, TCP, offsets

### IK

- ikpy chain built from the UR5e MJCF (`Chain.from_mjcf_file`).
- Target is a 3D point in a frame consistent with the chain (world → base via `world_to_base`).
- Command pattern used in motion: `cmd = [-aim_x, -aim_y, z]` (sign convention matched to this scene’s base/TCP setup—do not change casually).

### Tool Center Point (TCP) / pinch

TCP: reference point on the tool used for positioning (in this project, the gripper pinch site). Arm motion is planned so that point reaches the aim pose.

- Site `pinch` on the gripper is the operational point (`get_pinch_pos()`).
- Success checks compare TCP to target or to the box.

### XY_BIAS

- Models systematic TCP − box offset in world XY.
- Updated with an EMA after selected settles (`update_xy_bias`).
- Used in `aim_from_box`: `aim_xy = predicted_box_xy - XY_BIAS + GRASP_XY_OFFSET` (+ optional size term).
- For multi-trial Phase 4, restore a nominal bias each trial so one bad approach sample does not poison the next object.

### GRASP_XY_OFFSET / plan offsets

- Small fixed XY tweak for gripper geometry.
- Vertical offsets (`GRASP_Z_OFFSET`, etc.) set approach/grasp/lift relative to measured height.

### Lead time

- Aim point includes belt motion over approach/descend time plus a small `extra_lead_time()`, so the hand arrives where the object will be, not where it was at lock.

## User-tunable variables

| Variable | Meaning | Typical use |
|----------|---------|-------------|
| `belt_speed` | Conveyor speed (m/s) | Lower if tracking is unreliable |
| `direction` | Belt direction (`+1` or `-1`) | Must match spawn side and intercept logic |
| `PHASE4_SPAWN_X` | Box start position along the belt | e.g. `0.725` so the box moves toward the middle |
| `PHASE4_BELT_SPEED` | Belt speed during evaluation | Keep fixed for comparable trials |
| `PHASE4_SUITE` | List of `(name, half-size, rgba)` objects | Add or remove test objects |
| `PHASE4_YAW_RANGE_DEG` | Random in-plane rotation range (degrees) | Stress-test grasp orientation |
| `PHASE4_TRIALS_PER_OBJ` | Number of trials per object class | Increase for more stable success rates |
| `BELT_Z` | Belt height reference used in vision/grasp math | Must match the belt top in the XML |
| `DETECT_X_MIN` / `DETECT_X_MAX` | X range where the camera “sees” the object | Match overhead camera coverage |
| `INTERCEPT_X_TARGET` | Target X for grasping | Usually near belt centre (`0`) |
| `XY_BIAS` | Learned offset between TCP and box in XY | Compensates systematic hand/calibration error |
| `XY_BIAS_NOMINAL` | Starting bias restored each Phase 4 trial | Stops one trial from spoiling the next |
| `GRASP_XY_OFFSET` | Fixed extra XY shift at grasp | Fine-tunes gripper alignment |
| `GRASP_Z_OFFSET` | Grasp height above measured object top | Raise if the tool digs into the object |
| `APPROACH_Z_OFFSET` | Approach height above the object | Clearance before the final descend |
| `LIFT_Z_OFFSET` | Lift height after grasp | Clear the belt after picking |
| `WRIST_YAW_OFFSET` | Constant added to planned wrist angle | Calibrate so long boxes are pinched on the short side |
| `WRIST_DURATION` | Time allowed for wrist rotation (s) | Keep short while the belt is moving |
| `HOME_Q` | Home joint configuration | Safe start/end pose |
| `PLACE_Q` | Place joint configuration | Drop-off pose after a successful pick |
| `SHOW_CAMERA_VIEW` | Show OpenCV overhead window | Disable to speed up long evaluation runs |

## Usage

### Single pick (default-style main)

```bash
python Main.py
```

Flow: home → wait for object in camera zone → intercept → pick → place → home.

### Phase 4 evaluation

Run with the Phase 4 main that loops `PHASE4_SUITE`, spawns each object, calls the same wait + `safe_pick_box_moving`, and prints a summary table.

### Viewer

- MuJoCo passive viewer for the 3D scene
- OpenCV window for the overhead camera (if enabled)
- Close both when the script exits (`cv2.destroyAllWindows()`; leave the viewer context)

## License / acknowledgements

- MuJoCo — DeepMind / Google
- MuJoCo Menagerie — [google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) (UR5e, Robotiq 2F-85 models; check each model’s license)
- ikpy, OpenCV, Ultralytics YOLO — respective open-source licenses
```

Only content fixes: `UR5E_XMl` → `UR5E_XML`, removed a stray quote after “belt plane”, and light markdown structure for headings/tables/code. Meaning is unchanged.
