# pip install mujoco, mujoco-python-viewer, opencv-python, ikpy, glfw, scipy, numpy, pillow, PyOpenGl, sympy
# YOLO: pip install ultralytics opencv-python

import mujoco
import mujoco.viewer
import time
import traceback
import cv2
import numpy as np
from ikpy.chain import Chain
from ultralytics import YOLO

UR5E_XML = 'C:/Users/u3009531/PycharmProjects/PythonProject/mujoco_menagerie/universal_robots_ur5e/ur5e.xml'
xml_path = 'C:/Users/u3009531/PycharmProjects/PythonProject/mujoco_menagerie/universal_robots_ur5e/scene.xml'
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

TCP_OFFSET = [0,0,0]
ACTIVE_MASK = [False, False, True, True, True, True, True, True, False]
arm_chain = Chain.from_mjcf_file(UR5E_XML, base_elements=['base'], last_link_vector=TCP_OFFSET, active_links_mask=ACTIVE_MASK, name="ur5e")
belt_geom_id = model.geom("belt_geom").id
box_geom_id   = model.geom("box_geom").id
wall_left_id  = model.geom("wall1").id
wall_right_id = model.geom("wall2").id
box_body_id = model.body("box_body").id
vel_sensor_id = model.sensor('box_velocity').id
gripper_act_id = model.actuator("fingers_actuator").id
pinch_site_id  = model.site("pinch").id
ARM_JOINTS = ["shoulder_pan_joint","shoulder_lift_joint","elbow_joint","wrist_1_joint","wrist_2_joint","wrist_3_joint"]
ARM_ACTUATOR_NAMES = ["shoulder_pan","shoulder_lift","elbow","wrist_1","wrist_2","wrist_3"]
arm_act_ids=[model.actuator(name).id for name in ARM_ACTUATOR_NAMES]
arm_dof_ids = [model.joint(n).dofadr[0] for n in ARM_JOINTS]

# Belt parameters
belt_speed = 0.2
direction  = 1.0           # +1 or -1
cooldown   = 0.0           # seconds remaining before next allowed reverse
COOLDOWN_T = 0.4           # prevent chatter

HOME_Q = np.array([-1.57, -1.2, 1.4, -1.7, -1.57, 0.0])
ABOVE_BELT_Q = np.array([1.57, -1.2, 1.4, -1.7, -1.57, 0.0])       # hovering above belt
PLACE_Q = np.array([-0.8, -1.3, 1.5, -1.7, -1.57, 0.0])             # place location

APPROACH_Z = 0.52  # high hover – clear of belt / walls
GRASP_Z    = 0.47   # just above box top
LIFT_Z     = 0.6   # lift height after grasp
GRASP_WAIT = 0.8               # seconds to wait after closing gripper

XY_BIAS = np.array([0.06, 0.12])   # TCP_xy - box_xy (world)
GRASP_XY_OFFSET = np.array([0.0, -0.045])

POS_TOL     = 0.012      # m – Cartesian “close enough”
JOINT_TOL   = 0.03       # rad – joint-space “close enough”
SETTLE_TIME = 0.35       # s – must stay within tol this long
MAX_SETTLE  = 5.0        # s – safety timeout
INTERCEPT_X_TARGET = 0.0          # want grasp near middle of belt
INTERCEPT_X_TOL    = 0.12         # accept ±12 cm around target
MIN_LEAD_TIME      = 0.7          # need at least this much time to move
MAX_LEAD_TIME      = 1.8

CAM_NAME = "overhead_cam"
cam_id = model.camera(CAM_NAME).id
assert cam_id >= 0
RENDER_H, RENDER_W = 480,640
renderer = mujoco.Renderer(model, height=RENDER_H, width=RENDER_W)
BELT_Z=0.16
DEPTH_BELT_MARGIN = 0.01
BOX_HALF_H = 0.05
INTERCEPT_X_TARGET = 0         # pick near middle of belt
INTERCEPT_X_TOL = 0.12
DETECT_X_MIN = 0.3              # camera watches this end (example +X side)
DETECT_X_MAX = 0.7
_vision_prev = None
USE_VISION = True
SHOW_CAMERA_VIEW = True
_vision_frame_counter = 0
SHOW_EVERY_N_STEPS = 5
USE_IMAGE_BELT_DEPTH = False
_last_object_height = BOX_HALF_H
HEIGHT_SEG_THRESH = 0.015   # 2 cm above belt = object
HEIGHT_SEG_MAX = 0.15
MIN_OBJ_AREA = 80          # pixels; lower for small objects (e.g. 40)
USE_HEIGHT_SEGMENT = True  # True = Step 2 primary
BELT_X_MIN, BELT_X_MAX = -0.65,0.65
BELT_Y_MIN, BELT_Y_MAX = 0.68,0.78
_last_features = None

# --- YOLO / tracking (replace existing YOLO_* block) ---
yolo_model = YOLO("yolo11n.pt")
YOLO_CONF     = 0.20
YOLO_IOU      = 0.45
YOLO_IMGSZ    = 320
_last_bbox    = None          # (x1,y1,x2,y2) for drawing
_track_xy     = None          # locked world XY
_track_t      = 0.0
_height_fixed = False         # depth measured only once
_last_uv      = None
YOLO_EVERY_N   = 8            # run YOLO only every N sense calls
SENSE_EVERY_N  = 4            # full sense_object only every N get_box_pos_vision calls
_yolo_counter  = 0
_sense_counter = 0
_last_yolo_det = None
_last_feats    = None

MIN_HEIGHT        = 0.010
MAX_HEIGHT        = 0.150
MIN_WIDTH_M       = 0.015
MAX_WIDTH_M       = 0.25
MAX_ASPECT        = 8.0

FINGER_CLEARANCE   = 0.012   # above object top at grasp (m)
APPROACH_STANDOFF  = 0.08    # approach above grasp_z
LIFT_CLEARANCE     = 0.12    # lift above object top
GRIP_MARGIN        = 0.025   # extra opening beyond min(L,W)
GRIP_MIN           = 0.02    # minimum command span (m) — tune to your gripper
GRIP_MAX           = 0.085   # max opening (Robotiq-like); clamp to actuator range
ASPECT_LONG        = 1.45    # above this → align to short axis
OPEN_WIDTH_M   = 0.085     # fingers fully open, metres between pads
CLOSED_WIDTH_M = 0.0
GRIP_CTRL_OPEN = 0.0
GRIP_CTRL_CLOSED = 255.0

GRASP_Z_OFFSET    = 0.25    # grasp_z  = BELT_Z + h + this
APPROACH_Z_OFFSET = 0.29    # approach = BELT_Z + h + this
LIFT_Z_OFFSET     = 0.35    # lift_z   = BELT_Z + h + this
GRASP_Y_RESIDUAL = -0.03
WRIST_APPLIED = False
WRIST_DURATION = 0.1

PHASE4_SUITE = [
    ("cube_m", (0.03, 0.03, 0.03), (0, 0.8, 0, 1)),
    ("long_s", (0.06, 0.02, 0.03), (1, 0.5, 0, 1)),
    ("long_m", (0.08, 0.025, 0.025), (0.8, 0, 0.8, 1)),
    ("cyl_approx", (0.03, 0.03, 0.04), (0, 0.7, 0.7, 1)),
    ("flat", (0.06, 0.03, 0.015), (1, 1, 0, 1)),
    ("cube_s",     (0.02, 0.02, 0.02), (1, 0, 0, 1)),
]

PHASE4_SPAWN_X = 0.65
PHASE4_SPAWN_Y = 0.70
PHASE4_BELT_SPEED = 0.20          # fixed; or use a range below
PHASE4_TRIALS_PER_OBJ = 2
PHASE4_TIMEOUT_WAIT = 80.0
PHASE4_YAW_RANGE_DEG = (-45,45)

PHASE4_MODE = False
PHASE4_ALLOW_REVERSE = False      # no wall reverse during eval
XY_BIAS_NOMINAL = np.array([0.06,0.12], dtype=float)



if model.neq > 0:
    data.eq_active[:] = 1

# Helpers:
def qpos_to_ikpy(q6: np.ndarray) -> np.ndarray:
    full=np.zeros(len(arm_chain.links))
    full[arm_chain.active_links_mask] = np.asarray(q6, dtype=float)
    return full
def ikpy_to_qpos(full: np.ndarray) -> np.ndarray:
    """Full IKPy vector -> 6 arm joints."""
    return np.asarray(full[arm_chain.active_links_mask], dtype=float)
def get_joint_qpos() -> np.ndarray:
    return np.array([data.joint(name).qpos[0] for name in ARM_JOINTS])
def set_arm_ctrl(q):
    """Send joint position commands to the 6 arm actuators."""
    for i, act_id in enumerate(arm_act_ids):
        data.ctrl[act_id] = float(q[i])
def wait_until_joints(viewer, q_target, tol=JOINT_TOL, settle=SETTLE_TIME, timeout=MAX_SETTLE):
    """
    Keep stepping until joint error < tol for `settle` seconds of sim time.
    Commands q_target the whole time so the arm finishes moving.
    """
    q_target = np.asarray(q_target, dtype=float)
    t_ok = 0.0
    t_all = 0.0
    while t_all < timeout:
        set_arm_ctrl(q_target)
        step_sim(viewer)
        t_all += model.opt.timestep

        err = np.linalg.norm(get_joint_qpos() - q_target)
        if err < tol:
            t_ok += model.opt.timestep
            if t_ok >= settle:
                viewer.sync()
                print(f"  Joints settled")
                return True
        else:
            t_ok = 0.0

    print(f"  WARNING: joints not settled (err={np.linalg.norm(get_joint_qpos() - q_target):.4f})")
    viewer.sync()
    return False
def stop_belt():
    """Set conveyor surface velocity to zero (box stops sliding)."""
    model.geom_surfacevel[belt_geom_id] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
def open_gripper():
    if gripper_act_id >= 0:
        data.ctrl[gripper_act_id] = 0.0
def close_gripper():
    data.ctrl[gripper_act_id] = 255.0
def get_box_pos():
    """Ground-truth position of the box."""
    return data.body("box_body").xpos.copy()
def get_pinch_pos():
    return data.site_xpos[pinch_site_id].copy()
def move_to_q(viewer, q_target, duration=1.0):
    """Interpolate in joint space, then wait until really there."""
    q_target = np.asarray(q_target, dtype=float)
    q_start = get_joint_qpos()
    n = max(1, int(duration / model.opt.timestep))
    for t in range(n):
        a = (t + 1) / n
        set_arm_ctrl((1 - a) * q_start + a * q_target)
        step_sim(viewer)
    wait_until_joints(viewer, q_target)
def set_belt_direction(dir_sign: float):
    model.geom_surfacevel[belt_geom_id] = [dir_sign * belt_speed, 0.0, 0.0, 0, 0, 0]
def box_hits_end_wall() -> bool:
    """True if the box is currently contacting either end wall."""
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if (g1 == box_geom_id and g2 in (wall_left_id, wall_right_id)) or \
           (g2 == box_geom_id and g1 in (wall_left_id, wall_right_id)):
            return True
    return False
def starting_box():
    # data.joint("box_joint").qpos[0] = np.random.uniform(0, 0.65)  # x  (centre of belt)
    data.joint("box_joint").qpos[0] = 10  # x  (centre of belt)
    data.joint("box_joint").qpos[1] = 10  # y  (belt centre – match your XML)
    data.joint("box_joint").qpos[2] = 0.2  # z  (on top of belt)
    data.joint("box_joint").qpos[3] = 1.0  # qw
    data.joint("box_joint").qpos[4] = 0.0  # qx
    data.joint("box_joint").qpos[5] = 0.0  # qy
    data.joint("box_joint").qpos[6] = 0.0  # qz
def update_belt():
    global direction, cooldown
    if PHASE4_MODE and not PHASE4_ALLOW_REVERSE:
        cooldown = max(0.0, cooldown - model.opt.timestep)
        return
    if cooldown <= 0.0 and box_hits_end_wall():
        direction *= -1.0
        set_belt_direction(direction)
        cooldown = COOLDOWN_T
    cooldown = max(0.0, cooldown - model.opt.timestep)
def hold(viewer, duration: float):
    """
    Keep the current arm/gripper commands and advance the simulation
    for `duration` seconds of *simulation* time.
    """
    n = max(1, int(duration / model.opt.timestep))
    for _ in range(n):
        step_sim(viewer)
    viewer.sync()
def step_sim(viewer):
    global _vision_frame_counter
    update_belt()
    mujoco.mj_step(model, data)
    viewer.sync()
    _vision_frame_counter += 1
    if SHOW_CAMERA_VIEW and (_vision_frame_counter % SHOW_EVERY_N_STEPS == 0):
        try:
            rgb = render_overhead()
            uv, bbox = None, None
            p = get_box_pos_vision()
            if p is not None:
                uv = world_to_uv(p)
                if uv is not None and _last_features is not None:
                    # scale box from locked metric size
                    mpp = meters_per_pixel_on_belt()
                    half = 0.5 * max(_last_features["length"],
                                     _last_features["width"])
                    s = max(12.0, half / (mpp + 1e-9))
                    bbox = (uv[0] - s, uv[1] - s, uv[0] + s, uv[1] + s)
            show_camera_view(rgb, uv=uv, text="track", bbox=bbox)
        except Exception as e:
            print("vision display error:", e)

# Trials
def set_box_geom(size_xyz, rgba=(1, 0, 0, 1)):
    """Change box half-size and colour once per trial."""
    model.geom_size[box_geom_id][:] = np.asarray(size_xyz, dtype=float)
    model.geom_rgba[box_geom_id][:] = np.asarray(rgba, dtype=float)
def spawn_box_phase4(size_xyz, rgba, yaw_deg=None):
    set_box_geom(size_xyz, rgba)
    hz = float(size_xyz[2])
    z = BELT_Z + hz + 0.0025   # 1 mm air gap, then settle — or exact: BELT_TOP_Z + hz

    if yaw_deg is None:
        yaw_deg = float(np.random.uniform(*PHASE4_YAW_RANGE_DEG))
    yaw = np.deg2rad(yaw_deg)

    data.joint("box_joint").qpos[0] = PHASE4_SPAWN_X
    data.joint("box_joint").qpos[1] = PHASE4_SPAWN_Y
    data.joint("box_joint").qpos[2] = z
    data.joint("box_joint").qpos[3] = np.cos(yaw / 2.0)
    data.joint("box_joint").qpos[4:6] = 0.0
    data.joint("box_joint").qpos[6] = np.sin(yaw / 2.0)

    jnt = model.joint("box_joint")
    adr = jnt.dofadr[0]
    data.qvel[adr:adr + 6] = 0.0

    mujoco.mj_forward(model, data)

    stop_belt()
    for _ in range(5):
        mujoco.mj_step(model, data)
    set_belt_direction(direction)

    return float(yaw_deg)
def measure_xy_err_at_close():
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(get_pinch_pos()[:2] - get_box_pos()[:2]))
def classify_failure(feats, success, xy_err_close, waited_ok):
    if not waited_ok:
        return "timeout_or_no_intercept"
    if feats is None or not feats.get("quality_ok", False):
        return "bad_mask_or_no_feats"
    if success:
        return "ok"
    if xy_err_close is not None and xy_err_close > 0.04:
        return "xy_error_at_close"
    return "lift_fail_or_unknown"
def summarize_phase4(rows):
    from collections import defaultdict
    by_class = defaultdict(list)
    fail_counts = defaultdict(int)
    for r in rows:
        by_class[r["class"]].append(r)
        if not r["success"]:
            fail_counts[r["fail"] or "unknown"] += 1

    print("\n========== SUCCESS TABLE ==========")
    print(f"{'class':<12} {'n':>4} {'ok':>4} {'rate':>7} {'mean_xy':>8} "
          f"{'mean_t':>7} {'yaw_list'}")
    for cls, lst in by_class.items():
        n = len(lst)
        ok = sum(1 for r in lst if r["success"])
        xy = [r["xy_err_close"] for r in lst if r["xy_err_close"] is not None]
        ts = [r["t_pick"] for r in lst if r["t_pick"] is not None]
        yaws = [r.get("yaw_deg") for r in lst if r.get("yaw_deg") is not None]
        mean_xy = float(np.mean(xy)) if xy else float("nan")
        mean_t = float(np.mean(ts)) if ts else float("nan")
        yaw_str = ", ".join(f"{y:.1f}" for y in yaws) if yaws else "-"
        print(f"{cls:<12} {n:4d} {ok:4d} {ok / max(n, 1):7.1%} {mean_xy:8.3f} "
              f"{mean_t:7.2f} [{yaw_str}]")

    print("\n----- per-trial yaw / result -----")
    for r in rows:
        print(f"  {r['class']:<10} yaw_deg={r.get('yaw_deg', float('nan')):6.1f}  "
              f"success={r['success']}  tag={r['fail']}")

    print("\n----- failure count -----")
    for k, v in sorted(fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print("===========================================\n")


# Helpers arm
def plan_grasp(feats: dict) -> dict:
    if feats is None:
        raise ValueError("plan_grasp: feats is None")

    h = float(feats["height"])
    L = float(feats["length"])
    W = float(feats["width"])
    yaw = float(feats["yaw"])
    aspect = float(feats.get("aspect", L / (W + 1e-9)))
    circ = float(feats.get("circularity", 0.7))

    object_half_h = h/2

    # --- vertical (PROVEN offsets) ---
    grasp_z    = BELT_Z + h + GRASP_Z_OFFSET - 0.02
    approach_z = BELT_Z + h + APPROACH_Z_OFFSET
    lift_z     = BELT_Z + h + LIFT_Z_OFFSET

    # --- gripper pre-shape ---
    grip_width = float(np.clip(min(L, W) + GRIP_MARGIN, GRIP_MIN, GRIP_MAX))
    grip_ctrl  = width_to_grip_ctrl(grip_width)

    if aspect >= ASPECT_LONG:
        strategy = "top_long"
        wrist_yaw = yaw + np.pi / 2.0
    elif circ >= 0.85 and aspect < 1.25:
        strategy = "top_round"
        wrist_yaw = 0.0
    else:
        strategy = "top_square"
        wrist_yaw = yaw
    wrist_yaw = (wrist_yaw + np.pi) % (2 * np.pi) - np.pi

    plan = {
        "strategy": strategy,
        "approach_z": approach_z,
        "grasp_z": grasp_z,
        "lift_z": lift_z,
        "grip_width_m": grip_width,
        "grip_ctrl": grip_ctrl,
        "wrist_yaw": wrist_yaw,
        "object_height": h,
        "object_L": L,
        "object_W": W,
        "aspect": aspect,
    }
    print(f"[PHASE2 plan] {strategy}  h={h:.3f}  "
          f"Z app/grasp/lift={approach_z:.3f}/{grasp_z:.3f}/{lift_z:.3f}  "
          f"grip_w={grip_width:.3f}m ctrl={grip_ctrl:.1f}  "
          f"yaw_deg={np.rad2deg(wrist_yaw):.1f}")
    return plan
def open_gripper_to(width_m=None):
    """Open to a metric width, or fully open if width_m is None."""
    if gripper_act_id < 0:
        return
    if width_m is None:
        data.ctrl[gripper_act_id] = GRIP_CTRL_OPEN
    else:
        data.ctrl[gripper_act_id] = width_to_grip_ctrl(width_m)
def set_gripper_from_plan(plan):
    """Pre-shape fingers to planned opening before descend."""
    if gripper_act_id >= 0:
        data.ctrl[gripper_act_id] = plan["grip_ctrl"]
def apply_wrist_yaw(viewer, wrist_yaw, duration=None):
    """Set wrist_3 = plan yaw + offset. Keep duration short so the belt does not run away."""
    if duration is None:
        duration = WRIST_DURATION
    q = get_joint_qpos().copy()
    q[5] = float(wrist_yaw)
    q[5] = (q[5] + np.pi) % (2 * np.pi) - np.pi
    print(f"[wrist] plan_deg={np.rad2deg(wrist_yaw):.1f}  "
          f"cmd_deg={np.rad2deg(q[5]):.1f}")
    move_to_q(viewer, q, duration=duration)
def width_to_grip_ctrl(width_m: float) -> float:
    w = float(np.clip(width_m, CLOSED_WIDTH_M, OPEN_WIDTH_M))
    if OPEN_WIDTH_M <= CLOSED_WIDTH_M + 1e-9:
        return GRIP_CTRL_OPEN
    # larger width → more open → ctrl nearer 0
    alpha = (OPEN_WIDTH_M - w) / (OPEN_WIDTH_M - CLOSED_WIDTH_M)
    return float(GRIP_CTRL_OPEN + alpha * (GRIP_CTRL_CLOSED - GRIP_CTRL_OPEN))
def grasp_xy_offset_from_feats(feats):
    """
    Base GRASP_XY_OFFSET plus size-aware Y tweak.
    Keeps X as you have it; adapts Y for small W.
    """
    base = np.array(GRASP_XY_OFFSET, dtype=float).copy()
    W = float(feats.get("width", 0.06))
    # For W < 5 cm, pull a bit harder toward centre (more negative Y if residual is +Y)
    if W < 0.055:
        base[1] = base[1] + GRASP_Y_RESIDUAL * 1.0   # e.g. -0.045 + (-0.03) = -0.075
    else:
        base[1] = base[1] + GRASP_Y_RESIDUAL * 0.7
    return base

# YOLO
def yolo_detect(rgb):
    """Run YOLO (throttled). Keep only detections on the belt."""
    global _yolo_counter, _last_yolo_det
    _yolo_counter += 1
    if _yolo_counter % YOLO_EVERY_N != 0 and _last_yolo_det is not None:
        return _last_yolo_det

    results = yolo_model.predict(
        source=rgb,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        device="cpu",
    )
    dets = []
    if not results or results[0].boxes is None:
        _last_yolo_det = dets
        return dets

    r = results[0]
    names = r.names
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        name = names.get(cls, str(cls))
        u = 0.5 * (x1 + x2)
        v = 0.5 * (y1 + y2)
        p = pixel_to_world_on_plane(u, v, z_plane=BELT_Z)
        if p is None or not on_belt_xy(p):
            continue
        dets.append({
            "xyxy": (float(x1), float(y1), float(x2), float(y2)),
            "conf": conf,
            "cls": cls,
            "name": name,
            "uv": (float(u), float(v)),
            "mask": None,
        })
    dets.sort(key=lambda d: d["conf"], reverse=True)
    _last_yolo_det = dets
    return dets
def yolo_detect_rgb_only(rgb):
    """Fast YOLO on RGB only. No depth render."""
    results = yolo_model.predict(
        source=rgb, conf=YOLO_CONF, iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ, verbose=False, device="cpu"
    )
    dets = []
    if not results or results[0].boxes is None:
        return dets
    r = results[0]
    for box in r.boxes:
        x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls  = int(box.cls[0])
        name = r.names.get(cls, str(cls))
        u = 0.5*(x1+x2)
        v = 0.5*(y1+y2)
        p = pixel_to_world_on_plane(u, v, z_plane=BELT_Z)
        if p is None or not on_belt_xy(p):
            continue
        dets.append({
            "xyxy": (float(x1),float(y1),float(x2),float(y2)),
            "conf": conf, "cls": cls, "name": name,
            "uv": (float(u), float(v)), "mask": None,
        })
    dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets
def yolo_detect_timed(rgb):
    """Run YOLO and return (dets, latency_s)."""
    t0 = time.perf_counter()
    results = yolo_model.predict(
        source=rgb,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        verbose=False,
        device="cpu",
    )
    t_yolo = time.perf_counter() - t0

    dets = []
    if results and results[0].boxes is not None:
        r = results[0]
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = r.names.get(cls, str(cls))
            u = 0.5 * (x1 + x2)
            v = 0.5 * (y1 + y2)
            p = pixel_to_world_on_plane(u, v, z_plane=BELT_Z)
            if p is None or not on_belt_xy(p):
                continue
            dets.append({
                "xyxy": (float(x1), float(y1), float(x2), float(y2)),
                "conf": conf,
                "cls": cls,
                "name": name,
                "uv": (float(u), float(v)),
                "mask": None,
            })
        dets.sort(key=lambda d: d["conf"], reverse=True)
    return dets, t_yolo
def measure_object_once_timed(label="lock"):
    """Full Phase-1 style measure with latency breakdown."""
    t0 = time.perf_counter()
    frame = capture_overhead_frame()          # includes render
    t_render = time.perf_counter() - t0

    rgb = frame["rgb"]
    height_map = frame["height"]

    t1 = time.perf_counter()
    dets, t_yolo = yolo_detect_timed(rgb)     # optional path
    # primary: height features (your Phase-1)
    mask = segment_object_by_height(height_map)
    feats = geometry_features_from_mask(mask, height_map)
    t_post = time.perf_counter() - t1

    t_total = time.perf_counter() - t0
    # print(f"[latency {label}] render={t_render*1000:.1f}ms  "
    #       f"yolo={t_yolo*1000:.1f}ms  post={t_post*1000:.1f}ms  "
    #       f"total={t_total*1000:.1f}ms")

    return feats, t_total, frame
def compensate_for_latency(p_xy, latency_s, v=None):
    """
    p_xy: centre at image/capture time (world X,Y)
    latency_s: seconds from capture (or from shutter) to 'now'
    """
    if v is None:
        v = known_belt_velocity()
    p = np.asarray(p_xy, dtype=float).copy()
    p[0] += v[0] * latency_s
    p[1] += v[1] * latency_s
    return p

# OpenCV:
def world_to_uv(p_world):
    mujoco.mj_forward(model, data)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)
    fovy = model.cam_fovy[cam_id] * np.pi / 180.0
    fy = (RENDER_H / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    cx, cy = RENDER_W / 2.0, RENDER_H / 2.0
    v_c = cam_mat.T @ (np.asarray(p_world, float) - cam_pos)
    if v_c[2] >= -1e-9:
        return None
    u = cx + fx * (v_c[0] / -v_c[2])
    v = cy - fy * (v_c[1] / -v_c[2])
    if not (0 <= u < RENDER_W and 0 <= v < RENDER_H):
        return None
    return float(u), float(v)
def measure_height_once():
    """One-shot: depth + reliable XY from height (or red). YOLO only for optional label."""
    global _last_object_height, _last_features, _height_fixed
    global _track_xy, _track_t, _last_bbox, _last_uv

    frame = capture_overhead_frame()
    rgb        = frame["rgb"]
    height_map = frame["height"]

    # --- primary: height mask on belt (colour-independent, reliable for cube) ---
    mask = segment_object_by_height(height_map)
    feats = geometry_features_from_mask(mask, height_map)
    bbox = None
    uv = None

    if feats is not None:
        uv = feats["uv"]
        # bbox from the same contour that produced the features
        mask_u8 = (mask.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            # pick contour whose centre matches feats uv (on belt)
            best = None
            best_d = 1e9
            for c in cnts:
                if cv2.contourArea(c) < MIN_OBJ_AREA:
                    continue
                M = cv2.moments(c)
                if M["m00"] < 1e-6:
                    continue
                cu = M["m10"] / M["m00"]
                cv_ = M["m01"] / M["m00"]
                d = (cu - uv[0])**2 + (cv_ - uv[1])**2
                if d < best_d:
                    best_d = d
                    best = c
            if best is not None:
                x, y, w, h = cv2.boundingRect(best)
                bbox = (float(x), float(y), float(x + w), float(y + h))

    # --- fallback: red HSV if height failed ---
    if feats is None:
        uv = detect_box_uv(rgb)
        if uv is not None:
            p = pixel_to_world_on_plane(uv[0], uv[1], z_plane=BELT_Z)
            if p is not None and on_belt_xy(p):
                h = height_at_uv(height_map, uv[0], uv[1])
                if 0.005 < h < 0.25:
                    feats = {
                        "uv": uv,
                        "center_xy": (float(p[0]), float(p[1])),
                        "center_z": float(BELT_Z + h),
                        "height": float(h),
                        "width": 0.06, "length": 0.06,
                        "yaw": 0.0, "circularity": 0.8, "aspect": 1.0,
                        "extent": 0.9, "area_px": 0.0, "rect_angle_deg": 0.0,
                    }
                    # small box around uv for drawing
                    s = 20
                    bbox = (uv[0]-s, uv[1]-s, uv[0]+s, uv[1]+s)

    if feats is not None:
        _last_features = feats
        _last_object_height = feats["height"]
        _height_fixed = True
        _track_xy = np.array(feats["center_xy"], dtype=float)
        _track_t = data.time
        _last_bbox = bbox
        _last_uv = feats["uv"]
        print(f"[height ONCE] h={feats['height']:.4f}  xy={feats['center_xy']}")
        show_camera_view(rgb, _last_uv, text=f"h={feats['height']:.3f}", bbox=bbox)
        return feats

    show_camera_view(rgb, None, text="no object")
    return None
def render_overhead():
    renderer.update_scene(data, camera=CAM_NAME)
    return renderer.render()
def render_overhead_rgb_depth():
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    rgb = renderer.render()

    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=CAM_NAME)
    depth = renderer.render()
    renderer.disable_depth_rendering()

    return rgb, np.asarray(depth, dtype=np.float32)
def camera_world_z():
    mujoco.mj_forward(model, data)
    return float(data.cam_xpos[cam_id][2])
def expected_belt_depth():
    return camera_world_z() - BELT_Z
def depth_to_height_above_belt(depth, belt_depth=None):
    if belt_depth is None:
        belt_depth = expected_belt_depth()
    return belt_depth - depth
def capture_overhead_frame():
    rgb, depth = render_overhead_rgb_depth()
    belt_depth = expected_belt_depth()   # fixed plane — correct for top-down
    height = depth_to_height_above_belt(depth, belt_depth=belt_depth)
    return {
        "rgb": rgb,
        "depth": depth,
        "height": height,
        "belt_depth": belt_depth,
        "cam_z": camera_world_z(),
    }
def height_at_uv(height_map, u, v):
    h, w = height_map.shape
    ui = int(np.clip(round(u), 0, w - 1))
    vi = int(np.clip(round(v), 0, h - 1))
    return float(height_map[vi, ui])
def object_height_stats(height_map, mask):
    """
    mask: (H, W) bool – object pixels
    returns median/max height above belt on the object.
    """
    vals = height_map[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return {
        "median": float(np.median(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
    }
def debug_depth_at_red_object():
    frame = capture_overhead_frame()
    rgb, height = frame["rgb"], frame["height"]
    uv = detect_box_uv(rgb)  # your existing function; test only
    if uv is None:
        print("no blob")
        return
    h = height_at_uv(height, uv[0], uv[1])
    print(f"uv={uv} height_above_belt={h:.4f} m  belt_depth={frame['belt_depth']:.4f}")
def show_depth_debug(depth, height):
    # Normalize for display only
    d = depth.copy()
    d[~np.isfinite(d)] = 0
    d_show = (255 * (d - d.min()) / (d.max() - d.min() + 1e-6)).astype(np.uint8)
    h = np.clip(height, 0, 0.2)
    h_show = (255 * h / 0.2).astype(np.uint8)
    cv2.imshow("depth", d_show)
    cv2.imshow("height_above_belt", h_show)
    cv2.waitKey(1)
def detect_box_uv(rgb):
    """Return (u, v) or None. Tune HSV for your box colour."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    m2 = cv2.inRange(hsv, (170, 80, 60), (180, 255, 255))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 80:
        return None
    M = cv2.moments(c)
    if M["m00"] < 1e-6:
        return None
    return float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])
def pixel_to_world_on_plane(u, v, z_plane=BELT_Z):
    mujoco.mj_forward(model, data)
    cam_pos = data.cam_xpos[cam_id].copy()
    cam_mat = data.cam_xmat[cam_id].reshape(3, 3)

    fovy = model.cam_fovy[cam_id] * np.pi / 180.0
    fy = (RENDER_H / 2.0) / np.tan(fovy / 2.0)
    fx = fy
    cx, cy = RENDER_W / 2.0, RENDER_H / 2.0

    x = (u - cx) / fx
    y = -(v - cy) / fy
    ray_c = np.array([x, y, -1.0])
    ray_c /= np.linalg.norm(ray_c)
    ray_w = cam_mat @ ray_c

    if abs(ray_w[2]) < 1e-9:
        return None
    t = (z_plane - cam_pos[2]) / ray_w[2]
    if t < 0:
        return None
    return cam_pos + t * ray_w
def show_camera_view(rgb, uv=None, text="", mask=None, bbox=None):
    if not SHOW_CAMERA_VIEW:
        return
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if uv is not None:
        cv2.circle(vis, (int(uv[0]), int(uv[1])), 8, (0, 255, 0), 2)
    if text:
        cv2.putText(vis, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)
    cv2.imshow("Overhead camera", vis)
    cv2.waitKey(1)
def get_box_pos_vision():
    if not _height_fixed or _track_xy is None:
        return None
    dt = data.time - _track_t          # sim time, not wall time
    v  = known_belt_velocity()
    xy = _track_xy + v[:2] * dt
    z  = BELT_Z + _last_object_height
    return np.array([xy[0], xy[1], z], dtype=float)
def get_measured_object_height():
    """For grasp_z / BOX_HALF_H at pick start."""
    return float(_last_object_height)
def known_belt_velocity():
    """Robot knows belt speed (no need to estimate from vision)."""
    return np.array([direction * belt_speed, 0.0, 0.0])
def predict_pos_from(p, dt):
    v = known_belt_velocity()
    out = np.asarray(p, dtype=float).copy()
    out[0] += v[0] * dt
    out[1] += v[1] * dt
    return out
def wait_for_box_at_camera_then_intercept(viewer, timeout=100) -> bool:
    reset_perception()
    start = time.time()
    seen = False
    wrist_done = False          # local only — not a Phase-3 global

    while time.time() - start < timeout:
        step_sim(viewer)

        if not _height_fixed:
            feats = measure_object_once("first")
            if feats is None:
                continue
            p = np.array([feats["center_xy"][0], feats["center_xy"][1],
                          feats["center_z"]], dtype=float)
        else:
            p = get_box_pos_vision()
            if p is None:
                continue

        if not seen:
            if DETECT_X_MIN < p[0] < DETECT_X_MAX:
                seen = True
                print(f"[vision] seen at x={p[0]:.3f}")

                # ----- early wrist (still Phase-2 pipeline) -----
                if not wrist_done:
                    feats = get_perception_features()
                    if feats is not None:
                        plan = plan_grasp(feats)
                        apply_wrist_yaw(viewer, plan["wrist_yaw"], duration=WRIST_DURATION)
                        wrist_done = True
            else:
                continue

        T = estimate_motion_time()
        p_pred = predict_pos_from(p, T)
        v = known_belt_velocity()
        approaching = (v[0] * (INTERCEPT_X_TARGET - p[0])) > 0
        if approaching and abs(p_pred[0] - INTERCEPT_X_TARGET) < INTERCEPT_X_TOL:
            feats = measure_object_once("re-lock")
            if feats is None:
                continue
            # optional: refresh wrist after re-lock if aspect/yaw changed a lot
            if not wrist_done and feats is not None:
                plan = plan_grasp(feats)
                apply_wrist_yaw(viewer, plan["wrist_yaw"], duration=WRIST_DURATION)
                wrist_done = True
            p = np.array([feats["center_xy"][0], feats["center_xy"][1],
                          feats["center_z"]], dtype=float)
            print(f"[vision] Go pick x={p[0]:.3f} pred={p_pred[0]:.3f} T={T:.2f}")
            return True

    print("[vision] timeout")
    return False
def get_box_pos_sense():
    if USE_VISION:
        p = get_box_pos_vision()
        if p is not None:
            return p
    return get_box_pos()  # fallback
def get_box_vel_sense():
    # Prefer known belt speed (stable); vision only for position
    return known_belt_velocity()
def segment_object_by_height(height_map, thresh=HEIGHT_SEG_THRESH):
    """
    height_map: (H, W) metres above belt
    returns mask (H, W) bool
    """
    mask = np.isfinite(height_map) & (height_map > thresh)
    # light cleanup
    mask_u8 = (mask.astype(np.uint8)) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask_u8 > 0
def on_belt_xy(p):
    return (BELT_X_MIN <= p[0] <= BELT_X_MAX and
            BELT_Y_MIN <= p[1] <= BELT_Y_MAX)
def detect_uv_from_mask(mask):
    """Largest blob whose center projects onto the belt."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        if cv2.contourArea(c) < MIN_OBJ_AREA:
            break
        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            continue
        u = float(M["m10"] / M["m00"])
        v = float(M["m01"] / M["m00"])
        p = pixel_to_world_on_plane(u, v, z_plane=BELT_Z)
        if p is not None and on_belt_xy(p):
            return (u, v)
    return None
def detect_object_uv(rgb, height_map=None):
    """
    Primary: height mask (any color).
    Fallback: red HSV (your old detect_box_uv).
    """
    if USE_HEIGHT_SEGMENT and height_map is not None:
        mask = segment_object_by_height(height_map)
        uv = detect_uv_from_mask(mask)
        if uv is not None:
            return uv, mask

    uv = detect_box_uv(rgb)  # existing red detector
    return uv, None
def meters_per_pixel_on_belt():
    """
    Approximate m/px at belt distance from camera intrinsics + depth.
    """
    belt_depth = expected_belt_depth()          # ~ cam_z - BELT_Z
    fovy = model.cam_fovy[cam_id] * np.pi / 180.0
    fy = (RENDER_H / 2.0) / np.tan(fovy / 2.0)
    # pinhole: size_m = size_px * depth / fy
    return float(belt_depth / fy)
def geometry_features_from_mask(mask, height_map):
    """
    Build full Phase-1 feature vector from height mask.
    Returns dict or None.
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    chosen, uv, p = None, None, None
    for c in cnts:
        if cv2.contourArea(c) < MIN_OBJ_AREA:
            break
        M = cv2.moments(c)
        if M["m00"] < 1e-6:
            continue
        u = float(M["m10"] / M["m00"])
        v = float(M["m01"] / M["m00"])
        pw = pixel_to_world_on_plane(u, v, z_plane=BELT_Z)
        if pw is not None and on_belt_xy(pw):
            chosen, uv, p = c, (u, v), pw
            break
    if chosen is None:
        return None

    area_px = float(cv2.contourArea(chosen))
    peri = float(cv2.arcLength(chosen, True))
    circularity = float(4.0 * np.pi * area_px / (peri * peri + 1e-6))

    (cx, cy), (w_px, h_px), angle_deg = cv2.minAreaRect(chosen)
    if w_px < h_px:
        w_px, h_px = h_px, w_px
        angle_deg += 90.0
    length_px = max(w_px, h_px)
    width_px  = min(w_px, h_px)

    mpp = meters_per_pixel_on_belt()
    length_m = length_px * mpp
    width_m  = width_px * mpp
    aspect   = float(length_m / (width_m + 1e-9))

    x, y, bw, bh = cv2.boundingRect(chosen)
    extent = float(area_px / (bw * bh + 1e-6))
    yaw = np.deg2rad(angle_deg)
    bbox = (float(x), float(y), float(x + bw), float(y + bh))

    cont_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(cont_mask, [chosen], -1, 1, -1)
    hs = height_map[cont_mask > 0]
    hs = hs[np.isfinite(hs) & (hs > HEIGHT_SEG_THRESH) & (hs < HEIGHT_SEG_MAX)]
    if hs.size == 0:
        h_med = height_at_uv(height_map, uv[0], uv[1])
    else:
        h_med = float(np.median(hs))

    feats = {
        "uv": uv,
        "bbox": bbox,
        "center_xy": (float(p[0]), float(p[1])),
        "center_z": float(BELT_Z + h_med),
        "height": float(h_med),
        "width": float(width_m),
        "length": float(length_m),
        "yaw": float(yaw),
        "circularity": float(np.clip(circularity, 0.0, 1.5)),
        "aspect": float(aspect),
        "extent": float(extent),
        "area_px": area_px,
        "rect_angle_deg": float(angle_deg),
        "quality_ok": False,   # set below
    }
    feats["quality_ok"] = features_quality_ok(feats)
    return feats
def sense_object():
    """
    Expensive perception – call only when needed.
    YOLO on belt → height from depth → lock track.
    """
    global _last_features, _last_object_height, _track_xy, _track_t, _last_bbox

    frame = capture_overhead_frame()
    rgb        = frame["rgb"]
    height_map = frame["height"]

    dets = yolo_detect(rgb)
    feats = None
    bbox  = None

    if dets:
        d = dets[0]
        u, v = d["uv"]
        bbox = d["xyxy"]
        ui, vi = int(round(u)), int(round(v))
        hs = height_map[max(0, vi-6):vi+7, max(0, ui-6):ui+7].ravel()
        hs = hs[np.isfinite(hs) & (hs > 0.008) & (hs < 0.20)]
        h = float(np.median(hs)) if hs.size > 0 else height_at_uv(height_map, u, v)
        p = pixel_to_world_on_plane(u, v, z_plane=BELT_Z)
        if p is not None and 0.005 < h < 0.25:
            mpp = meters_per_pixel_on_belt()
            x1, y1, x2, y2 = d["xyxy"]
            feats = {
                "uv": (u, v),
                "center_xy": (float(p[0]), float(p[1])),
                "center_z": float(BELT_Z + h),
                "height": h,
                "width":  abs(x2 - x1) * mpp,
                "length": abs(y2 - y1) * mpp,
                "yaw": 0.0,
                "circularity": 0.0,
                "aspect": 1.0,
                "extent": 0.0,
                "area_px": 0.0,
                "rect_angle_deg": 0.0,
                "yolo_name": d["name"],
                "yolo_conf": d["conf"],
            }

    # fallback: height segmentation (still colour-independent)
    if feats is None:
        mask = segment_object_by_height(height_map)
        feats = geometry_features_from_mask(mask, height_map)

    if feats is not None:
        _last_features = feats
        if 0.005 < feats["height"] < 0.25:
            _last_object_height = feats["height"]
        _track_xy = np.array(feats["center_xy"], dtype=float)
        _track_t  = data.time
        _last_bbox = bbox
        show_camera_view(rgb, feats.get("uv"),
                         text=f"h={feats['height']:.3f} {feats.get('yolo_name','')}",
                         bbox=bbox)
    else:
        show_camera_view(rgb, None, text="no object", bbox=None)
    return feats
def get_last_features():
    return _last_features
def features_quality_ok(feats):
    """Phase 1 quality gate — reject table / noise / absurd shapes."""
    if feats is None:
        return False
    h = feats["height"]
    w = feats["width"]
    L = feats["length"]
    if not (MIN_HEIGHT <= h <= MAX_HEIGHT):
        return False
    if not (MIN_WIDTH_M <= w <= MAX_WIDTH_M):
        return False
    if not (MIN_WIDTH_M <= L <= MAX_WIDTH_M):
        return False
    if feats["aspect"] > MAX_ASPECT:
        return False
    if feats["area_px"] < MIN_OBJ_AREA:
        return False
    cx, cy = feats["center_xy"]
    if not on_belt_xy(np.array([cx, cy, BELT_Z])):
        return False
    return True
def measure_object_once(label="lock"):
    """
    PHASE 1 single measurement + latency compensation.
    centre_xy is shifted by belt velocity * (time spent measuring)
    so the lock matches the box *now*, not the image time.
    """
    global _last_features, _last_object_height, _height_fixed
    global _track_xy, _track_t, _last_uv, _last_bbox

    t0 = time.perf_counter()

    frame = capture_overhead_frame()
    t_render = time.perf_counter() - t0

    rgb = frame["rgb"]
    height_map = frame["height"]

    t1 = time.perf_counter()
    mask = segment_object_by_height(height_map)
    feats = geometry_features_from_mask(mask, height_map)
    t_post = time.perf_counter() - t1

    t_total = time.perf_counter() - t0

    if feats is None or not feats["quality_ok"]:
        show_camera_view(rgb, None, text=f"{label}: reject")
        return None

    # ----- image-time centre -----
    xy_img = np.array(feats["center_xy"], dtype=float)

    # ----- compensate to 'now' (end of measurement) -----
    xy_now = compensate_for_latency(xy_img, t_total)
    feats["center_xy_raw"] = (float(xy_img[0]), float(xy_img[1]))
    feats["center_xy"] = (float(xy_now[0]), float(xy_now[1]))
    feats["center_z"] = float(BELT_Z + feats["height"])
    feats["latency_s"] = float(t_total)

    # lock on compensated pose, timestamp = current sim time
    _last_features = feats
    _last_object_height = feats["height"]
    _height_fixed = True
    _track_xy = xy_now.copy()
    _track_t = float(data.time)
    _last_uv = feats["uv"]
    _last_bbox = feats.get("bbox")

    # print(f"[latency {label}] render={t_render*1000:.1f}ms  "
    #       f"post={t_post*1000:.1f}ms  total={t_total*1000:.1f}ms  "
    #       f"shift_x={xy_now[0]-xy_img[0]:+.4f}m")
    print(f"[PHASE1 {label}] h={feats['height']:.4f}  "
          f"xy_raw=({xy_img[0]:.3f},{xy_img[1]:.3f})  "
          f"xy_now=({xy_now[0]:.3f},{xy_now[1]:.3f})  "
          f"L={feats['length']:.3f} W={feats['width']:.3f}")

    # draw using compensated world → image (square tracks live box)
    uv_draw = world_to_uv([xy_now[0], xy_now[1], BELT_Z + feats["height"]])
    if uv_draw is None:
        uv_draw = feats["uv"]
    show_camera_view(rgb, uv_draw,
                     text=f"h={feats['height']:.3f} lat={t_total*1000:.0f}ms",
                     bbox=feats.get("bbox"))
    return feats
def get_perception_features():
    """Last accepted Phase-1 feature vector (or None)."""
    return _last_features
def reset_perception():
    global _height_fixed, _track_xy, _track_t, _last_features
    global _last_uv, _last_bbox
    _height_fixed = False
    _track_xy = None
    _track_t = 0.0
    _last_features = None
    _last_uv = None
    _last_bbox = None

# IK:
def world_to_base(p_world: np.ndarray) -> np.ndarray:
    """World position -> position in arm base frame."""
    # body "base" from ur5e.xml
    R = data.body("base").xmat.reshape(3, 3)
    t = data.body("base").xpos
    return R.T @ (p_world - t)
def base_to_world(p_base: np.ndarray) -> np.ndarray:
    R = data.body("base").xmat.reshape(3, 3)
    t = data.body("base").xpos
    return R @ p_base + t
def top_down_orientation():
    # Columns = x, y, z axes of the tip in base frame
    # z down, x forward-ish – common for table pick
    return np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, 1.0],
    ])
def extra_lead_time():
    v = abs(direction * belt_speed)
    # ~2–4 cm worth of time at current speed
    return float(np.clip(0.04 + 0.17 * v, 0.05, 0.12))
def solve_ik(target_world: np.ndarray, orientation=None) -> np.ndarray | None:
    """
    Solve IK for tip at target_world (xyz in world).
    Returns 6 joint angles or None on failure.
    """
    target_base = world_to_base(np.asarray(target_world, dtype=float))
    q0_full = qpos_to_ikpy(get_joint_qpos())

    kwargs = dict(
        target_position=target_base,
        initial_position=q0_full,
    )
    if orientation is not None:
        kwargs["target_orientation"] = orientation
        kwargs["orientation_mode"] = "all"   # position + orientation

    try:
        full = arm_chain.inverse_kinematics(**kwargs)
        q6 = ikpy_to_qpos(full)
        return q6
    except Exception as e:
        print("IKPy failed:", e)
        return None
def get_box_vel_world() -> np.ndarray:
    """Best estimate of box linear velocity in world frame."""
    # Preferred: if you have a velocimeter / free-joint qvel
    # Free joint qvel layout: [vx, vy, vz, wx, wy, wz]
    jnt = model.joint("box_joint")
    if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
        # qvel index for free joint starts at dofadr
        adr = jnt.dofadr[0]
        return data.qvel[adr:adr+3].copy()
    # Fallback: known belt velocity
    return np.array([direction * belt_speed, 0.0, 0.0])
def predict_box_pos(dt: float) -> np.ndarray:
    """Constant-velocity prediction of box position after dt seconds."""
    p = get_box_pos_sense()
    v = get_box_vel_sense()
    # Only trust horizontal motion from the belt
    p_pred = p.copy()
    p_pred[0] += v[0] * dt
    p_pred[1] += v[1] * dt
    return p_pred
def aim_from_box(box_pos, t_lead, use_bias=True, xy_offset=None):
    v = get_box_vel_sense()
    aim = np.asarray(box_pos, dtype=float).copy()
    t_lead = float(t_lead) + extra_lead_time()
    aim[0] += v[0] * t_lead
    aim[1] += v[1] * t_lead
    if use_bias:
        aim[0] -= XY_BIAS[0]
        aim[1] -= XY_BIAS[1]
    off = GRASP_XY_OFFSET if xy_offset is None else np.asarray(xy_offset, dtype=float)
    aim[0] += off[0]
    aim[1] += off[1]
    return aim
def estimate_motion_time() -> float:
    q = get_joint_qpos()
    dq = np.linalg.norm(q - ABOVE_BELT_Q)
    T = 0.35 + dq / 1.5
    # At high speed, prefer faster, earlier motion
    v = abs(direction * belt_speed)
    if v > 0.25:
        T *= 0.75
    return float(np.clip(T, 0.45, 1.2))  # lower MAX than 1.8
def plan_intercept_times():
    """
    Returns dict of times that scale with how far the arm must move
    and are consistent with current belt speed.
    """
    T_move = estimate_motion_time()          # from current q to above-belt

    # Split the motion budget
    T_approach = 0.55 * T_move               # get above the line of the box
    T_descend  = 0.35 * T_move               # vertical + final XY
    T_match    = 0.2                        # fixed: close gripper while tracking
    T_polish   = 0.15                        # fixed short polish window

    # Extra lead used when aiming during polish (seconds of belt travel)
    polish_lead = 0.2

    return {
        "T_move": T_move,
        "T_approach": T_approach,
        "T_descend": T_descend,
        "T_match": T_match,
        "T_polish": T_polish,
        "polish_lead": polish_lead,
    }
def wait_for_good_intercept(viewer, timeout=60.0):
    start = time.time()
    while time.time() - start < timeout:
        step_sim(viewer)
        p = get_box_pos_sense()
        if p is None:
            continue

        v = known_belt_velocity()[0]
        if abs(v) < 1e-6:
            continue

        T = estimate_motion_time()
        # Time until box reaches middle
        t_to_mid = (INTERCEPT_X_TARGET - p[0]) / v

        # Start when we need about T seconds (with margin)
        if 0.0 < t_to_mid < T + 0.35:   # box still approaching middle
            print(f"GO t_to_mid={t_to_mid:.2f} T_arm={T:.2f} x={p[0]:.3f} v={v:.2f}")
            return True
    return False
def debug_y_offset():
    mujoco.mj_forward(model, data)
    box = get_box_pos()
    tcp = get_pinch_pos()
    print("XY_BIAS   :", XY_BIAS,"GRASP_XY_OFFSET:", GRASP_XY_OFFSET)
    print("box world :", box,"TCP world :", tcp)
    print("TCP - box :", tcp - box)
def update_xy_bias():
    """Call after a move settles. EMA so bias tracks slowly."""
    global XY_BIAS
    box = get_box_pos()
    tcp = get_pinch_pos()
    meas = tcp[:2] - box[:2]
    XY_BIAS = 0.7 * XY_BIAS + 0.3 * meas
    print(f"  XY_BIAS -> {XY_BIAS}, meas={meas}")
def polish_tcp_xy(viewer, target_xy, tol=0.025, max_iters=300):
    """
    Fine XY alignment of real pinch (TCP) to target_xy.
    target_xy: desired real TCP X,Y in world (usually true box xy).
    """
    target_xy = np.asarray(target_xy, float).reshape(2)

    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        cur = get_pinch_pos()
        err = target_xy - cur[:2]
        dist = float(np.linalg.norm(err))
        if dist < tol:
            print(f"  polish OK xy={dist:.4f}")
            hold(viewer, 0.1)
            return True

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, None, pinch_site_id)
        Jxy = jacp[:2, arm_dof_ids]
        q = get_joint_qpos().copy()

        if np.linalg.norm(Jxy) > 1e-6:
            step = np.clip(0.7 * err, -0.02, 0.02)
            A = Jxy @ Jxy.T + 1e-3 * np.eye(2)
            dq = Jxy.T @ np.linalg.solve(A, step)
            dq = np.clip(dq, -0.025, 0.025)
            q = q + dq
        else:
            # Jacobian unusable (welded gripper): small joint nudges
            # If TCP is +X/+Y of box, err is negative → these reduce |err|
            q[0] += np.clip(-0.5 * err[0], -0.02, 0.02)  # pan
            q[1] += np.clip(-0.4 * err[1], -0.02, 0.02)  # lift
            q[2] += np.clip(-0.3 * err[1], -0.02, 0.02)  # elbow
            # If xy_err grows, flip the signs on the three lines above

        set_arm_ctrl(q)
        step_sim(viewer)

        # if it % 50 == 0:
        #     print(f"  polish it={it} xy={dist:.3f} err={err} tcp={cur[:2]}")

    dist = np.linalg.norm(get_pinch_pos()[:2] - target_xy)
    print(f"  polish end xy={dist:.4f}")
    return dist < tol * 1.2
def move_pinch_to(viewer, target_pos, duration=1.2, label="",
                  do_polish=True, success_mode="target"):
    target = np.asarray(target_pos, float).reshape(3)
    print(f" move_pinch_to [{label}] ik_cmd={target}")

    q_goal = solve_ik(target, orientation=None)
    if q_goal is None:
        print(" IKPy failed")
        return False

    q0 = get_joint_qpos()
    for i in range(6):
        d = q_goal[i] - q0[i]
        q_goal[i] = q0[i] + (d + np.pi) % (2 * np.pi) - np.pi

    n = max(1, int(duration / model.opt.timestep))
    for t in range(n):
        a = (t + 1) / n
        s = a * a * (3 - 2 * a)
        set_arm_ctrl((1 - s) * q0 + s * q_goal)
        step_sim(viewer)

    wait_until_joints(viewer, q_goal, tol=0.05, settle=0.25)

    # Polish only for final grasp alignment (unchanged logic)
    if do_polish and success_mode == "box":
        box_xy = get_box_pos()[:2]
        print(f" polish toward box XY {box_xy}")
        polish_tcp_xy(viewer, box_xy, tol=0.025)

    mujoco.mj_forward(model, data)
    tcp = get_pinch_pos()
    box = get_box_pos()

    if success_mode == "box":
        xy_err = np.linalg.norm(tcp[:2] - box[:2])
        ok = xy_err < 0.04
        print(f" [{label}] box mode xy_err={xy_err:.4f} ok={ok}")
    else:
        # your existing target-mode check (sign-flip convention)
        xy_err = np.linalg.norm(tcp[:2] - np.array([-target[0], -target[1]]))
        ok = xy_err < 0.08
        print(f" [{label}] target mode xy_err={xy_err:.4f} tcp={tcp} ok={ok}")

    # ----- FIX 2: bias only from trustworthy settles -----
    if ok:
        update_xy_bias()
    else:
        print(f"  skip XY_BIAS update (xy_err={xy_err:.4f})")

    return ok
def safe_pick_box_moving(viewer) -> bool:
    global XY_BIAS

    feats = get_perception_features()
    plan = plan_grasp(feats)
    xy_off = grasp_xy_offset_from_feats(feats)
    print(f"[PHASE2 xy_off] {xy_off}")

    # every aim_from_box during this pick:
    # aim = aim_from_box(get_box_pos_sense(), t_lead=..., use_bias=True, xy_offset=xy_off)
    APPROACH_Z = plan["approach_z"]
    GRASP_Z    = plan["grasp_z"]
    LIFT_Z     = plan["lift_z"]

    open_gripper()                    # fully open while approaching
    # After approach, pre-shape before descend:
    # set_gripper_from_plan(plan) is called just before tracking descend

    times = plan_intercept_times()
    T_app = times["T_approach"]
    T_des = times["T_descend"]

    print(f"[plan] T_app={T_app:.2f} T_des={T_des:.2f} "
          f"v_belt={direction * belt_speed:.2f}")

    # ========== APPROACH ==========
    aim = aim_from_box(get_box_pos_sense(), t_lead=T_app + T_des, use_bias=True, xy_offset=xy_off)
    above = np.array([-aim[0], -aim[1], APPROACH_Z])
    print(f"[moving] approach aim_world={aim}, cmd={above}")
    ok = move_pinch_to(
        viewer, above,
        duration=T_app,
        label="approach_moving",
        do_polish=False,
        success_mode="target",
    )
    if not ok:
        print("Approach failed")
        return False

    # Pre-shape gripper to object width
    set_gripper_from_plan(plan)
    hold(viewer, 0.15)

    # Optional wrist (comment out if it destabilizes your setup)

    # ========== TRACKING DESCEND ==========
    n_steps = max(1, int(T_des / model.opt.timestep))
    replan_every = max(5, n_steps // 8)

    for k in range(n_steps):
        t_rem = (n_steps - k) * model.opt.timestep
        aim = aim_from_box(get_box_pos_sense(), t_lead=t_rem, use_bias=True, xy_offset=xy_off)
        alpha = (k + 1) / n_steps
        z = (1 - alpha) * APPROACH_Z + alpha * GRASP_Z
        target = np.array([-aim[0], -aim[1], z])

        if k % replan_every == 0 or k == n_steps - 1:
            q_goal = solve_ik(target, orientation=None)
            if q_goal is not None:
                q0 = get_joint_qpos()
                for i in range(6):
                    d = q_goal[i] - q0[i]
                    q_goal[i] = q0[i] + (d + np.pi) % (2 * np.pi) - np.pi
                set_arm_ctrl(q_goal)
        step_sim(viewer)

    # ========== POLISH ==========
    v = get_box_vel_world()
    x_lead = 0.12
    center = get_box_pos_sense()[:2] + xy_off
    center[0] += v[0] * x_lead
    polish_tcp_xy(viewer, center, tol=0.012, max_iters=100)

    debug_y_offset()

    # ========== CLOSE + MATCH ==========
    close_gripper()
    T_match_fast = 0.15
    n_match = max(1, int(T_match_fast / model.opt.timestep))
    for k in range(n_match):
        t_rem = (n_match - k) * model.opt.timestep
        aim = aim_from_box(get_box_pos_sense(), t_lead=t_rem, use_bias=True, xy_offset=xy_off)
        target = np.array([-aim[0], -aim[1], GRASP_Z])
        q_goal = solve_ik(target, orientation=None)
        if q_goal is not None:
            set_arm_ctrl(q_goal)
        step_sim(viewer)


    p = get_pinch_pos()
    lift = np.array([-p[0], -p[1], LIFT_Z])
    move_pinch_to(viewer, lift, duration=0.55, label="lift",
                  do_polish=False, success_mode="target")

    return get_box_pos()[2] > (BELT_Z + plan["object_height"] + 0.05)

# def main():
#     global direction,cooldown
#     set_belt_direction(direction)
#     open_gripper()
#     if model.nkey > 0:
#         mujoco.mj_resetDataKeyframe(model, data, 0)
#     else:
#         mujoco.mj_resetData(model,data)
#
#     starting_box()
#
#     for i, name in enumerate(ARM_JOINTS):
#         data.joint(name).qpos[0] = HOME_Q[i]
#
#     mujoco.mj_forward(model, data)
#     print("Box:", get_box_pos())
#
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         print("Start, Belt speed:",belt_speed)
#
#         move_to_q(viewer,HOME_Q,duration=1)
#         open_gripper()
#
#         mujoco.mj_forward(model, data)
#
#         if not wait_for_box_at_camera_then_intercept(viewer):
#             return
#
#         print(f"[ready] h={_last_object_height:.4f}  track={_track_xy}")
#         feats = get_perception_features()
#         if feats is None or not feats.get("quality_ok", False):
#             print("[PHASE1/2] no features at go-pick")
#             return
#         print("[PHASE1 ready]",
#               f"h={feats['height']:.4f}",
#               f"L={feats['length']:.3f}",
#               f"W={feats['width']:.3f}",
#               f"yaw_deg={np.rad2deg(feats['yaw']):.1f}",
#               f"aspect={feats['aspect']:.2f}",
#               f"circ={feats['circularity']:.2f}")
#
#
#
#         plan = plan_grasp(feats)
#         success = safe_pick_box_moving(viewer)
#         if success:
#             move_to_q(viewer, PLACE_Q,duration=1)
#             open_gripper()
#             print("Dropped")
#         move_to_q(viewer, HOME_Q, duration=1.2)
#         open_gripper()
#         set_belt_direction(direction)
#         print("Done")
#
#
#
#         while viewer.is_running():
#             step_sim(viewer)
#         cv2.destroyAllWindows()

def main():
    global direction, cooldown, belt_speed, PHASE4_MODE, XY_BIAS, XY_BIAS_NOMINAL

    open_gripper()
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)

    for i, name in enumerate(ARM_JOINTS):
        data.joint(name).qpos[0] = HOME_Q[i]
    mujoco.mj_forward(model, data)

    PHASE4_MODE = True
    PHASE4_ALLOW_REVERSE = False

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("========== EVALUATION ==========")
        print(f"Spawn always x={PHASE4_SPAWN_X}, direction=-1, speed={PHASE4_BELT_SPEED}")
        rows = []

        starting_box()

        for class_name, size_xyz, rgba in PHASE4_SUITE:
            for trial_idx in range(PHASE4_TRIALS_PER_OBJ):
                print(f"\n=== {class_name}  trial {trial_idx + 1}/{PHASE4_TRIALS_PER_OBJ} ===")

                # --- scene reset only (not bias) ---
                reset_perception()          # vision track only
                cooldown = 0.0
                open_gripper()
                XY_BIAS = XY_BIAS_NOMINAL.copy()

                for i, name in enumerate(ARM_JOINTS):
                    data.joint(name).qpos[0] = HOME_Q[i]
                mujoco.mj_forward(model, data)

                move_to_q(viewer, HOME_Q, duration=0.8)
                open_gripper()

                # --- fixed belt: toward middle ---
                belt_speed = float(PHASE4_BELT_SPEED)
                direction = -1.0
                set_belt_direction(direction)

                # --- one spawn per trial ---
                yaw_deg = spawn_box_phase4(size_xyz, rgba)
                print(f"[PHASE4] class={class_name} size={size_xyz} "
                      f"pos={get_box_pos()[:3]} dir={direction:.0f} "
                      f"v={direction * belt_speed:.2f} yaw_deg={yaw_deg:.1f}")

                row = {
                    "class": class_name,
                    "size": size_xyz,
                    "belt_speed": belt_speed,
                    "direction": direction,
                    "yaw_deg": yaw_deg,
                    "success": False,
                    "xy_err_close": None,
                    "t_pick": None,
                    "height": None,
                    "L": None,
                    "W": None,
                    "aspect": None,
                    "fail": None,
                }

                t0 = time.time()
                waited_ok = False
                success = False
                xy_err = None
                feats = None

                try:
                    # EXACT same process as single-pick main
                    waited_ok = wait_for_box_at_camera_then_intercept(
                        viewer, timeout=PHASE4_TIMEOUT_WAIT
                    )
                    feats = get_perception_features()
                    if feats is not None:
                        row["height"] = feats.get("height")
                        row["L"] = feats.get("length")
                        row["W"] = feats.get("width")
                        row["aspect"] = feats.get("aspect")

                    if waited_ok and feats is not None and feats.get("quality_ok", False):
                        success = bool(safe_pick_box_moving(viewer))
                        xy_err = measure_xy_err_at_close()
                        h = float(feats.get("height", 0.05))
                        lifted = get_box_pos()[2] > (BELT_Z + h + 0.04)
                        success = bool(success and lifted)
                except Exception as e:
                    print(f"[PHASE4] exception: {e}")
                    traceback.print_exc()
                    row["fail"] = f"exception:{type(e).__name__}"
                    success = False

                t_pick = time.time() - t0
                row["success"] = success
                row["xy_err_close"] = xy_err
                row["t_pick"] = t_pick
                if row["fail"] is None:
                    row["fail"] = classify_failure(feats, success, xy_err, waited_ok)

                print(f"[PHASE4] result success={success} xy_err={xy_err} "
                      f"t={t_pick:.2f}s tag={row['fail']}  XY_BIAS={XY_BIAS}")

                # natural finish (same idea as your single-pick main)
                if success:
                    move_to_q(viewer, PLACE_Q, duration=0.8)
                    open_gripper()
                    hold(viewer, 1.5)
                    print("Dropped")
                else:
                    open_gripper()
                open_gripper()
                reset_perception()
                XY_BIAS = XY_BIAS_NOMINAL.copy()

                rows.append(row)

        summarize_phase4(rows)
        PHASE4_MODE = False
        while viewer.is_running():
            step_sim(viewer)
        SHOW_CAMERA_VIEW = False  # stop imshow in step_sim
        try:
            cv2.destroyAllWindows()
            for _ in range(5):
                cv2.waitKey(1)
        except Exception:
            pass
        # exiting the `with viewer` context closes the MuJoCo window

    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception:
        pass

if __name__ == "__main__":
    main()



""" 
Iterations:
August 7th, 3:40pm - pick up static box
August 10th, 2:39pm - pick up moving box at end of belt
August 10th, 6:07pm - pick up moving box near middle of belt
August 11th, 4:49pm - pick up moving box near middle of belt with camera vision under low belt speeds
August 13th, 10:57am - benchmark
August 17th, 5:13pm - works better - with implemented YOLO
August 18th, 5:48pm - working yaw
August 19th, 6:23 pm - working with different objects - same central orientation
August 20th, 6:00pm - final iteration
"""