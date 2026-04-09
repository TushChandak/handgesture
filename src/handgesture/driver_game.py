from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

LANE_COUNT = 3
BODY_SMOOTHING = 0.14
LANE_SWITCH_MARGIN_RATIO = 0.16
LANE_SWITCH_HOLD_SECONDS = 0.18
START_HOLD_SECONDS = 0.85
COUNTDOWN_SECONDS = 3.0
ROUND_SECONDS = 25.0
ROAD_SPEED = 360.0
PLAYER_Y_RATIO = 0.82
PLAYER_WIDTH = 110
PLAYER_HEIGHT = 150
OBSTACLE_SIZE = 116
SPAWN_DELAY_MIN = 0.75
SPAWN_DELAY_MAX = 1.20


@dataclass
class ObstacleSprite:
    name: str
    image: np.ndarray


@dataclass
class DrivingObstacle:
    lane: int
    y: float
    speed: float
    sprite: ObstacleSprite


@dataclass
class TrackingInfo:
    body_x: float | None
    detected: bool
    bbox: tuple[int, int, int, int] | None


@dataclass
class DrivingState:
    phase: str = "waiting"
    target_lane: int = 1
    player_lane: int = 1
    stable_lane: int | None = None
    stable_lane_since: float = 0.0
    pending_lane: int | None = None
    pending_lane_since: float = 0.0
    hold_progress: float = 0.0
    countdown_started_at: float = 0.0
    round_started_at: float = 0.0
    result_until: float = 0.0
    message: str = "Stand in the glowing lane to start"
    score: int = 0
    best_score: int = 0
    distance: float = 0.0
    road_offset: float = 0.0
    smoothed_x: float | None = None
    detected: bool = False
    last_spawn_at: float = 0.0
    next_spawn_delay: float = 1.0
    obstacles: list[DrivingObstacle] = field(default_factory=list)


def play_sound(kind: str) -> None:
    if winsound is None:
        return
    try:
        if kind == "start":
            winsound.Beep(620, 100)
            winsound.Beep(760, 110)
            winsound.Beep(940, 130)
        elif kind == "crash":
            winsound.Beep(420, 180)
            winsound.Beep(320, 220)
        elif kind == "win":
            winsound.Beep(760, 90)
            winsound.Beep(940, 100)
            winsound.Beep(1120, 150)
        else:
            winsound.Beep(880, 55)
    except RuntimeError:
        pass


def launch_sound(kind: str) -> None:
    threading.Thread(target=play_sound, args=(kind,), daemon=True).start()


def save_frame(frame) -> Path:
    output_dir = Path("captures")
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"driver-frame-{timestamp}.png"
    cv2.imwrite(str(output_path), frame)
    return output_path


def lane_name(lane: int) -> str:
    return ["left", "center", "right"][lane]


def road_bounds(frame_width: int) -> tuple[int, int]:
    return int(frame_width * 0.22), int(frame_width * 0.78)


def lane_center_x(frame_width: int, lane: int) -> int:
    road_left, road_right = road_bounds(frame_width)
    lane_width = (road_right - road_left) / LANE_COUNT
    return int(road_left + lane_width * (lane + 0.5))


def lane_from_body_x(body_x: float, frame_width: int) -> int:
    road_left, road_right = road_bounds(frame_width)
    clamped_x = min(max(body_x, road_left), road_right - 1)
    lane_width = max(1.0, (road_right - road_left) / LANE_COUNT)
    return min(LANE_COUNT - 1, int((clamped_x - road_left) / lane_width))


def choose_next_target(previous_lane: int | None = None) -> int:
    choices = [lane for lane in range(LANE_COUNT) if lane != previous_lane]
    return random.choice(choices or [1])


def asset_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "obstacles"


def load_obstacle_sprites() -> list[ObstacleSprite]:
    sprite_dir = asset_dir()
    filenames = [
        "traffic-cone.png",
        "rock.png",
        "wood-crate.png",
        "puddle.png",
        "broken-cart.png",
        "tire-stack.png",
    ]
    sprites: list[ObstacleSprite] = []
    for filename in filenames:
        path = sprite_dir / filename
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Missing obstacle sprite: {path}")
        resized = cv2.resize(image, (OBSTACLE_SIZE, OBSTACLE_SIZE), interpolation=cv2.INTER_AREA)
        sprites.append(ObstacleSprite(name=path.stem, image=resized))
    return sprites


def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load face detector from {cascade_path}")
    return detector


def detect_body_position(
    frame: np.ndarray,
    detector: cv2.CascadeClassifier,
    previous_smoothed_x: float | None,
) -> TrackingInfo:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(detections) == 0:
        return TrackingInfo(body_x=None, detected=False, bbox=None)

    x, y, w, h = max(detections, key=lambda item: item[2] * item[3])
    center_x = x + w / 2.0
    if previous_smoothed_x is None:
        smoothed_x = center_x
    else:
        smoothed_x = (1.0 - BODY_SMOOTHING) * previous_smoothed_x + BODY_SMOOTHING * center_x
    return TrackingInfo(body_x=smoothed_x, detected=True, bbox=(int(x), int(y), int(w), int(h)))


def alpha_blit(frame: np.ndarray, sprite: np.ndarray, top_left_x: int, top_left_y: int) -> None:
    sprite_h, sprite_w = sprite.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, top_left_x)
    y1 = max(0, top_left_y)
    x2 = min(frame_w, top_left_x + sprite_w)
    y2 = min(frame_h, top_left_y + sprite_h)
    if x1 >= x2 or y1 >= y2:
        return

    sprite_x1 = x1 - top_left_x
    sprite_y1 = y1 - top_left_y
    sprite_x2 = sprite_x1 + (x2 - x1)
    sprite_y2 = sprite_y1 + (y2 - y1)

    roi = frame[y1:y2, x1:x2]
    sprite_roi = sprite[sprite_y1:sprite_y2, sprite_x1:sprite_x2]
    if sprite_roi.shape[2] < 4:
        roi[:] = sprite_roi[:, :, :3]
        return

    alpha = sprite_roi[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (sprite_roi[:, :, :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def draw_background(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    for y in range(height):
        blend = y / max(1, height - 1)
        sky = np.array([245, 232, 190], dtype=np.float32)
        horizon = np.array([128, 184, 255], dtype=np.float32)
        frame[y, :, :] = ((1.0 - blend) * sky + blend * horizon).astype(np.uint8)

    cv2.circle(frame, (width - 140, 100), 55, (70, 220, 255), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (160, 120), (78, 30), 0, 0, 360, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (260, 145), (90, 34), 0, 0, 360, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(frame, (950, 145), (100, 36), 0, 0, 360, (255, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (0, int(height * 0.76)), (width, height), (78, 170, 88), -1)


def draw_road(frame: np.ndarray, state: DrivingState) -> None:
    height, width = frame.shape[:2]
    road_left, road_right = road_bounds(width)
    cv2.rectangle(frame, (road_left, 0), (road_right, height), (62, 62, 66), -1)
    cv2.rectangle(frame, (road_left - 18, 0), (road_left, height), (215, 228, 248), -1)
    cv2.rectangle(frame, (road_right, 0), (road_right + 18, height), (215, 228, 248), -1)

    lane_width = (road_right - road_left) / LANE_COUNT
    for lane in range(1, LANE_COUNT):
        x = int(road_left + lane_width * lane)
        stripe_y = -90 + int(state.road_offset % 110.0)
        while stripe_y < height:
            cv2.line(frame, (x, stripe_y), (x, min(height, stripe_y + 60)), (242, 242, 242), 6, lineType=cv2.LINE_AA)
            stripe_y += 110

    if state.phase in {"waiting", "countdown"}:
        lane_x0 = int(road_left + lane_width * state.target_lane)
        lane_x1 = int(lane_x0 + lane_width)
        overlay = frame.copy()
        cv2.rectangle(overlay, (lane_x0 + 16, 0), (lane_x1 - 16, height), (72, 220, 255), -1)
        cv2.addWeighted(overlay, 0.16, frame, 0.84, 0.0, frame)


def draw_player_car(frame: np.ndarray, lane: int) -> tuple[int, int]:
    height, width = frame.shape[:2]
    center_x = lane_center_x(width, lane)
    center_y = int(height * PLAYER_Y_RATIO)
    half_w = PLAYER_WIDTH // 2
    half_h = PLAYER_HEIGHT // 2

    body = np.array(
        [
            (center_x - half_w, center_y + half_h - 10),
            (center_x - half_w + 14, center_y - half_h + 28),
            (center_x - 26, center_y - half_h),
            (center_x + 26, center_y - half_h),
            (center_x + half_w - 14, center_y - half_h + 28),
            (center_x + half_w, center_y + half_h - 10),
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(frame, body, (35, 45, 235), lineType=cv2.LINE_AA)
    cv2.polylines(frame, [body], True, (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (center_x - 28, center_y - 44), (center_x + 28, center_y + 8), (250, 224, 165), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(frame, (center_x - 28, center_y - 44), (center_x + 28, center_y + 8), (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.line(frame, (center_x, center_y - 44), (center_x, center_y + 8), (255, 255, 255), 2, lineType=cv2.LINE_AA)

    for wheel_x in (center_x - 42, center_x + 42):
        cv2.circle(frame, (wheel_x, center_y - 30), 12, (24, 24, 24), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (wheel_x, center_y + 42), 12, (24, 24, 24), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (wheel_x, center_y - 30), 5, (185, 185, 185), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (wheel_x, center_y + 42), 5, (185, 185, 185), -1, lineType=cv2.LINE_AA)

    cv2.circle(frame, (center_x - 22, center_y - 46), 9, (180, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (center_x + 22, center_y - 46), 9, (180, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (center_x - 24, center_y + 54), 8, (55, 55, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (center_x + 24, center_y + 54), 8, (55, 55, 255), -1, lineType=cv2.LINE_AA)
    return center_x, center_y


def draw_obstacle(frame: np.ndarray, obstacle: DrivingObstacle) -> None:
    center_x = lane_center_x(frame.shape[1], obstacle.lane)
    top_left_x = int(center_x - obstacle.sprite.image.shape[1] / 2)
    top_left_y = int(obstacle.y - obstacle.sprite.image.shape[0] / 2)
    alpha_blit(frame, obstacle.sprite.image, top_left_x, top_left_y)


def draw_tracking_preview(
    frame: np.ndarray,
    camera_frame: np.ndarray,
    tracking: TrackingInfo,
    state: DrivingState,
) -> None:
    preview = cv2.resize(camera_frame, (240, 135), interpolation=cv2.INTER_AREA)
    for lane in range(1, LANE_COUNT):
        x = int(preview.shape[1] * lane / LANE_COUNT)
        cv2.line(preview, (x, 0), (x, preview.shape[0]), (70, 220, 255), 1, lineType=cv2.LINE_AA)

    if tracking.bbox is not None:
        x, y, w, h = tracking.bbox
        scale_x = preview.shape[1] / camera_frame.shape[1]
        scale_y = preview.shape[0] / camera_frame.shape[0]
        pt1 = (int(x * scale_x), int(y * scale_y))
        pt2 = (int((x + w) * scale_x), int((y + h) * scale_y))
        cv2.rectangle(preview, pt1, pt2, (90, 255, 140), 2, lineType=cv2.LINE_AA)

    if tracking.body_x is not None and camera_frame.shape[1] > 0:
        marker_x = int((tracking.body_x / camera_frame.shape[1]) * preview.shape[1])
        cv2.circle(preview, (marker_x, preview.shape[0] // 2), 8, (75, 255, 140), -1, lineType=cv2.LINE_AA)
        cv2.circle(preview, (marker_x, preview.shape[0] // 2), 12, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    x0 = frame.shape[1] - preview.shape[1] - 20
    y0 = 20
    frame[y0 : y0 + preview.shape[0], x0 : x0 + preview.shape[1]] = preview
    cv2.rectangle(frame, (x0, y0), (x0 + preview.shape[1], y0 + preview.shape[0]), (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "camera", (x0 + 8, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"target: {lane_name(state.target_lane)}", (x0 + 8, y0 + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_centered_text(
    frame: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(12, (frame.shape[1] - text_width) // 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, lineType=cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, state: DrivingState, fps: float, now: float) -> None:
    cv2.putText(frame, f"FPS: {fps:.1f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"Dodges: {state.score}", (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.80, (255, 242, 170), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"Best: {state.best_score}", (16, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"Distance: {state.distance:.0f} m", (16, 134), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Lane: {lane_name(state.player_lane).title()}",
        (16, 168),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (82, 222, 255),
        2,
        lineType=cv2.LINE_AA,
    )

    detection_color = (80, 255, 140) if state.detected else (90, 140, 255)
    detection_text = "player tracked" if state.detected else "step into view"
    cv2.putText(frame, detection_text, (16, frame.shape[0] - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.65, detection_color, 2, lineType=cv2.LINE_AA)
    cv2.putText(frame, "Press Q to quit | S to save frame", (16, frame.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    draw_centered_text(frame, state.message, 40, 0.88, (255, 255, 255), 2)

    if state.phase == "waiting":
        draw_centered_text(frame, f"Target lane: {lane_name(state.target_lane).title()}", 78, 0.92, (90, 225, 255), 2)
        bar_width = 280
        x0 = (frame.shape[1] - bar_width) // 2
        y0 = 92
        cv2.rectangle(frame, (x0, y0), (x0 + bar_width, y0 + 22), (255, 255, 255), 2, lineType=cv2.LINE_AA)
        fill_width = int((bar_width - 4) * state.hold_progress)
        if fill_width > 0:
            cv2.rectangle(frame, (x0 + 2, y0 + 2), (x0 + 2 + fill_width, y0 + 20), (90, 225, 255), -1, lineType=cv2.LINE_AA)
    elif state.phase == "countdown":
        elapsed = now - state.countdown_started_at
        countdown_value = max(1, int(COUNTDOWN_SECONDS - elapsed) + 1)
        draw_centered_text(frame, str(countdown_value), frame.shape[0] // 2, 3.2, (90, 220, 255), 6)
    elif state.phase == "playing":
        remaining = max(0.0, ROUND_SECONDS - (now - state.round_started_at))
        cv2.putText(frame, f"Time: {remaining:0.1f}s", (16, 202), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    elif state.phase == "crashed":
        draw_centered_text(frame, "Bump!", frame.shape[0] // 2 - 10, 1.7, (110, 130, 255), 3)
        draw_centered_text(frame, "Move back to the start lane to try again", frame.shape[0] // 2 + 28, 0.82, (255, 255, 255), 2)
    elif state.phase == "win":
        draw_centered_text(frame, "Great driving!", frame.shape[0] // 2 - 10, 1.55, (90, 240, 255), 3)
        draw_centered_text(frame, f"You dodged {state.score} obstacles", frame.shape[0] // 2 + 28, 0.90, (255, 255, 255), 2)


def reset_to_waiting(state: DrivingState, now: float, keep_best: bool = True) -> None:
    best_score = state.best_score if keep_best else 0
    state.phase = "waiting"
    state.target_lane = choose_next_target(state.target_lane)
    state.player_lane = 1
    state.stable_lane = None
    state.stable_lane_since = now
    state.pending_lane = None
    state.pending_lane_since = 0.0
    state.hold_progress = 0.0
    state.countdown_started_at = 0.0
    state.round_started_at = 0.0
    state.result_until = 0.0
    state.score = 0
    state.best_score = best_score
    state.distance = 0.0
    state.road_offset = 0.0
    state.last_spawn_at = now
    state.next_spawn_delay = random.uniform(SPAWN_DELAY_MIN, SPAWN_DELAY_MAX)
    state.obstacles.clear()
    state.message = f"Stand in the {lane_name(state.target_lane)} lane to start"


def start_round(state: DrivingState, now: float) -> None:
    state.phase = "playing"
    state.round_started_at = now
    state.road_offset = 0.0
    state.distance = 0.0
    state.score = 0
    state.obstacles.clear()
    state.last_spawn_at = now
    state.next_spawn_delay = 0.80
    state.message = "Drive left, center, and right to dodge"


def finish_round(state: DrivingState, now: float, phase: str, message: str) -> None:
    state.phase = phase
    state.result_until = now + 2.6
    state.best_score = max(state.best_score, state.score)
    state.message = message
    state.obstacles.clear()


def lane_change_candidate(state: DrivingState, tracked_x: float, frame_width: int) -> int:
    road_left, road_right = road_bounds(frame_width)
    lane_width = (road_right - road_left) / LANE_COUNT
    margin = lane_width * LANE_SWITCH_MARGIN_RATIO
    left_boundary = road_left + lane_width
    right_boundary = road_left + lane_width * 2
    current_lane = state.player_lane

    if current_lane == 0:
        return 1 if tracked_x >= left_boundary + margin else 0
    if current_lane == 2:
        return 1 if tracked_x <= right_boundary - margin else 2
    if tracked_x <= left_boundary - margin:
        return 0
    if tracked_x >= right_boundary + margin:
        return 2
    return 1


def maybe_update_player_lane(state: DrivingState, tracked_x: float | None, frame_width: int, now: float) -> None:
    if tracked_x is None:
        state.hold_progress = 0.0
        state.pending_lane = None
        return

    candidate_lane = lane_change_candidate(state, tracked_x, frame_width)
    if candidate_lane != state.player_lane:
        if state.pending_lane != candidate_lane:
            state.pending_lane = candidate_lane
            state.pending_lane_since = now
        elif now - state.pending_lane_since >= LANE_SWITCH_HOLD_SECONDS:
            state.player_lane = candidate_lane
            state.pending_lane = None
            state.stable_lane = candidate_lane
            state.stable_lane_since = now
    else:
        state.pending_lane = None
        if state.stable_lane != state.player_lane:
            state.stable_lane = state.player_lane
            state.stable_lane_since = now

    if state.phase == "waiting" and state.player_lane == state.target_lane:
        hold_time = max(0.0, now - state.stable_lane_since)
        state.hold_progress = min(1.0, hold_time / START_HOLD_SECONDS)
    elif state.phase == "waiting":
        state.hold_progress = 0.0


def spawn_obstacle(state: DrivingState, sprites: list[ObstacleSprite]) -> None:
    boost = min(140.0, state.score * 7.0)
    obstacle = DrivingObstacle(
        lane=random.randint(0, LANE_COUNT - 1),
        y=-OBSTACLE_SIZE,
        speed=ROAD_SPEED + 120.0 + random.uniform(-40.0, 60.0) + boost,
        sprite=random.choice(sprites),
    )
    state.obstacles.append(obstacle)


def update_game_state(
    state: DrivingState,
    tracked_x: float | None,
    frame_shape: tuple[int, int, int],
    now: float,
    dt: float,
    sprites: list[ObstacleSprite],
) -> None:
    maybe_update_player_lane(state, tracked_x, frame_shape[1], now)

    if state.phase == "waiting":
        if tracked_x is None:
            state.message = f"Stand in the {lane_name(state.target_lane)} lane to start"
        elif state.player_lane == state.target_lane:
            state.message = "Hold steady... starting engine"
            if state.hold_progress >= 1.0:
                state.phase = "countdown"
                state.countdown_started_at = now
                state.message = "Ready, set..."
                launch_sound("start")
        else:
            state.message = f"Move to the {lane_name(state.target_lane)} lane"
        return

    if state.phase == "countdown":
        if now - state.countdown_started_at >= COUNTDOWN_SECONDS:
            start_round(state, now)
        return

    if state.phase == "playing":
        state.road_offset += ROAD_SPEED * dt
        state.distance += dt * 16.0

        if now - state.last_spawn_at >= state.next_spawn_delay:
            spawn_obstacle(state, sprites)
            state.last_spawn_at = now
            reduction = min(0.24, state.score * 0.01)
            state.next_spawn_delay = random.uniform(max(0.55, SPAWN_DELAY_MIN - reduction), max(0.80, SPAWN_DELAY_MAX - reduction))

        car_y = int(frame_shape[0] * PLAYER_Y_RATIO)
        collision_gap = PLAYER_HEIGHT * 0.42
        for obstacle in state.obstacles[:]:
            obstacle.y += obstacle.speed * dt
            same_lane = obstacle.lane == state.player_lane
            close_y = abs(obstacle.y - car_y) <= collision_gap
            if same_lane and close_y:
                finish_round(state, now, "crashed", "Bump! Let's line up and try again")
                launch_sound("crash")
                return

            if obstacle.y - OBSTACLE_SIZE * 0.5 > frame_shape[0]:
                state.obstacles.remove(obstacle)
                state.score += 1
                state.best_score = max(state.best_score, state.score)
                state.message = random.choice(["Nice dodge!", "Great driving!", "Zoom past!", "You missed it!"])

        if now - state.round_started_at >= ROUND_SECONDS:
            finish_round(state, now, "win", "Great driving!")
            launch_sound("win")
        return

    if state.phase in {"crashed", "win"} and now >= state.result_until:
        reset_to_waiting(state, now)


def run_driver_game(camera_index: int, min_detection: float, min_tracking: float) -> None:
    _ = (min_detection, min_tracking)
    sprites = load_obstacle_sprites()
    detector = create_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}. Check webcam permissions or try another index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    state = DrivingState()
    previous_time = time.perf_counter()
    reset_to_waiting(state, previous_time, keep_best=False)

    try:
        while True:
            ok, camera_frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read a frame from the webcam.")

            camera_frame = cv2.flip(camera_frame, 1)
            now = time.perf_counter()
            dt = max(1e-3, now - previous_time)
            previous_time = now

            tracking = detect_body_position(camera_frame, detector, state.smoothed_x)
            state.smoothed_x = tracking.body_x
            state.detected = tracking.detected

            update_game_state(state, state.smoothed_x, camera_frame.shape, now, dt, sprites)

            frame = np.zeros_like(camera_frame)
            draw_background(frame)
            draw_road(frame, state)
            for obstacle in state.obstacles:
                draw_obstacle(frame, obstacle)
            draw_player_car(frame, state.player_lane)
            draw_tracking_preview(frame, camera_frame, tracking, state)
            draw_overlay(frame, state, fps=1.0 / dt, now=now)

            cv2.imshow("Motion Driver", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                saved_path = save_frame(frame)
                print(f"Saved screenshot to {saved_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()




