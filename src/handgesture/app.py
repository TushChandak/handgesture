from __future__ import annotations

import argparse
import math
import random
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmark

from handgesture.driver_game import run_driver_game

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
GROUND_MARGIN = 110
BALLOON_RADIUS = 36
BALLOON_GRAVITY = 165.0
BALLOON_FLOAT_LIFT = 20.0
BALLOON_DRAG = 0.996
BALLOON_BOUNCE = 0.72
BALLOON_MAX_SPEED = 780.0
HIT_DISTANCE = 56.0
GROUND_HIT_SOUND_MIN_SPEED = 120.0
HIT_COOLDOWN = 0.10
FACE_TRACK_DISTANCE = 180.0
FACE_SMOOTHING = 0.28
FACE_MAX_MISSING = 8


@dataclass
class BalloonState:
    x: float
    y: float
    vx: float
    vy: float
    radius: int = BALLOON_RADIUS
    hue: tuple[int, int, int] = (135, 190, 255)


@dataclass
class HandInfo:
    label: str
    anchor: tuple[int, int]
    colliders: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PopBurst:
    x: float
    y: float
    life: float
    max_life: float
    radius: float
    color: tuple[int, int, int]
    label: str


@dataclass
class FaceTrack:
    name: str
    style: str
    color: tuple[int, int, int]
    bbox: tuple[float, float, float, float] | None = None
    missing: int = 0


@dataclass
class GameState:
    score: int = 0
    best_score: int = 0
    message: str = "Tap the balloon upward"
    message_until: float = 0.0
    last_hit_time: float = 0.0



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces, track hands, and play kid-friendly camera games."
    )
    parser.add_argument("--game", choices=("balloon", "driver"), default="balloon", help="Which camera game to launch.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index to open.")
    parser.add_argument("--max-hands", type=int, default=2, help="Maximum number of hands to track.")
    parser.add_argument("--min-detection", type=float, default=0.6, help="Minimum confidence for initial hand detection.")
    parser.add_argument("--min-tracking", type=float, default=0.6, help="Minimum confidence for landmark tracking between frames.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models") / "hand_landmarker.task",
        help="Path to a MediaPipe hand landmarker task model file.",
    )
    return parser.parse_args()



def clip_speed(value: float) -> float:
    return max(-BALLOON_MAX_SPEED, min(BALLOON_MAX_SPEED, value))



def save_frame(frame) -> Path:
    output_dir = Path("captures")
    output_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"balloon-frame-{timestamp}.png"
    cv2.imwrite(str(output_path), frame)
    return output_path



def ensure_model_file(model_path: Path) -> Path:
    model_path = model_path.resolve()
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model to {model_path} ...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Failed to download the MediaPipe hand model automatically. "
            f"Download it manually from {MODEL_URL} and place it at {model_path}."
        ) from exc
    return model_path



def create_landmarker(model_path: Path, max_hands: int, min_detection: float, min_tracking: float):
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=min_detection,
        min_hand_presence_confidence=min_detection,
        min_tracking_confidence=min_tracking,
    )
    return vision.HandLandmarker.create_from_options(options)



def create_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        raise RuntimeError(f"Failed to load face detector from {cascade_path}")
    return detector



def landmark_point(hand_landmarks, landmark: HandLandmark, frame_shape) -> tuple[int, int]:
    point = hand_landmarks[landmark]
    return int(point.x * frame_shape[1]), int(point.y * frame_shape[0])



def detect_faces(frame, face_detector: cv2.CascadeClassifier) -> list[tuple[int, int, int, int]]:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    return sorted(list(faces), key=lambda item: item[0])



def update_face_tracks(tracks: list[FaceTrack], detections: list[tuple[int, int, int, int]]) -> set[int]:
    used_detection_ids: set[int] = set()
    for track in tracks:
        if track.bbox is None:
            continue
        tx, ty, tw, th = track.bbox
        track_center = (tx + tw / 2.0, ty + th / 2.0)
        best_index = None
        best_distance = float("inf")
        for index, (x, y, w, h) in enumerate(detections):
            if index in used_detection_ids:
                continue
            center = (x + w / 2.0, y + h / 2.0)
            distance = math.hypot(center[0] - track_center[0], center[1] - track_center[1])
            if distance < best_distance:
                best_distance = distance
                best_index = index
        if best_index is not None and best_distance <= FACE_TRACK_DISTANCE:
            x, y, w, h = detections[best_index]
            used_detection_ids.add(best_index)
            alpha = FACE_SMOOTHING
            track.bbox = (
                tx * (1.0 - alpha) + x * alpha,
                ty * (1.0 - alpha) + y * alpha,
                tw * (1.0 - alpha) + w * alpha,
                th * (1.0 - alpha) + h * alpha,
            )
            track.missing = 0
        else:
            track.missing += 1
            if track.missing > FACE_MAX_MISSING:
                track.bbox = None

    for index, detection in enumerate(detections):
        if index in used_detection_ids:
            continue
        for track in tracks:
            if track.bbox is None:
                track.bbox = tuple(float(value) for value in detection)
                track.missing = 0
                used_detection_ids.add(index)
                break

    return used_detection_ids



def play_sound(kind: str) -> None:
    if winsound is None:
        return
    try:
        if kind == "hit":
            winsound.Beep(880, 40)
        else:
            winsound.Beep(740, 60)
            winsound.Beep(620, 70)
    except RuntimeError:
        pass



def launch_sound(kind: str) -> None:
    threading.Thread(target=play_sound, args=(kind,), daemon=True).start()



def trigger_burst(bursts: list[PopBurst], x: float, y: float, color: tuple[int, int, int], label: str) -> None:
    bursts.append(PopBurst(x=x, y=y, life=0.45, max_life=0.45, radius=16.0, color=color, label=label))



def update_bursts(bursts: list[PopBurst], dt: float) -> None:
    for burst in bursts[:]:
        burst.life -= dt
        burst.radius += 220.0 * dt
        if burst.life <= 0:
            bursts.remove(burst)



def draw_bursts(frame, bursts: list[PopBurst]) -> None:
    for burst in bursts:
        strength = burst.life / burst.max_life
        center = (int(burst.x), int(burst.y))
        radius = int(burst.radius)
        color = tuple(int(channel * (0.45 + 0.55 * strength)) for channel in burst.color)
        cv2.circle(frame, center, radius, color, max(2, int(5 * strength)))
        for angle_deg in range(0, 360, 36):
            radians = math.radians(angle_deg)
            inner = (int(center[0] + math.cos(radians) * radius * 0.35), int(center[1] + math.sin(radians) * radius * 0.35))
            outer = (int(center[0] + math.cos(radians) * radius * 1.05), int(center[1] + math.sin(radians) * radius * 1.05))
            cv2.line(frame, inner, outer, color, max(1, int(3 * strength)))
        cv2.putText(frame, burst.label, (center[0] - 24, center[1] - radius - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

def draw_bluey_overlay(frame, x: int, y: int, w: int, h: int) -> None:
    outline = (28, 24, 24)
    dark_blue = (106, 105, 152)
    light_blue = (206, 224, 239)
    cream = (173, 220, 245)
    nose_color = (78, 81, 108)

    cx = x + w // 2
    top = max(0, y - int(h * 0.30))
    head_center = (cx, y + int(h * 0.55))
    head_axes = (int(w * 0.72), int(h * 0.95))

    left_ear = np.array([(cx - int(w * 0.82), y + int(h * 0.18)), (cx - int(w * 0.42), top), (cx - int(w * 0.10), y + int(h * 0.34))], dtype=np.int32)
    right_ear = np.array([(cx + int(w * 0.82), y + int(h * 0.18)), (cx + int(w * 0.42), top), (cx + int(w * 0.10), y + int(h * 0.34))], dtype=np.int32)
    cv2.fillConvexPoly(frame, left_ear, dark_blue)
    cv2.fillConvexPoly(frame, right_ear, dark_blue)
    cv2.polylines(frame, [left_ear], True, outline, 5)
    cv2.polylines(frame, [right_ear], True, outline, 5)

    left_inner = np.array([(cx - int(w * 0.54), y + int(h * 0.18)), (cx - int(w * 0.38), y + int(h * 0.02)), (cx - int(w * 0.18), y + int(h * 0.34))], dtype=np.int32)
    right_inner = np.array([(cx + int(w * 0.54), y + int(h * 0.18)), (cx + int(w * 0.38), y + int(h * 0.02)), (cx + int(w * 0.18), y + int(h * 0.34))], dtype=np.int32)
    cv2.fillConvexPoly(frame, left_inner, cream)
    cv2.fillConvexPoly(frame, right_inner, cream)

    cv2.ellipse(frame, head_center, head_axes, 0, 0, 360, dark_blue, -1)
    cv2.ellipse(frame, head_center, head_axes, 0, 0, 360, outline, 5)
    cv2.ellipse(frame, (cx, y + int(h * 0.22)), (int(w * 0.45), int(h * 0.22)), 0, 180, 360, light_blue, -1)
    cv2.ellipse(frame, (cx - int(w * 0.26), y + int(h * 0.26)), (int(w * 0.17), int(h * 0.12)), 0, 200, 20, light_blue, -1)
    cv2.ellipse(frame, (cx + int(w * 0.26), y + int(h * 0.26)), (int(w * 0.17), int(h * 0.12)), 0, 160, 340, light_blue, -1)
    cv2.ellipse(frame, (cx, y + int(h * 0.92)), (int(w * 0.52), int(h * 0.40)), 0, 0, 360, cream, -1)
    cv2.ellipse(frame, (cx - int(w * 0.18), y + int(h * 1.02)), (int(w * 0.20), int(h * 0.26)), 0, 30, 210, light_blue, -1)
    cv2.ellipse(frame, (cx + int(w * 0.32), y + int(h * 0.92)), (int(w * 0.18), int(h * 0.22)), 0, 180, 360, dark_blue, -1)

    eye_y = y + int(h * 0.62)
    left_eye = (cx - int(w * 0.22), eye_y)
    right_eye = (cx + int(w * 0.18), eye_y)
    eye_axes = (int(w * 0.17), int(h * 0.33))
    cv2.ellipse(frame, left_eye, eye_axes, 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, right_eye, eye_axes, 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, left_eye, eye_axes, 0, 0, 360, outline, 4)
    cv2.ellipse(frame, right_eye, eye_axes, 0, 0, 360, outline, 4)
    pupil_axes = (max(6, int(w * 0.045)), max(10, int(h * 0.10)))
    cv2.ellipse(frame, (left_eye[0] + int(w * 0.02), eye_y + int(h * 0.02)), pupil_axes, 0, 0, 360, outline, -1)
    cv2.ellipse(frame, (right_eye[0] - int(w * 0.02), eye_y + int(h * 0.02)), pupil_axes, 0, 0, 360, outline, -1)
    for px, py in [(left_eye[0] + int(w * 0.02), eye_y + int(h * 0.02)), (right_eye[0] - int(w * 0.02), eye_y + int(h * 0.02))]:
        cv2.circle(frame, (px - 4, py - 6), 3, (255, 255, 255), -1)
        cv2.circle(frame, (px + 4, py + 2), 2, (255, 255, 255), -1)

    nose = np.array([(cx, y + int(h * 0.78)), (cx - int(w * 0.12), y + int(h * 0.98)), (cx, y + int(h * 1.14)), (cx + int(w * 0.12), y + int(h * 0.98))], dtype=np.int32)
    cv2.fillConvexPoly(frame, nose, nose_color)
    cv2.polylines(frame, [nose], True, outline, 4)
    cv2.ellipse(frame, (cx - int(w * 0.05), y + int(h * 1.28)), (int(w * 0.16), int(h * 0.10)), 15, 205, 335, (136, 126, 55), 4)



def draw_bingo_overlay(frame, x: int, y: int, w: int, h: int) -> None:
    outline = (60, 50, 45)
    orange = (115, 170, 225)
    cream = (220, 235, 246)
    nose_color = (70, 63, 60)

    cx = x + w // 2
    top = max(0, y - int(h * 0.28))
    head_center = (cx, y + int(h * 0.58))
    head_axes = (int(w * 0.72), int(h * 0.90))

    left_ear = np.array([(cx - int(w * 0.78), y + int(h * 0.18)), (cx - int(w * 0.42), top), (cx - int(w * 0.12), y + int(h * 0.30))], dtype=np.int32)
    right_ear = np.array([(cx + int(w * 0.78), y + int(h * 0.18)), (cx + int(w * 0.42), top), (cx + int(w * 0.12), y + int(h * 0.30))], dtype=np.int32)
    cv2.fillConvexPoly(frame, left_ear, orange)
    cv2.fillConvexPoly(frame, right_ear, orange)
    cv2.polylines(frame, [left_ear], True, outline, 4)
    cv2.polylines(frame, [right_ear], True, outline, 4)

    left_inner = np.array([(cx - int(w * 0.52), y + int(h * 0.18)), (cx - int(w * 0.36), y + int(h * 0.04)), (cx - int(w * 0.18), y + int(h * 0.30))], dtype=np.int32)
    right_inner = np.array([(cx + int(w * 0.52), y + int(h * 0.18)), (cx + int(w * 0.36), y + int(h * 0.04)), (cx + int(w * 0.18), y + int(h * 0.30))], dtype=np.int32)
    cv2.fillConvexPoly(frame, left_inner, cream)
    cv2.fillConvexPoly(frame, right_inner, cream)

    cv2.ellipse(frame, head_center, head_axes, 0, 0, 360, orange, -1)
    cv2.ellipse(frame, head_center, head_axes, 0, 0, 360, outline, 4)
    cv2.ellipse(frame, (cx, y + int(h * 0.28)), (int(w * 0.30), int(h * 0.16)), 0, 180, 360, cream, -1)
    cv2.ellipse(frame, (cx, y + int(h * 0.98)), (int(w * 0.48), int(h * 0.34)), 0, 0, 360, cream, -1)
    cv2.ellipse(frame, (cx - int(w * 0.30), y + int(h * 0.68)), (int(w * 0.18), int(h * 0.44)), 0, 250, 110, cream, -1)
    cv2.ellipse(frame, (cx + int(w * 0.32), y + int(h * 0.68)), (int(w * 0.14), int(h * 0.34)), 0, 70, 290, cream, -1)
    cv2.ellipse(frame, (cx + int(w * 0.18), y + int(h * 0.78)), (int(w * 0.18), int(h * 0.16)), 0, 180, 360, cream, -1)
    cv2.ellipse(frame, (cx - int(w * 0.06), y + int(h * 1.10)), (int(w * 0.24), int(h * 0.18)), 0, 40, 220, (200, 220, 245), -1)

    eye_y = y + int(h * 0.62)
    left_eye = (cx - int(w * 0.20), eye_y)
    right_eye = (cx + int(w * 0.16), eye_y)
    eye_axes = (int(w * 0.15), int(h * 0.28))
    cv2.ellipse(frame, left_eye, eye_axes, 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, right_eye, eye_axes, 0, 0, 360, (255, 255, 255), -1)
    cv2.ellipse(frame, left_eye, eye_axes, 0, 0, 360, outline, 3)
    cv2.ellipse(frame, right_eye, eye_axes, 0, 0, 360, outline, 3)
    pupil_axes = (max(6, int(w * 0.04)), max(10, int(h * 0.09)))
    cv2.ellipse(frame, (left_eye[0] + int(w * 0.02), eye_y + int(h * 0.01)), pupil_axes, 0, 0, 360, (30, 30, 35), -1)
    cv2.ellipse(frame, (right_eye[0] - int(w * 0.02), eye_y + int(h * 0.01)), pupil_axes, 0, 0, 360, (30, 30, 35), -1)
    for px, py in [(left_eye[0] + int(w * 0.02), eye_y + int(h * 0.01)), (right_eye[0] - int(w * 0.02), eye_y + int(h * 0.01))]:
        cv2.circle(frame, (px + 3, py - 5), 2, (255, 255, 255), -1)
        cv2.circle(frame, (px + 7, py + 1), 2, (230, 230, 230), -1)

    nose = np.array([(cx, y + int(h * 0.82)), (cx - int(w * 0.11), y + int(h * 0.96)), (cx, y + int(h * 1.08)), (cx + int(w * 0.11), y + int(h * 0.96))], dtype=np.int32)
    cv2.fillConvexPoly(frame, nose, nose_color)
    cv2.polylines(frame, [nose], True, outline, 3)
    cv2.ellipse(frame, (cx, y + int(h * 0.88)), (int(w * 0.05), int(h * 0.03)), 0, 0, 360, cream, -1)
    cv2.ellipse(frame, (cx - int(w * 0.10), y + int(h * 1.22)), (int(w * 0.10), int(h * 0.08)), 20, 210, 340, (185, 190, 155), 3)



def draw_faces(frame, face_detector: cv2.CascadeClassifier, tracks: list[FaceTrack]) -> int:
    detections = detect_faces(frame, face_detector)
    used_detection_ids = update_face_tracks(tracks, detections)

    for track in tracks:
        if track.bbox is None:
            continue
        x, y, w, h = (int(value) for value in track.bbox)
        if track.style == "bluey":
            draw_bluey_overlay(frame, x, y, w, h)
        else:
            draw_bingo_overlay(frame, x, y, w, h)
        cv2.putText(frame, track.name, (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, track.color, 2)

    for index, (x, y, w, h) in enumerate(detections):
        if index in used_detection_ids:
            continue
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)
        cv2.putText(frame, f"Face {index + 1}", (x, max(30, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2)

    return len(detections)



def draw_hands(frame, results) -> list[HandInfo]:
    hand_infos: list[HandInfo] = []
    if not results.hand_landmarks:
        return hand_infos

    hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
    landmark_style = vision.drawing_styles.get_default_hand_landmarks_style()
    connection_style = vision.drawing_styles.get_default_hand_connections_style()

    for index, hand_landmarks in enumerate(results.hand_landmarks):
        vision.drawing_utils.draw_landmarks(frame, hand_landmarks, hand_connections, landmark_style, connection_style)
        label = f"Hand {index + 1}"
        if results.handedness and index < len(results.handedness):
            handedness = results.handedness[index][0].category_name
            if handedness:
                label = {"Left": "Right", "Right": "Left"}.get(handedness, handedness)
        wrist = landmark_point(hand_landmarks, HandLandmark.WRIST, frame.shape)
        colliders = [landmark_point(hand_landmarks, HandLandmark(landmark_index), frame.shape) for landmark_index in range(len(hand_landmarks))]
        cv2.putText(frame, label, (wrist[0], max(30, wrist[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        hand_infos.append(HandInfo(label=label, anchor=wrist, colliders=colliders))

    return hand_infos

def draw_ground(frame) -> int:
    ground_y = frame.shape[0] - GROUND_MARGIN
    cv2.line(frame, (0, ground_y), (frame.shape[1], ground_y), (90, 220, 90), 3)
    cv2.putText(frame, "Ground", (12, ground_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 220, 90), 2)
    return ground_y



def draw_balloon(frame, balloon: BalloonState, ground_y: int) -> None:
    center = (int(balloon.x), int(balloon.y))
    knot = (center[0], center[1] + balloon.radius)
    string_end = (center[0], min(ground_y - 8, center[1] + balloon.radius + 42))
    shadow = frame.copy()
    cv2.ellipse(shadow, (center[0], ground_y + 8), (balloon.radius + 10, 12), 0, 0, 360, (30, 30, 30), -1)
    cv2.addWeighted(shadow, 0.15, frame, 0.85, 0, frame)
    cv2.line(frame, knot, string_end, (220, 220, 220), 2)
    cv2.circle(frame, center, balloon.radius, balloon.hue, -1)
    cv2.circle(frame, (center[0] - 10, center[1] - 12), 10, (240, 245, 255), -1)
    cv2.circle(frame, center, balloon.radius, (255, 255, 255), 2)
    cv2.circle(frame, (center[0], center[1] + balloon.radius + 4), 4, balloon.hue, -1)



def register_hit(game_state: GameState, bursts: list[PopBurst], balloon: BalloonState, now: float) -> None:
    game_state.score += 1
    game_state.best_score = max(game_state.best_score, game_state.score)
    game_state.message = random.choice(["Great hit!", "Keepy Uppy!", "Nice tap!", "Up we go!"])
    game_state.message_until = now + 0.9
    game_state.last_hit_time = now
    trigger_burst(bursts, balloon.x, balloon.y - balloon.radius * 0.3, (150, 220, 255), "+1")
    launch_sound("hit")



def register_ground_pop(game_state: GameState, bursts: list[PopBurst], balloon: BalloonState, ground_y: int) -> None:
    game_state.best_score = max(game_state.best_score, game_state.score)
    game_state.score = 0
    game_state.message = "Boop! Try again"
    game_state.message_until = time.perf_counter() + 1.0
    trigger_burst(bursts, balloon.x, ground_y, (120, 190, 255), "POP")
    launch_sound("ground")



def update_balloon(
    balloon: BalloonState,
    dt: float,
    frame_shape,
    hand_infos: list[HandInfo],
    previous_anchors: dict[int, tuple[int, int]],
    bursts: list[PopBurst],
    game_state: GameState,
    now: float,
) -> dict[int, tuple[int, int]]:
    ground_y = frame_shape[0] - GROUND_MARGIN
    balloon.vy += (BALLOON_GRAVITY - BALLOON_FLOAT_LIFT) * dt
    balloon.vx *= BALLOON_DRAG
    balloon.vy *= BALLOON_DRAG
    balloon.x += balloon.vx * dt
    balloon.y += balloon.vy * dt

    hit_registered = False
    for hand_index, hand in enumerate(hand_infos):
        previous_anchor = previous_anchors.get(hand_index, hand.anchor)
        hand_vx = (hand.anchor[0] - previous_anchor[0]) / max(dt, 1e-3)
        hand_vy = (hand.anchor[1] - previous_anchor[1]) / max(dt, 1e-3)
        for point in hand.colliders:
            dx = balloon.x - point[0]
            dy = balloon.y - point[1]
            distance = math.hypot(dx, dy)
            min_distance = balloon.radius + HIT_DISTANCE
            if distance > min_distance:
                continue
            normal_x = dx / distance if distance > 1e-6 else 0.0
            normal_y = dy / distance if distance > 1e-6 else -1.0
            balloon.x = point[0] + normal_x * min_distance
            balloon.y = point[1] + normal_y * min_distance
            upward_boost = max(520.0, -hand_vy * 1.35 + 360.0)
            balloon.vx = clip_speed(balloon.vx + hand_vx * 0.42 + normal_x * 120.0)
            balloon.vy = clip_speed(min(balloon.vy - upward_boost, -360.0))
            if now - game_state.last_hit_time > HIT_COOLDOWN:
                register_hit(game_state, bursts, balloon, now)
                hit_registered = True
            break
        if hit_registered:
            break

    if balloon.x - balloon.radius < 0:
        balloon.x = balloon.radius
        balloon.vx = abs(balloon.vx) * BALLOON_BOUNCE
    elif balloon.x + balloon.radius > frame_shape[1]:
        balloon.x = frame_shape[1] - balloon.radius
        balloon.vx = -abs(balloon.vx) * BALLOON_BOUNCE

    if balloon.y - balloon.radius < 0:
        balloon.y = balloon.radius
        balloon.vy = abs(balloon.vy) * 0.25

    if balloon.y + balloon.radius > ground_y:
        impact_speed = abs(balloon.vy)
        balloon.y = ground_y - balloon.radius
        if impact_speed >= GROUND_HIT_SOUND_MIN_SPEED:
            register_ground_pop(game_state, bursts, balloon, ground_y)
        balloon.vy = -max(160.0, impact_speed * BALLOON_BOUNCE)
        balloon.vx *= 0.90

    return {index: hand.anchor for index, hand in enumerate(hand_infos)}



def draw_overlay(frame, fps: float, face_count: int, hand_count: int, balloon: BalloonState, game_state: GameState, now: float) -> None:
    speed = math.hypot(balloon.vx, balloon.vy)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)
    cv2.putText(frame, f"Faces: {face_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)
    cv2.putText(frame, f"Hands: {hand_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 2)
    cv2.putText(frame, f"Balloon Speed: {speed:.0f}px/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 220, 255), 2)
    cv2.putText(frame, f"Keepy Uppy: {game_state.score}", (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 230, 120), 2)
    cv2.putText(frame, f"Best: {game_state.best_score}", (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    if now <= game_state.message_until:
        cv2.putText(frame, game_state.message, (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 205, 120), 2)
    cv2.putText(frame, "Soft balloon mode for kids", (10, frame.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, "Press Q to quit | S to save frame", (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)



def run_camera(camera_index: int, max_hands: int, min_detection: float, min_tracking: float, model_path: Path) -> None:
    resolved_model_path = ensure_model_file(model_path)
    face_detector = create_face_detector()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}. Check webcam permissions or try another index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    previous_time = time.perf_counter()
    video_start_time = previous_time
    previous_anchors: dict[int, tuple[int, int]] = {}
    balloon: BalloonState | None = None
    bursts: list[PopBurst] = []
    game_state = GameState()
    face_tracks = [
        FaceTrack(name="Bluey", style="bluey", color=(255, 190, 80)),
        FaceTrack(name="Bingo", style="bingo", color=(110, 170, 255)),
    ]

    try:
        with create_landmarker(resolved_model_path, max_hands=max_hands, min_detection=min_detection, min_tracking=min_tracking) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Failed to read a frame from the webcam.")

                frame = cv2.flip(frame, 1)
                if balloon is None:
                    balloon = BalloonState(x=frame.shape[1] * 0.52, y=frame.shape[0] * 0.30, vx=70.0, vy=-70.0)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                now = time.perf_counter()
                timestamp_ms = int((now - video_start_time) * 1000)
                results = hands.detect_for_video(mp_image, timestamp_ms=timestamp_ms)

                face_count = draw_faces(frame, face_detector, face_tracks)
                hand_infos = draw_hands(frame, results)
                ground_y = draw_ground(frame)
                dt = max(1e-3, now - previous_time)
                previous_anchors = update_balloon(balloon, dt, frame.shape, hand_infos, previous_anchors, bursts, game_state, now)
                update_bursts(bursts, dt)
                draw_bursts(frame, bursts)
                draw_balloon(frame, balloon, ground_y)
                fps = 1.0 / dt
                previous_time = now

                draw_overlay(frame, fps=fps, face_count=face_count, hand_count=len(hand_infos), balloon=balloon, game_state=game_state, now=now)
                cv2.imshow("Face Hand Balloon Tracker", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    saved_path = save_frame(frame)
                    print(f"Saved screenshot to {saved_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()



def main() -> None:
    args = parse_args()
    if args.game == "driver":
        run_driver_game(
            camera_index=args.camera,
            min_detection=args.min_detection,
            min_tracking=args.min_tracking,
        )
        return

    run_camera(
        camera_index=args.camera,
        max_hands=args.max_hands,
        min_detection=args.min_detection,
        min_tracking=args.min_tracking,
        model_path=args.model,
    )


if __name__ == "__main__":
    main()
