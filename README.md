# Face Hand Balloon Tracker

Small Python project that uses:

- OpenCV for webcam access and face detection
- MediaPipe for hand landmark tracking
- A softer virtual balloon game for kids
- A motion-controlled driving game with camera-based lane control
- Stable Bluey-style and Bingo-style face overlays for the first two faces
- Friendly pop effects and simple score feedback

## Requirements

- Python 3.10+
- A webcam
- Internet access on first run so the MediaPipe hand model can be downloaded automatically
- Windows if you want the built-in pop sounds

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

## Run

Balloon game:

```powershell
handgesture --game balloon
```

Driving game:

```powershell
handgesture --game driver
```

Or run the module directly:

```powershell
python -m handgesture.app --game driver
```

## Options

```powershell
handgesture --game balloon --camera 0 --max-hands 2 --min-detection 0.6 --min-tracking 0.6
```

Driving mode uses the same camera and tracking thresholds:

```powershell
handgesture --game driver --camera 0 --min-detection 0.6 --min-tracking 0.6
```

If you already have a `hand_landmarker.task` model file, you can point balloon mode at it directly:

```powershell
handgesture --game balloon --model .\models\hand_landmarker.task
```

## Balloon Mode

- A Bluey-style overlay on the first detected face
- A Bingo-style overlay on the second detected face
- Hand landmarks labeled `Left` or `Right` in the mirrored camera view
- A soft floating balloon and a raised ground line
- Easier hand hits to keep the balloon in the air
- A `Keepy Uppy` score and best score
- Friendly pop effects and sounds when the balloon is hit or touches the ground

## Driver Mode

- OpenCV face tracking estimates the kid's left, center, or right lane position
- Three driving lanes map to left, center, and right body positions
- A glowing start lane must be held briefly before the round begins
- A `3, 2, 1` countdown starts the car
- The road scrolls while obstacle sprites move downward toward the player
- Dodging obstacles increases score during a short 25 second round
- A live camera preview in the corner shows tracking and lane zones
- Obstacle art loads from `assets\obstacles`

Controls:

- `q` to quit
- `s` to save a screenshot of the current frame

## Notes

- Face overlays use smoothed face tracking so they stay steadier on screen.
- Face detection is local OpenCV detection, not person identity recognition.
- On first run, balloon mode stores the downloaded hand model in `models\hand_landmarker.task`.
- Driver mode stays in OpenCV so it runs with the current repo dependencies and does not require Pygame.

