from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SIZE = 512
OUTPUT_DIR = Path("assets") / "obstacles"


def bgra(r: int, g: int, b: int, a: int = 255) -> tuple[int, int, int, int]:
    return b, g, r, a


def new_canvas() -> np.ndarray:
    return np.zeros((SIZE, SIZE, 4), dtype=np.uint8)


def alpha_blend(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    base_rgb = base[:, :, :3].astype(np.float32)
    base_alpha = (base[:, :, 3:4].astype(np.float32)) / 255.0
    layer_rgb = layer[:, :, :3].astype(np.float32)
    layer_alpha = (layer[:, :, 3:4].astype(np.float32)) / 255.0

    out_alpha = layer_alpha + base_alpha * (1.0 - layer_alpha)
    safe_alpha = np.where(out_alpha == 0.0, 1.0, out_alpha)
    out_rgb = (layer_rgb * layer_alpha + base_rgb * base_alpha * (1.0 - layer_alpha)) / safe_alpha

    result = np.zeros_like(base)
    result[:, :, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    result[:, :, 3] = np.clip(out_alpha[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    return result


def add_shadow(canvas: np.ndarray, center: tuple[int, int], axes: tuple[int, int], alpha: int = 80) -> None:
    layer = new_canvas()
    cv2.ellipse(layer, center, axes, 0, 0, 360, bgra(25, 40, 80, alpha), -1, lineType=cv2.LINE_AA)
    canvas[:] = alpha_blend(canvas, layer)


def add_glow(canvas: np.ndarray, center: tuple[int, int], axes: tuple[int, int], color: tuple[int, int, int, int]) -> None:
    layer = new_canvas()
    cv2.ellipse(layer, center, axes, 0, 0, 360, color, -1, lineType=cv2.LINE_AA)
    layer[:, :, 3] = cv2.GaussianBlur(layer[:, :, 3], (0, 0), 16)
    canvas[:] = alpha_blend(canvas, layer)


def outline_poly(canvas: np.ndarray, points: np.ndarray, fill: tuple[int, int, int, int], outline: tuple[int, int, int, int], thickness: int = 8) -> None:
    cv2.fillConvexPoly(canvas, points, fill, lineType=cv2.LINE_AA)
    cv2.polylines(canvas, [points], True, outline, thickness, lineType=cv2.LINE_AA)


def flatten_on_bg(image: np.ndarray, color: tuple[int, int, int] = (218, 214, 210)) -> np.ndarray:
    bg = np.full((image.shape[0], image.shape[1], 3), color, dtype=np.uint8)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    fg = image[:, :, :3].astype(np.float32)
    out = fg * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_cone() -> np.ndarray:
    canvas = new_canvas()
    add_shadow(canvas, (256, 426), (124, 34))
    add_glow(canvas, (256, 270), (118, 150), bgra(255, 140, 70, 46))

    cone = np.array([(256, 96), (152, 376), (360, 376)], dtype=np.int32)
    outline_poly(canvas, cone, bgra(255, 118, 56), bgra(70, 25, 18), 9)
    cv2.rectangle(canvas, (132, 366), (380, 416), bgra(255, 170, 72), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (132, 366), (380, 416), bgra(70, 25, 18), 8, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (178, 198), (334, 236), bgra(255, 248, 230), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (165, 272), (348, 308), bgra(255, 248, 230), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (224, 180), (42, 18), 20, 0, 360, bgra(255, 218, 175, 190), -1, lineType=cv2.LINE_AA)
    return canvas


def draw_rock() -> np.ndarray:
    canvas = new_canvas()
    add_shadow(canvas, (256, 424), (126, 34))
    add_glow(canvas, (256, 286), (124, 108), bgra(140, 180, 255, 34))

    rock = np.array(
        [(132, 334), (164, 222), (246, 158), (336, 176), (388, 250), (378, 356), (304, 396), (186, 392)],
        dtype=np.int32,
    )
    outline_poly(canvas, rock, bgra(123, 130, 154), bgra(44, 58, 90), 9)
    highlight = np.array([(188, 234), (244, 194), (296, 202), (252, 258)], dtype=np.int32)
    cv2.fillConvexPoly(canvas, highlight, bgra(193, 202, 225, 200), lineType=cv2.LINE_AA)
    cv2.line(canvas, (228, 278), (198, 340), bgra(66, 80, 108), 6, lineType=cv2.LINE_AA)
    cv2.line(canvas, (282, 250), (326, 324), bgra(66, 80, 108), 6, lineType=cv2.LINE_AA)
    cv2.line(canvas, (254, 292), (308, 352), bgra(66, 80, 108), 5, lineType=cv2.LINE_AA)
    return canvas


def draw_crate() -> np.ndarray:
    canvas = new_canvas()
    add_shadow(canvas, (256, 430), (120, 30))
    add_glow(canvas, (256, 262), (120, 132), bgra(255, 210, 110, 36))

    cv2.rectangle(canvas, (150, 158), (362, 382), bgra(200, 126, 60), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (150, 158), (362, 382), bgra(96, 44, 18), 9, lineType=cv2.LINE_AA)
    for y in (206, 254, 302):
        cv2.line(canvas, (158, y), (354, y), bgra(130, 74, 28), 10, lineType=cv2.LINE_AA)
    cv2.line(canvas, (182, 182), (330, 330), bgra(116, 60, 22), 12, lineType=cv2.LINE_AA)
    cv2.line(canvas, (330, 182), (182, 330), bgra(116, 60, 22), 12, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (188, 176), (246, 198), bgra(255, 205, 150, 150), -1, lineType=cv2.LINE_AA)
    return canvas


def draw_puddle() -> np.ndarray:
    canvas = new_canvas()
    add_shadow(canvas, (256, 420), (132, 28))
    add_glow(canvas, (256, 314), (136, 78), bgra(120, 220, 255, 40))

    cv2.ellipse(canvas, (256, 306), (138, 66), 0, 0, 360, bgra(62, 186, 255), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (184, 324), (74, 42), -8, 0, 360, bgra(86, 210, 255), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (326, 326), (82, 44), 12, 0, 360, bgra(76, 196, 255), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (256, 306), (138, 66), 0, 0, 360, bgra(28, 95, 190), 8, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (220, 288), (34, 12), -6, 0, 360, bgra(218, 248, 255, 170), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (294, 330), (42, 14), 6, 0, 360, bgra(218, 248, 255, 120), -1, lineType=cv2.LINE_AA)
    return canvas


def draw_cart() -> np.ndarray:
    canvas = new_canvas()
    add_shadow(canvas, (256, 430), (126, 32))
    add_glow(canvas, (254, 262), (136, 124), bgra(220, 90, 90, 34))

    cv2.rectangle(canvas, (156, 212), (344, 332), bgra(211, 75, 68), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (156, 212), (344, 332), bgra(72, 28, 24), 9, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (182, 182), (318, 220), bgra(255, 198, 84), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(canvas, (182, 182), (318, 220), bgra(72, 28, 24), 8, lineType=cv2.LINE_AA)
    cv2.line(canvas, (344, 236), (410, 186), bgra(255, 198, 84), 12, lineType=cv2.LINE_AA)
    cv2.line(canvas, (408, 188), (436, 164), bgra(72, 28, 24), 6, lineType=cv2.LINE_AA)
    for x in (196, 304):
        cv2.circle(canvas, (x, 350), 34, bgra(28, 40, 88), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, 350), 34, bgra(16, 22, 40), 9, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (x, 350), 12, bgra(196, 214, 255), -1, lineType=cv2.LINE_AA)
    cv2.ellipse(canvas, (220, 228), (32, 12), -18, 0, 360, bgra(255, 210, 180, 170), -1, lineType=cv2.LINE_AA)
    return canvas


def draw_tire_stack() -> np.ndarray:
    canvas = new_canvas()
    add_shadow(canvas, (256, 430), (120, 30))
    add_glow(canvas, (256, 274), (122, 132), bgra(120, 180, 255, 24))

    centers = [(256, 188), (232, 270), (280, 270), (256, 350)]
    for cx, cy in centers:
        cv2.ellipse(canvas, (cx, cy), (82, 40), 0, 0, 360, bgra(44, 50, 70), -1, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy), (82, 40), 0, 0, 360, bgra(12, 18, 34), 10, lineType=cv2.LINE_AA)
        cv2.ellipse(canvas, (cx, cy), (32, 14), 0, 0, 360, bgra(118, 128, 154), -1, lineType=cv2.LINE_AA)
    return canvas


def compose_preview(images: dict[str, np.ndarray]) -> np.ndarray:
    card_h = 440
    card_w = 440
    sheet = np.full((960, 1400, 3), (208, 204, 200), dtype=np.uint8)
    cv2.putText(sheet, "Obstacle Sprite Set", (438, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (45, 45, 55), 3, lineType=cv2.LINE_AA)
    cv2.putText(sheet, "Cartoon style matched to the red racer reference", (346, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (70, 70, 80), 2, lineType=cv2.LINE_AA)

    names = list(images.keys())
    for index, name in enumerate(names):
        row = index // 3
        col = index % 3
        x0 = 80 + col * card_w
        y0 = 150 + row * card_h
        cv2.rectangle(sheet, (x0, y0), (x0 + 360, y0 + 320), (230, 228, 226), -1, lineType=cv2.LINE_AA)
        cv2.rectangle(sheet, (x0, y0), (x0 + 360, y0 + 320), (168, 164, 160), 2, lineType=cv2.LINE_AA)
        sprite = cv2.resize(flatten_on_bg(images[name], (230, 228, 226)), (320, 320))
        sheet[y0 : y0 + 320, x0 + 20 : x0 + 340] = sprite
        cv2.putText(sheet, name.replace("-", " ").title(), (x0 + 96, y0 + 356), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (40, 40, 48), 2, lineType=cv2.LINE_AA)
    return sheet


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sprites = {
        "traffic-cone": draw_cone(),
        "rock": draw_rock(),
        "wood-crate": draw_crate(),
        "puddle": draw_puddle(),
        "broken-cart": draw_cart(),
        "tire-stack": draw_tire_stack(),
    }

    for name, image in sprites.items():
        cv2.imwrite(str(OUTPUT_DIR / f"{name}.png"), image)

    preview = compose_preview(sprites)
    cv2.imwrite(str(OUTPUT_DIR / "obstacle-preview-sheet.png"), preview)
    print(f"Saved {len(sprites)} obstacle sprites to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
