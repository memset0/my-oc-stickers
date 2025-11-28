"""
处理 data/4_located 内的 PNG，将非白色像素扩展为蒙版导出到 data/5_transport。

执行流程：
1. 以 5% 容忍度判定白色像素，并生成“非白”掩码。
2. 将掩码按圆形半径 12px 进行膨胀，确保蒙版区域足够覆盖。
3. 对膨胀结果施加 2px 羽化，使边缘更加柔和。
4. 根据 MODE 输出：
   - MODE=TEST：蒙版区域涂成红色，其他像素保持原样。
   - MODE=PRODUCT：保留蒙版区域原像素，其他像素置为透明。
5. 裁去四周整行/列的全透明像素，再补上 5px padding。
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "data" / "4_located"
TARGET_DIR = ROOT_DIR / "data" / "5_transport"

WHITE_TOLERANCE_IN = 0.01
WHITE_TOLERANCE_OUT = 0.09
MASK_EXPAND_RADIUS = 10
FEATHER_RADIUS = 1
PADDING_PX = 2
MASK_COLOR = np.array([255, 0, 0, 255], dtype=np.uint8)

STRUCTURING_OFFSETS = tuple(
    (dy, dx)
    for dy in range(-MASK_EXPAND_RADIUS, MASK_EXPAND_RADIUS + 1)
    for dx in range(-MASK_EXPAND_RADIUS, MASK_EXPAND_RADIUS + 1)
    if dx * dx + dy * dy <= MASK_EXPAND_RADIUS * MASK_EXPAND_RADIUS
)


class Mode(str, Enum):
    TEST = "TEST"
    PRODUCT = "PRODUCT"


def _white_mask(arr: np.ndarray, tolerance: float) -> np.ndarray:
    """返回 True 表示在指定容忍度下判定为白色或透明的像素。"""
    white_level = int(255 * (1 - tolerance))
    alpha = arr[..., 3]
    rgb = arr[..., :3]
    rgb_white = np.all(rgb >= white_level, axis=-1)
    return np.logical_or(alpha == 0, rgb_white)


def _outer_background_mask(white_mask: np.ndarray) -> np.ndarray:
    """BFS 从边界出发，仅保留与边缘连通的宽松白色区域。"""
    height, width = white_mask.shape
    if height == 0 or width == 0:
        return white_mask

    reachable = np.zeros_like(white_mask, dtype=bool)
    queue: deque[tuple[int, int]] = deque()

    def enqueue(y: int, x: int) -> None:
        if not white_mask[y, x] or reachable[y, x]:
            return
        reachable[y, x] = True
        queue.append((y, x))

    for x in range(width):
        enqueue(0, x)
        enqueue(height - 1, x)
    for y in range(height):
        enqueue(y, 0)
        enqueue(y, width - 1)

    while queue:
        y, x = queue.popleft()
        if y > 0:
            enqueue(y - 1, x)
        if y + 1 < height:
            enqueue(y + 1, x)
        if x > 0:
            enqueue(y, x - 1)
        if x + 1 < width:
            enqueue(y, x + 1)

    return reachable


def _expand_mask(mask: np.ndarray) -> np.ndarray:
    """按圆形半径 10px 膨胀掩码。"""
    if MASK_EXPAND_RADIUS <= 0:
        return mask
    pad = MASK_EXPAND_RADIUS
    padded = np.pad(mask, pad_width=pad, mode="constant", constant_values=False)
    expanded = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    for dy, dx in STRUCTURING_OFFSETS:
        expanded |= padded[pad + dy: pad + dy + height, pad + dx: pad + dx + width]
    return expanded


def _feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """对布尔掩码应用羽化，返回 0-1 的软边蒙版。"""
    if radius <= 0:
        return mask.astype(np.float32)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    blurred = mask_image.filter(ImageFilter.GaussianBlur(radius=radius))
    feather = np.asarray(blurred, dtype=np.float32) / 255.0
    return np.clip(feather, 0.0, 1.0)


def _trim_transparent_edges(arr: np.ndarray, padding: int) -> np.ndarray:
    """裁去四周整行/列的透明区域，并补上指定 padding。"""
    alpha = arr[..., 3]
    height, width = alpha.shape
    top = 0
    while top < height and np.all(alpha[top] == 0):
        top += 1
    bottom = height
    while bottom > top and np.all(alpha[bottom - 1] == 0):
        bottom -= 1
    left = 0
    while left < width and np.all(alpha[:, left] == 0):
        left += 1
    right = width
    while right > left and np.all(alpha[:, right - 1] == 0):
        right -= 1

    if top >= bottom or left >= right:
        cropped = arr.copy()
    else:
        cropped = arr[top:bottom, left:right]

    if padding <= 0:
        return cropped.copy()

    new_height = cropped.shape[0] + padding * 2
    new_width = cropped.shape[1] + padding * 2
    padded = np.zeros((new_height, new_width, arr.shape[2]), dtype=arr.dtype)
    padded[padding:padding + cropped.shape[0], padding:padding + cropped.shape[1]] = cropped
    return padded


def _build_subject_mask(arr: np.ndarray) -> np.ndarray:
    """结合内外容忍度生成主体蒙版。"""
    outer_white = _white_mask(arr, WHITE_TOLERANCE_OUT)
    outer_background = _outer_background_mask(outer_white)
    inner_white = _white_mask(arr, WHITE_TOLERANCE_IN)
    preserved = np.logical_or(outer_background, inner_white)
    return np.logical_not(preserved)


def _render(arr: np.ndarray, mask: np.ndarray, mode: Mode) -> np.ndarray:
    """根据模式输出图像数组。"""
    mask_float = mask.astype(np.float32)
    mask_4d = mask_float[..., None]
    arr_float = arr.astype(np.float32)
    if mode is Mode.TEST:
        result = arr_float * (1.0 - mask_4d) + MASK_COLOR.astype(np.float32) * mask_4d
        return np.clip(result, 0, 255).astype(np.uint8)
    if mode is Mode.PRODUCT:
        result = np.zeros_like(arr_float)
        result[..., :3] = arr_float[..., :3] * mask_4d + 255.0 * (1.0 - mask_4d)
        result[..., 3] = arr_float[..., 3] * mask_float
        return np.clip(result, 0, 255).astype(np.uint8)
    raise ValueError(f"不支持的模式：{mode}")


def process_image(path: Path, mode: Mode) -> None:
    """执行单图处理并写入目标目录。"""
    target_path = TARGET_DIR / path.name
    if target_path.exists():
        raise FileExistsError(f"目标已存在：{target_path.name}")
    with Image.open(path) as raw:
        working = raw.convert("RGBA")
        arr = np.asarray(working, dtype=np.uint8)
        mask = _build_subject_mask(arr)
        mask = _expand_mask(mask)
        mask = _feather_mask(mask, FEATHER_RADIUS)
        rendered = _render(arr, mask, mode)
        rendered = _trim_transparent_edges(rendered, PADDING_PX)
        result = Image.fromarray(rendered, mode="RGBA")
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        result.save(target_path, format="PNG")


def _resolve_mode(cli_mode: str | None) -> Mode:
    """优先使用 CLI，其次环境变量 MODE，默认 TEST。"""
    candidates = [
        cli_mode,
        os.environ.get("MODE"),
        Mode.TEST.value,
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_upper = candidate.upper()
        try:
            return Mode(candidate_upper)
        except ValueError:
            continue
    raise ValueError("无法解析 MODE")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in Mode],
        default=Mode.PRODUCT,
        help="TEST 使用红色显示蒙版；PRODUCT 保留蒙版像素并透明化背景",
    )
    args = parser.parse_args(argv)
    mode = _resolve_mode(args.mode)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"找不到源目录：{SOURCE_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(SOURCE_DIR.iterdir()):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in {".png"}:
            continue
        try:
            process_image(image_path, mode)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 跳过 {image_path.name}: {exc}")
        else:
            print(f"[OK] 已处理 {image_path.name}")


if __name__ == "__main__":
    main()
