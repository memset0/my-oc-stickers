"""
本脚本需求：
1. 扫描 data/3_edited 下的 PNG，将处理结果输出到 data/4_final。
2. 判断四边每一行/列是否为白边（>95% 像素接近白色，容忍度 5%），只记录需要裁剪的 offset。
3. 按照初次裁剪后的宽/高，为左右/上下各补偿 10% 的白边（向上取整），不足则以纯白像素补齐。
4. 最终执行“多退少补”裁剪：能裁就裁，越界时补白，确保结果尺寸含 10% 白边。
实现方案：统一以 Pillow + NumPy 转为 RGBA 数组，一次性得到白色像素掩码，再按比例统计首尾白行/列；
随后计算目标裁剪盒与补偿量，并通过新建白底画布粘贴结果，避免多余的多次读写。
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "data" / "3_edited"
TARGET_DIR = ROOT_DIR / "data" / "4_final"

WHITE_RATIO = 0.95
WHITE_TOLERANCE = 0.05
WHITE_LEVEL = int(255 * (1 - WHITE_TOLERANCE))
BORDER_RATIO = 0.05
WHITE_COLOR = (255, 255, 255, 255)


def _white_mask(img: Image.Image) -> np.ndarray:
    """在 RGBA 图像上生成布尔白色掩码，透明像素也视为白。"""
    arr = np.asarray(img, dtype=np.uint8)
    alpha = arr[..., 3]
    rgb = arr[..., :3]
    rgb_white = np.all(rgb >= WHITE_LEVEL, axis=-1)
    return np.logical_or(alpha == 0, rgb_white)


def _leading_count(ratios: np.ndarray) -> int:
    """统计前缀满足白色比例的连续元素个数。"""
    count = 0
    for ratio in ratios:
        if ratio >= WHITE_RATIO:
            count += 1
        else:
            break
    return count


def _compute_offsets(mask: np.ndarray) -> tuple[int, int, int, int]:
    """根据白色掩码计算上下左右需要裁掉的像素行/列数量。"""
    height, width = mask.shape
    row_ratio = mask.mean(axis=1)
    col_ratio = mask.mean(axis=0)

    top = _leading_count(row_ratio)
    bottom = _leading_count(row_ratio[::-1])
    left = _leading_count(col_ratio)
    right = _leading_count(col_ratio[::-1])

    if top + bottom >= height:
        top = bottom = 0
    if left + right >= width:
        left = right = 0
    return top, bottom, left, right


def _target_box(size: tuple[int, int], offsets: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """基于原图尺寸与 offsets 计算包含 10% 白边的目标盒。"""
    width, height = size
    top, bottom, left, right = offsets
    inner_left = left
    inner_right = width - right
    inner_top = top
    inner_bottom = height - bottom

    inner_width = max(1, inner_right - inner_left)
    inner_height = max(1, inner_bottom - inner_top)

    pad = math.ceil(min(inner_width, inner_height) * BORDER_RATIO)

    return (
        inner_left - pad,
        inner_top - pad,
        inner_right + pad,
        inner_bottom + pad,
    )


def _crop_with_padding(img: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """执行裁剪，若越界则补白。"""
    width, height = img.size
    left, top, right, bottom = box
    crop_box = (
        max(left, 0),
        max(top, 0),
        min(right, width),
        min(bottom, height),
    )
    cropped = img.crop(crop_box)

    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)

    if any((pad_left, pad_top, pad_right, pad_bottom)):
        new_width = cropped.width + pad_left + pad_right
        new_height = cropped.height + pad_top + pad_bottom
        canvas = Image.new("RGBA", (new_width, new_height), WHITE_COLOR)
        canvas.paste(cropped, (pad_left, pad_top))
        return canvas
    return cropped


def process_image(path: Path) -> None:
    """对单张图片执行判白、裁剪和补白输出到目标目录。"""
    with Image.open(path) as raw:
        working = raw.convert("RGBA")
        mask = _white_mask(working)
        offsets = _compute_offsets(mask)
        target_box = _target_box(working.size, offsets)
        result = _crop_with_padding(working, target_box)
        TARGET_DIR.mkdir(parents=True, exist_ok=True)
        result.save(TARGET_DIR / path.name, format="PNG")


def main() -> None:
    """遍历源目录逐一处理，并给出简单日志。"""
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"找不到源目录：{SOURCE_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(SOURCE_DIR.iterdir()):
        if not image_path.is_file():
            continue
        try:
            process_image(image_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 跳过 {image_path.name}: {exc}")
        else:
            print(f"[OK] 已处理 {image_path.name}")


if __name__ == "__main__":
    main()
