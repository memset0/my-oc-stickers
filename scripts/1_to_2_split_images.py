"""
本脚本需求：
1. 扫描 data/1_source 下的图片。
2. 通过文件名里的 [m x n] 得知横向 m、纵向 n。
3. 通过文件名前缀的数字得到 id。
4. 将图片按 n 行 m 列切分并保存到 data/2_split/%03d_%d_%d.png。
5. 若原图尺寸无法被整除，则补齐最少的空白像素行/列再切分。
实现方案：利用 pathlib 遍历、正则抽取 id/m/n，再用 Pillow 精确裁剪（无法整除时先扩展画布为透明/白色空白），逐块输出，确保仍为 PNG 无压缩损失。
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "data" / "1_source"
TARGET_DIR = ROOT_DIR / "data" / "2_split"

FILENAME_PATTERN = re.compile(
    r"^(?P<id>\d+).*?\[(?P<cols>\d+)x(?P<rows>\d+)\]", re.IGNORECASE
)


def parse_filename(path: Path) -> tuple[int, int, int]:
    """按照需求从文件名解析出 (id, m, n)，未匹配则抛错提示用户补齐命名。"""
    match = FILENAME_PATTERN.search(path.name)
    if not match:
        raise ValueError(f"文件名不符合约定格式：{path.name}")
    image_id = int(match.group("id"))
    cols = int(match.group("cols"))
    rows = int(match.group("rows"))
    if cols <= 0 or rows <= 0:
        raise ValueError(f"切分列/行必须为正整数：{path.name}")
    return image_id, cols, rows


def _padding_color(img: Image.Image):
    """根据模式选择空白颜色：若有 alpha 则透明，否则白色（或 0 作为保底）。"""
    bands = img.getbands()
    if "A" in bands:
        return tuple(0 for _ in bands)
    if img.mode == "L":
        return 255
    if img.mode == "RGB":
        return (255, 255, 255)
    return 0


def split_image(path: Path) -> None:
    """执行单张图片的切分与保存，确保输出命名为 %03d_%d_%d。"""
    image_id, cols, rows = parse_filename(path)
    existing_tiles: list[str] = []
    for row in range(rows):
        for col in range(cols):
            tile_path = TARGET_DIR / f"{image_id:03d}_{row + 1}_{col + 1}.png"
            if tile_path.exists():
                existing_tiles.append(tile_path.name)
    if existing_tiles:
        preview = ", ".join(existing_tiles[:3])
        if len(existing_tiles) > 3:
            preview += "..."
        raise FileExistsError(f"目标切片已存在（例如：{preview}），已跳过以避免覆盖")
    with Image.open(path) as img:
        width, height = img.size
        pad_w = (-width) % cols
        pad_h = (-height) % rows
        if pad_w or pad_h:
            # 任务要求无法整除时追加最少的空白像素行/列再切分。
            new_size = (width + pad_w, height + pad_h)
            padded = Image.new(img.mode, new_size, _padding_color(img))
            padded.paste(img, (0, 0))
            img = padded
            width, height = img.size
        tile_w = width // cols
        tile_h = height // rows
        for row in range(rows):
            for col in range(cols):
                left = col * tile_w
                upper = row * tile_h
                right = left + tile_w
                lower = upper + tile_h
                tile = img.crop((left, upper, right, lower))
                tile_path = TARGET_DIR / f"{image_id:03d}_{row + 1}_{col + 1}.png"
                tile.save(tile_path, format="PNG")


def main() -> None:
    """入口函数：遍历源目录，逐张执行切分，同时确保输出目录存在。"""
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"找不到源目录：{SOURCE_DIR}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(SOURCE_DIR.iterdir()):
        if not image_path.is_file():
            continue
        try:
            split_image(image_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 跳过 {image_path.name}: {exc}")
        else:
            print(f"[OK] 已切分 {image_path.name}")


if __name__ == "__main__":
    main()
