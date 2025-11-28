"""
本脚本需求：
1. 读取 data/4_final 下的全部图片，以 README.md 为目标插入预览。
2. 以随机种子 20040822 洗牌，确保生成顺序固定可复现。
实现方案：使用 pathlib 枚举图片、random.Random(seed) 打乱，再构造含相对路径的
<img> 片段，最后在 <!-- preview start --> 与 <!-- preview end --> 之间替换写回 README。
"""

from __future__ import annotations

import random
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT_DIR / "data" / "5_transport"
README_PATH = ROOT_DIR / "README.md"

PREVIEW_START = "<!-- preview start -->"
PREVIEW_END = "<!-- preview end -->"
RANDOM_SEED = 20040822
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def collect_final_images() -> list[str]:
    """扫描终稿目录，筛选图片并返回相对路径列表。"""
    if not FINAL_DIR.exists():
        raise FileNotFoundError(f"找不到最终图片目录：{FINAL_DIR}")
    images = [
        path
        for path in FINAL_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise ValueError("4_final 目录中未找到可用图片")

    images.sort(key=lambda p: p.name)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(images)
    return [path.relative_to(ROOT_DIR).as_posix() for path in images]


def render_preview_block(rel_paths: list[str]) -> str:
    """将图片相对路径渲染为 <img> 片段，供 README 注入使用。"""
    lines = []
    for rel_path in rel_paths:
        alt_text = Path(rel_path).stem
        lines.append(f'<img src="{rel_path}" alt="{alt_text}" height="100"/>')
    return "\n".join(lines)


def inject_preview(content: str, snippet: str) -> str:
    """把渲染结果替换进 README 预览占位符之间。"""
    start_idx = content.find(PREVIEW_START)
    if start_idx == -1:
        raise ValueError("README 中缺失 <!-- preview start --> 标记")
    end_idx = content.find(PREVIEW_END, start_idx)
    if end_idx == -1:
        raise ValueError("README 中缺失 <!-- preview end --> 标记")

    before = content[: start_idx + len(PREVIEW_START)]
    after = content[end_idx:]
    replacement = f"\n\n{snippet}\n\n"
    return before + replacement + after


def main() -> None:
    """入口：收集图片、渲染代码片段并写回 README。"""
    rel_paths = collect_final_images()
    preview_block = render_preview_block(rel_paths)
    readme_text = README_PATH.read_text(encoding="utf-8")
    updated = inject_preview(readme_text, preview_block)
    README_PATH.write_text(updated, encoding="utf-8")
    print(f"[OK] 已写入 {len(rel_paths)} 张图片到 README")


if __name__ == "__main__":
    main()
