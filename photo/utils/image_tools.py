import os
import zipfile
from io import BytesIO
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

# 将 rembg 模型目录固定到项目内的 models 文件夹，避免重复下载
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("U2NET_HOME", os.path.join(BASE_DIR, "models"))

from rembg import remove


def _hex_to_bgr(color_name: str) -> Tuple[int, int, int]:
    """根据名称返回 BGR 颜色值（OpenCV 使用 BGR 顺序）"""
    mapping = {
        "white": (255, 255, 255),
        "blue": (255, 0, 0),   # 纯蓝
        "red": (0, 0, 255),    # 纯红
    }
    return mapping.get(color_name, (255, 255, 255))


def remove_background(input_path: str) -> Image.Image:
    """使用 rembg 对头像进行抠图，返回带透明背景的 PIL Image"""
    with open(input_path, "rb") as f:
        input_bytes = f.read()
    output_bytes = remove(input_bytes)
    return Image.open(BytesIO(output_bytes)).convert("RGBA")


def apply_background(pil_img: Image.Image, bg_color_name: str) -> Image.Image:
    """将透明背景替换为指定纯色背景，返回 RGB 图片"""
    bg_color = _hex_to_bgr(bg_color_name)
    # PIL 使用 RGB 排序，因此需要从 BGR 转为 RGB
    r, g, b = bg_color[2], bg_color[1], bg_color[0]
    background = Image.new("RGB", pil_img.size, (r, g, b))
    background.paste(pil_img, mask=pil_img.split()[3])  # 使用 alpha 通道作为 mask
    return background


def resize_to_size(pil_img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """简单缩放图像到指定尺寸（不考虑复杂的人脸居中逻辑，作为 MVP）"""
    return pil_img.resize(size, Image.LANCZOS)


def save_versions(base_img: Image.Image, output_dir: str, prefix: str) -> Tuple[str, str]:
    """生成 1 寸、2 寸版本并保存到 output_dir，返回路径"""
    size_1inch = (295, 413)
    size_2inch = (413, 626)

    img_1 = resize_to_size(base_img, size_1inch)
    img_2 = resize_to_size(base_img, size_2inch)

    path_1 = os.path.join(output_dir, f"{prefix}_1inch_295x413.jpg")
    path_2 = os.path.join(output_dir, f"{prefix}_2inch_413x626.jpg")

    os.makedirs(output_dir, exist_ok=True)
    img_1.save(path_1, format="JPEG", quality=95)
    img_2.save(path_2, format="JPEG", quality=95)

    return path_1, path_2


def make_zip(files, output_dir: str, prefix: str) -> str:
    """将给定文件列表打包成 zip，返回 zip 文件路径"""
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{prefix}_id_photos.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if os.path.exists(f):
                zf.write(f, arcname=os.path.basename(f))
    return zip_path


def process_id_photo_set(input_path: str, output_dir: str, bg_color: str, prefix: str) -> str:
    """完整流程：抠图 + 换底色 + 生成 1 寸、2 寸 + 打包 zip"""
    # 1. 抠图
    pil_transparent = remove_background(input_path)

    # 2. 换底色
    pil_bg = apply_background(pil_transparent, bg_color)

    # 3. 保存 1 寸 / 2 寸
    path_1, path_2 = save_versions(pil_bg, output_dir, prefix)

    # 4. 打包 zip（顺便把抠完图的 PNG 原图也一起打包）
    alpha_path = os.path.join(output_dir, f"{prefix}_alpha.png")
    pil_transparent.save(alpha_path, format="PNG")

    zip_path = make_zip([path_1, path_2, alpha_path], output_dir, prefix)
    return zip_path


