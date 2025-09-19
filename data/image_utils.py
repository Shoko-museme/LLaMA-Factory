from __future__ import annotations

import base64
import copy
import logging
import math
from io import BytesIO
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

# ==== 常量 ====
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200


# ==== 工具函数 ====
def round_by_factor(number: int, factor: int) -> int:
    """返回最接近且能被 factor 整除的整数。"""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """向上取整到能被 factor 整除的整数。"""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """向下取整到能被 factor 整除的整数。"""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """
    计算缩放后的 (h, w)：
      1) h,w 均为 factor 的倍数
      2) 面积在 [min_pixels, max_pixels]
      3) 近似保持原始长宽比（极端长宽比>MAX_RATIO时报错）
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, "
            f"got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(height / beta, factor))
        w_bar = max(factor, floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def to_rgb(pil_image: Image.Image) -> Image.Image:
    """将图像转为 RGB；若是 RGBA，用白底合成。"""
    if pil_image.mode == "RGBA":
        white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
        white_background.paste(pil_image, mask=pil_image.split()[3])
        return white_background
    return pil_image.convert("RGB")


def fetch_image(ele: dict[str, str | Image.Image], size_factor: int = IMAGE_FACTOR) -> Image.Image:
    """
    读取并缩放图像，返回 PIL.Image（已按 smart_resize 调整尺寸）。
    支持输入：
      - {"image": PIL.Image | 本地路径 | http/https | file:// | data:image;base64,...}
      - 或 {"image_url": 同上}
    可选键：
      - resized_height/resized_width：期望的目标尺寸，会再经过 smart_resize 对齐到 factor 倍数与像素范围
      - min_pixels/max_pixels：覆盖默认像素上下限
    """
    # 取出 image 源
    if "image" in ele:
        image = ele["image"]
    else:
        image = ele["image_url"]

    # 加载为 PIL.Image
    image_obj: Optional[Image.Image] = None
    if isinstance(image, Image.Image):
        image_obj = image
    elif isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
        # 用 stream=True 防止 BytesIO 内存泄漏
        import requests  # 局部导入也可
        with requests.get(image, stream=True) as resp:
            resp.raise_for_status()
            with BytesIO(resp.content) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    elif isinstance(image, str) and image.startswith("file://"):
        image_obj = Image.open(image[7:])
    elif isinstance(image, str) and image.startswith("data:image"):
        if "base64," in image:
            _, b64 = image.split("base64,", 1)
            data = base64.b64decode(b64)
            with BytesIO(data) as bio:
                image_obj = copy.deepcopy(Image.open(bio))
    else:
        # 认为是本地路径
        image_obj = Image.open(image)  # type: ignore[arg-type]

    if image_obj is None:
        raise ValueError(f"Unrecognized image input: {image}")

    # 转 RGB
    image = to_rgb(image_obj)

    # 计算目标尺寸
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],  # type: ignore[index]
            ele["resized_width"],   # type: ignore[index]
            factor=size_factor,
        )
    else:
        width, height = image.size
        min_pixels = ele.get("min_pixels", MIN_PIXELS)  # type: ignore[assignment]
        max_pixels = ele.get("max_pixels", MAX_PIXELS)  # type: ignore[assignment]
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,  # type: ignore[arg-type]
            max_pixels=max_pixels,  # type: ignore[arg-type]
        )

    # 实际缩放（保持与原实现一致：不显式指定插值 => PIL 默认）
    image = image.resize((resized_width, resized_height))
    return image
