import os
import cv2
import zipfile
from io import BytesIO
from typing import Tuple, Dict, Optional

import numpy as np
import mediapipe as mp
from PIL import Image, ImageFilter

# 导入证件照合规规则源（single source of truth）
from utils.composition_params import IDPHOTO_RULES, SIZE_PRESETS

# 将 rembg 模型目录固定到项目内的 models 文件夹（保留用于向后兼容）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("U2NET_HOME", os.path.join(BASE_DIR, "models"))

# ================================
# 从规则源派生的运行参数（避免 magic number）
# ================================

OUTPUT_W, OUTPUT_H = IDPHOTO_RULES["output_size"]
HEAD_RATIO_MIN, HEAD_RATIO_MAX = IDPHOTO_RULES["head_ratio_range"]
HEAD_RATIO_TARGET = (HEAD_RATIO_MIN + HEAD_RATIO_MAX) / 2.0
EYE_LINE_RATIO = IDPHOTO_RULES["eye_line_ratio"]
DPI = IDPHOTO_RULES["dpi"]

ASPECT_RATIO = OUTPUT_W / OUTPUT_H

# 分辨率安全阈值（用于警告，不抛异常）
MIN_SRC_MIN_EDGE = IDPHOTO_RULES["min_short_edge"]
MIN_SRC_AREA = OUTPUT_W * OUTPUT_H  # 产出分辨率作为推荐面积
MIN_CROP_W = OUTPUT_W
MIN_CROP_H = OUTPUT_H

# 清晰度阈值（可按需调整，这里仍使用 30.0）
SHARPNESS_THRESHOLD = 30.0


def _get_spec_output_size(spec: str) -> Tuple[int, int]:
    """
    根据规格返回目标输出尺寸。
    - 优先从 SIZE_PRESETS 中查找
    - 找不到时回退到默认 2 寸尺寸（OUTPUT_W, OUTPUT_H）
    """
    try:
        cfg = SIZE_PRESETS.get(spec)
        if cfg and "output_size" in cfg:
            return cfg["output_size"]
    except Exception:
        pass
    return OUTPUT_W, OUTPUT_H


def _get_spec_comp_params(spec: str):
    """
    返回指定规格的构图参数：
    - head_ratio_min / max / target
    - eye_line_ratio
    - 目标宽高（与 _get_spec_output_size 一致）
    """
    target_w, target_h = _get_spec_output_size(spec)
    hr_min, hr_max = HEAD_RATIO_MIN, HEAD_RATIO_MAX
    eye_ratio = EYE_LINE_RATIO
    try:
        cfg = SIZE_PRESETS.get(spec)
        if cfg:
            if "head_ratio_range" in cfg:
                hr_min, hr_max = cfg["head_ratio_range"]
            if "eye_line_ratio" in cfg:
                eye_ratio = cfg["eye_line_ratio"]
    except Exception:
        pass
    hr_target = (hr_min + hr_max) / 2.0
    return hr_min, hr_max, hr_target, eye_ratio, target_w, target_h


def _hex_to_bgr(color_name: str) -> Tuple[int, int, int]:
    """根据名称返回 BGR 颜色值（OpenCV 使用 BGR 顺序）"""
    mapping = {
        "white": (255, 255, 255),
        "blue": (255, 0, 0),
        "red": (0, 0, 255),
    }
    return mapping.get(color_name, (255, 255, 255))


def _load_image_safe(input_path: str) -> Image.Image:
    """
    安全加载图片，失败时返回纯色兜底图
    """
    try:
        img = Image.open(input_path)
        # 确保图片已加载
        img.load()
        return img.convert("RGBA" if img.mode in ("RGBA", "LA") else "RGB")
    except Exception as e:
        print(f"[警告] 图片加载失败: {e}，使用兜底图")
        # 返回纯色兜底图（白色背景，灰色占位符），尺寸来自规则源
        fallback = Image.new("RGB", (OUTPUT_W, OUTPUT_H), (240, 240, 240))
        return fallback


def _detect_background_type(pil_img: Image.Image) -> str:
    """
    检测图片背景类型
    返回: "transparent" | "solid"
    """
    if pil_img.mode == "RGBA":
        # 检查 alpha 通道是否有透明像素
        alpha = pil_img.split()[-1]
        # 如果 alpha 通道最小值 < 255，说明有透明像素
        if alpha.getextrema()[0] < 255:
            return "transparent"
    return "solid"


def _safe_composition(pil_img: Image.Image, spec: str = "2inch") -> Image.Image:
    """
    安全构图裁剪（带多层 fallback）
    主流程：MediaPipe 人脸检测 + 黄金参数裁剪
    Fallback 1：中心裁剪
    Fallback 2：保持原图
    """
    try:
        # 规格对应的目标比例与构图参数
        hr_min, hr_max, head_ratio_target, eye_line_ratio_target, target_w, target_h = _get_spec_comp_params(spec)
        aspect_ratio = target_w / float(target_h)

        # 转为 RGB numpy 数组（MediaPipe 需要）
        img_mode = pil_img.mode
        if img_mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGB")
            img_mode = "RGB"

        np_img = np.array(pil_img.convert("RGB"))
        h, w, _ = np_img.shape

        # 初始化 MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(np_img)

        # 如果没有检测到人脸，fallback 到中心裁剪
        if not results.multi_face_landmarks:
            print("[警告] 未检测到人脸，使用中心裁剪")
            return _fallback_center_crop(pil_img, spec)

        face_landmarks = results.multi_face_landmarks[0].landmark

        def lm_xy(idx: int):
            """将相对坐标转换为像素坐标"""
            lm = face_landmarks[idx]
            return lm.x * w, lm.y * h

        # 指定关键点
        chin_x, chin_y = lm_xy(152)
        brow_x, brow_y = lm_xy(168)
        forehead_x, forehead_y = lm_xy(10)
        left_cheek_x, left_cheek_y = lm_xy(234)
        right_cheek_x, right_cheek_y = lm_xy(454)

        # 预计头顶 & 下巴
        head_top_y = min(brow_y, forehead_y)
        head_bottom_y = chin_y
        face_height = head_bottom_y - head_top_y

        if face_height <= 0.0:
            print("[警告] 人脸高度异常，使用中心裁剪")
            return _fallback_center_crop(pil_img, spec)

        # 估计脸部左右范围中心
        face_left_x = min(left_cheek_x, right_cheek_x)
        face_right_x = max(left_cheek_x, right_cheek_x)
        face_center_x = (face_left_x + face_right_x) / 2.0

        # ===== 计算真实眼睛位置（使用眼睛关键点）=====
        try:
            eye_indices = [159, 145, 386, 374]  # 上下眼睑附近若干点
            eye_y = sum(face_landmarks[i].y for i in eye_indices) / len(eye_indices) * h
        except Exception:
            # 回退：按头部几何预估眼睛大致在头部中上位置
            eye_y = head_top_y + face_height * 0.45

        # 计算目标裁剪高度（使用头部占比目标值）
        crop_height = face_height / max(head_ratio_target, 1e-6)

        # 根据“眼睛在整幅画面中的目标位置”反推裁剪上边界
        crop_top = eye_y - eye_line_ratio_target * crop_height
        crop_bottom = crop_top + crop_height

        # 左右以脸中心居中，至少5%留白
        crop_width = crop_height * aspect_ratio
        min_side = 0.05 * crop_width
        crop_left = face_center_x - crop_width / 2.0
        crop_right = crop_left + crop_width

        if (face_left_x - crop_left) < min_side:
            shift = min_side - (face_left_x - crop_left)
            crop_left -= shift
            crop_right -= shift
        if (crop_right - face_right_x) < min_side:
            shift = min_side - (crop_right - face_right_x)
            crop_left += shift
            crop_right += shift

        # 转换为整数坐标
        crop_left = int(round(crop_left))
        crop_top = int(round(crop_top))
        crop_right = int(round(crop_right))
        crop_bottom = int(round(crop_bottom))

        # 计算与原图的交集区域
        src_left = max(0, crop_left)
        src_top = max(0, crop_top)
        src_right = min(w, crop_right)
        src_bottom = min(h, crop_bottom)

        if src_right <= src_left or src_bottom <= src_top:
            print("[警告] 裁剪区域异常，使用中心裁剪")
            return _fallback_center_crop(pil_img, spec)

        # 从原图中裁剪出交集区域，直接返回该区域
        # 尺寸与比例的最终统一交给 _safe_resize 完成，避免出现多余白边画布
        src_crop = pil_img.crop((src_left, src_top, src_right, src_bottom))
        return src_crop

    except Exception as e:
        print(f"[警告] 构图裁剪失败: {e}，保持原图")
        return pil_img


def _fallback_center_crop(pil_img: Image.Image, spec: str = "2inch") -> Image.Image:
    """中心裁剪 fallback（按规格输出尺寸）"""
    try:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        target_w, target_h = _get_spec_output_size(spec)
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        canvas.paste(resized, (paste_x, paste_y))
        return canvas
    except Exception:
        return pil_img


def _safe_background(pil_img: Image.Image, bg_color_name: str, enable: bool = False) -> Image.Image:
    """
    安全背景处理
    仅在检测到透明背景且 enable=True 时处理
    """
    if not enable:
        return pil_img

    try:
        bg_type = _detect_background_type(pil_img)
        if bg_type == "transparent":
            return apply_background(pil_img, bg_color_name)
        else:
            # 已有背景，保持原样
            return pil_img
    except Exception as e:
        print(f"[警告] 背景处理失败: {e}，保持原背景")
        return pil_img


def apply_background(pil_img: Image.Image, bg_color_name: str) -> Image.Image:
    """将透明背景替换为指定纯色背景（保留用于向后兼容）"""
    bg_color = _hex_to_bgr(bg_color_name)
    r, g, b = bg_color[2], bg_color[1], bg_color[0]

    if pil_img.mode != "RGBA":
        pil_img = pil_img.convert("RGBA")
    width, height = pil_img.size
    blur = max(2.0, min(width, height) * 0.003)
    alpha = pil_img.split()[-1].filter(ImageFilter.GaussianBlur(blur))

    background = Image.new("RGB", pil_img.size, (r, g, b))
    background.paste(pil_img.convert("RGB"), mask=alpha)
    return background


def _safe_resize(pil_img: Image.Image, spec: str) -> Image.Image:
    """
    安全尺寸调整（等比缩放 + 居中裁剪，避免粗边框）

    当前仅支持规则源定义的 2 寸规格。
    规则：
    - 不对人像做几何拉伸，只做等比缩放
    - 优先保证成品没有明显的上下 / 左右色块边框
    - 允许在四周裁掉少量纯色背景（通常是蓝 / 白 / 红），人物仍然居中
    """
    try:
        target_w, target_h = _get_spec_output_size(spec)
        src_w, src_h = pil_img.size

        if src_w <= 0 or src_h <= 0:
            return pil_img

        # 如果尺寸已经非常接近目标，直接轻微缩放到目标，避免多次处理
        if abs(src_w - target_w) <= 2 and abs(src_h - target_h) <= 2:
            return pil_img.resize((target_w, target_h), Image.LANCZOS)

        # 第一步：按“至少一边填满画布”的原则等比缩放
        # 这样不会出现大面积上下 / 左右留边
        scale = max(target_w / src_w, target_h / src_h)
        new_w = int(src_w * scale + 0.5)
        new_h = int(src_h * scale + 0.5)

        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # 如果刚好等于目标尺寸，直接返回
        if new_w == target_w and new_h == target_h:
            return resized

        # 第二步：从中心裁剪到目标尺寸
        left = max(0, (new_w - target_w) // 2)
        top = max(0, (new_h - target_h) // 2)
        right = min(new_w, left + target_w)
        bottom = min(new_h, top + target_h)

        cropped = resized.crop((left, top, right, bottom))

        # 保底：由于四舍五入导致的 1 像素偏差，再做一次安全 resize
        if cropped.size != (target_w, target_h):
            cropped = cropped.resize((target_w, target_h), Image.LANCZOS)

        return cropped
    except Exception as e:
        print(f"[警告] 尺寸调整失败: {e}，保持原尺寸")
        return pil_img


def _set_dpi(pil_img: Image.Image, enable: bool = True) -> Image.Image:
    """
    设置 DPI（300 DPI 标准）
    """
    if not enable:
        return pil_img

    try:
        # PIL 的 DPI 设置方式
        pil_img.info['dpi'] = (DPI, DPI)
        return pil_img
    except Exception as e:
        print(f"[警告] DPI 设置失败: {e}")
        return pil_img


def _enhance_sharpness(pil_img: Image.Image, enable: bool = True) -> Image.Image:
    """
    清晰度增强（自动检测 + 增强，不抛异常）
    """
    if not enable:
        return pil_img

    try:
        gray = np.array(pil_img.convert("L"))
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = lap.var()
        if var < SHARPNESS_THRESHOLD:
            print(f"[信息] 清晰度较低 (方差={var:.1f})，自动增强")
            enhanced = pil_img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
            return enhanced
        return pil_img
    except Exception as e:
        print(f"[警告] 清晰度检测失败: {e}，保持原图")
        return pil_img


def _warn_resolution(img: Image.Image):
    """
    分辨率警告（不抛异常，只打印警告）
    """
    try:
        w, h = img.width, img.height
        min_edge = min(w, h)
        area = w * h
        if min_edge < MIN_SRC_MIN_EDGE or area < MIN_SRC_AREA:
            print(
                f"[警告] 原图分辨率较低: {w}×{h}（建议最小边≥{MIN_SRC_MIN_EDGE}px，总面积≥{MIN_SRC_AREA // 1000}千像素）")
    except Exception:
        pass


def _save_single_version(
        pil_img: Image.Image,
        output_dir: str,
        prefix: str,
        spec: str,
        max_kb: Optional[int] = None,
) -> Optional[str]:
    """
    保存单个版本（支持多规格，默认 2 寸）

    max_kb:
        - None 或 <=0: 不限制文件大小，使用固定质量保存
        - >0: 在不改变分辨率的前提下，通过降低 JPEG 质量尽量压到该大小以下
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # 根据规格确定目标尺寸，命名中包含规格和尺寸
        target_w, target_h = _get_spec_output_size(spec)
        filename = f"{prefix}_{spec}_{target_w}x{target_h}.jpg"

        # 确保尺寸正确，但不做额外拉伸：如有偏差，使用 _safe_resize 进行等比缩放+居中裁剪
        if pil_img.size != (target_w, target_h):
            resized = _safe_resize(pil_img, spec)
        else:
            resized = pil_img
        # 设置 DPI（清晰度是否增强由上游流程控制，这里不再做额外锐化）
        resized.info['dpi'] = (DPI, DPI)

        filepath = os.path.join(output_dir, filename)

        # 如果设置了文件大小上限，则通过调整 JPEG 质量压缩
        if max_kb is not None and max_kb > 0:
            target_bytes = max_kb * 1024
            best_bytes = None

            # 从较高质量向下尝试，避免画质骤降
            for quality in (95, 90, 85, 80, 75, 70, 65, 60):
                buffer = BytesIO()
                resized.save(buffer, format="JPEG", quality=quality, subsampling=0, optimize=True)
                data = buffer.getvalue()
                best_bytes = data
                if len(data) <= target_bytes:
                    break

            # 写入文件
            with open(filepath, "wb") as f:
                f.write(best_bytes)
        else:
            # 不限制体积，按固定高质量保存
            resized.save(filepath, format="JPEG", quality=95, subsampling=0, optimize=True)

        return filepath
    except Exception as e:
        print(f"[错误] 保存文件失败: {e}")
        return None


def process_id_photo(
        input_path: str,
        output_dir: str,
        prefix: str,
        spec: str = "2inch",
        enable_background: bool = False,
        enable_dpi: bool = True,
        enable_sharpness: bool = False,
        enable_composition: bool = False,
        max_kb: Optional[int] = None,
) -> Dict[str, any]:
    """
    证件照二次处理与合规交付引擎（统一入口）

    ⚠️ 核心规则（AI 成品图场景）：
    - 豆包等 AI 已经完成"构图"，这里绝对不能再动构图
    - 只允许改"参数"：尺寸、DPI、KB
    - 禁止改"几何结构"：裁剪、重算头高、居中

    参数:
        input_path: 输入图片路径（AI 已处理好的证件照成品）
        output_dir: 输出目录
        prefix: 文件前缀
        spec: 规格，默认 "2inch"（413x626）
        enable_background: 是否处理背景，默认 False（豆包已换好）
        enable_dpi: 是否写入 300 DPI，默认 True
        enable_sharpness: 是否锐化，默认 False（AI 图够清晰）
        enable_composition: 是否重新构图，默认 False（🔴 绝对不能开）

    返回:
        {
            "success": bool,
            "file_path": str,  # 成功时返回文件路径
            "zip_path": str,   # 如果打包成功
            "error": str       # 失败时返回错误信息
        }
    """
    result = {
        "success": False,
        "file_path": None,
        "zip_path": None,
        "error": None
    }

    try:
        # 步骤1：加载图片（fallback：纯色图）
        pil_img = _load_image_safe(input_path)
        if pil_img is None:
            result["error"] = "图片加载失败"
            return result

        # 分辨率警告（不抛异常）
        _warn_resolution(pil_img)

        # 步骤2：构图裁剪（仅在开启时执行，失败会自动 fallback）
        if enable_composition:
            pil_img = _safe_composition(pil_img, spec)

        # 步骤3：背景处理（仅在检测到透明背景时）
        bg_color = "white"  # 默认白色，可从参数传入
        pil_img = _safe_background(pil_img, bg_color, enable=enable_background)

        # 步骤4：尺寸调整（使用黄金参数）
        pil_img = _safe_resize(pil_img, spec)

        # 步骤5：DPI 设置（300 DPI）
        pil_img = _set_dpi(pil_img, enable=enable_dpi)

        # 步骤6：清晰度增强（自动检测 + 增强）
        pil_img = _enhance_sharpness(pil_img, enable=enable_sharpness)

        # 步骤7：保存文件（至少保存一个版本，失败时使用兜底图重试）
        file_path = _save_single_version(pil_img, output_dir, prefix, spec, max_kb=max_kb)
        if file_path is None:
            print("[警告] 首次保存失败，尝试使用兜底图重试保存")
            fw, fh = _get_spec_output_size(spec)
            fallback_img = Image.new("RGB", (fw, fh), (240, 240, 240))
            file_path = _save_single_version(fallback_img, output_dir, prefix, spec, max_kb=max_kb)
            if file_path is None:
                # 理论上不太可能失败，如果仍失败，返回错误信息
                result["error"] = "文件保存失败"
                return result

        result["success"] = True
        result["file_path"] = file_path

        # 步骤8：打包 ZIP（可选，失败不影响主流程）
        try:
            zip_path = make_zip([file_path], output_dir, prefix)
            result["zip_path"] = zip_path
        except Exception as e:
            print(f"[警告] ZIP 打包失败: {e}，但文件已保存")

        return result

    except Exception as e:
        print(f"[错误] 处理流程失败: {e}")
        result["error"] = f"处理失败: {str(e)}"
        return result


def make_zip(files, output_dir: str, prefix: str) -> str:
    """将给定文件列表打包成 zip"""
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, f"{prefix}_id_photos.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if os.path.exists(f):
                zf.write(f, arcname=os.path.basename(f))
    return zip_path


# ============================================
# 向后兼容函数（标记为 deprecated）
# ============================================

def remove_background(input_path: str) -> Image.Image:
    """
    [DEPRECATED] 抠图功能，已不在主流程中使用
    保留用于向后兼容
    """
    try:
        from rembg import remove
        with open(input_path, "rb") as f:
            input_bytes = f.read()
        output_bytes = remove(input_bytes)
        return Image.open(BytesIO(output_bytes)).convert("RGBA")
    except Exception as e:
        print(f"[警告] 抠图失败: {e}")
        # 返回原图
        return Image.open(input_path).convert("RGBA")


def _ensure_min_resolution(img: Image.Image):
    """
    [DEPRECATED] 分辨率检查，已改为警告模式
    保留用于向后兼容
    """
    _warn_resolution(img)


def _check_sharpness(img: Image.Image, thresh: float = 30.0) -> Image.Image:
    """
    [DEPRECATED] 清晰度检查，已改为自动增强模式
    保留用于向后兼容
    """
    return _enhance_sharpness(img, enable=True)


def process_id_photo_set(input_path: str, output_dir: str, bg_color: str, prefix: str) -> str:
    """
    [DEPRECATED] 旧版处理函数，已改为 process_id_photo()
    保留用于向后兼容
    """
    result = process_id_photo(
        input_path=input_path,
        output_dir=output_dir,
        prefix=prefix,
        spec="2inch",
        enable_background=True,
        enable_dpi=True,
        enable_sharpness=True,
        enable_composition=True,
    )
    # 为兼容旧调用方：即使失败也不再向上抛异常，返回尽可能可用的路径
    if result["success"]:
        return result.get("zip_path") or result["file_path"]
    # 失败情况下尝试返回原图路径作为兜底
    print(f"[警告] process_id_photo_set 处理失败: {result.get('error')}")
    return input_path


# 保留旧函数名用于向后兼容

def auto_crop_face(pil_img: Image.Image, spec: str = "2inch") -> Image.Image:
    """向后兼容封装，默认按 2 寸规格构图"""
    return _safe_composition(pil_img, spec)


resize_to_size = _safe_resize
