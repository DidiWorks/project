import os
import zipfile
from io import BytesIO
from typing import Tuple
import numpy as np
import mediapipe as mp
from PIL import Image,ImageFilter

# 将 rembg 模型目录固定到项目内的 models 文件夹，避免重复下载
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("U2NET_HOME", os.path.join(BASE_DIR, "models"))

from rembg import remove

# 分辨率 & 构图安全阈值
MIN_SRC_W = 500
MIN_SRC_H = 600
MIN_CROP_W = 295   # 至少不小于 1 寸宽
MIN_CROP_H = 413   # 至少不小于 1 寸高


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


def _ensure_min_resolution(img: Image.Image):
    """最小分辨率保护，避免过小原图导致放大模糊"""
    if img.width < MIN_SRC_W or img.height < MIN_SRC_H:
        raise ValueError("原图分辨率太低，无法生成高清证件照（至少 500x600）")


def apply_background(pil_img: Image.Image, bg_color_name: str) -> Image.Image:
    """将透明背景替换为指定纯色背景，返回 RGB 图片"""
    bg_color = _hex_to_bgr(bg_color_name)
    r, g, b = bg_color[2], bg_color[1], bg_color[0]

    # 分离alpha， 并做轻度高斯模糊 feather
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGBA")
    alpha = pil_img.split()[-1].filter(ImageFilter.GaussianBlur(1.2))

    background = Image.new("RGB", pil_img.size, (r, g, b))
    background.paste(pil_img, mask=pil_img.split()[3])  # 使用 alpha 通道作为 mask
    return background


def auto_crop_face(pil_img: Image.Image) -> Image.Image:
    """
    使用 MediaPipe Face Mesh 自动人脸检测 + 标准证件照构图裁剪。
    - 输入：PIL.Image（建议为抠完背景后的 RGBA / RGB）
    - 输出：裁剪后的 PIL.Image（保持 3:4 比例，不拉伸，超出部分自动补白/透明）
    """
    # 目标参数
    target_head_ratio = 0.55
    eye_height_ratio = 0.40

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

    # 如果没有检测到人脸，直接返回原图
    if not results.multi_face_landmarks:
        return pil_img

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

    # 预计头顶
    head_top_y = min(brow_y, forehead_y)
    head_bottom_y = chin_y
    face_height = head_bottom_y - head_top_y

    # 如果高度异常，直接返回原图
    if face_height <= 0.0:
        return pil_img

    # 估计脸部左右范围中心
    face_left_x = min(left_cheek_x, right_cheek_x)
    face_right_x = max(left_cheek_x, right_cheek_x)
    face_center_x = (face_left_x + face_right_x) / 2.0

    # 估计眼睛大概位置，用头顶和下巴之间 45% 位置近似眼睛高度（可按需微调为 0.40）
    eye_y_est = head_top_y + face_height * 0.45

    # 计算目标裁剪高度：head_height 占整张比例为 target_head_ratio
    crop_height = face_height / max(target_head_ratio, 1e-6)
    # 根据 3:4 比例计算宽度（w/h = 3/4 → width = height * 3/4）
    crop_width = crop_height * (3.0 / 4.0)

    # 根据眼睛位置 反推裁剪上边界，使眼睛在图片 eye_height_ratio 位置
    crop_top = eye_y_est - eye_height_ratio * crop_height
    crop_bottom = crop_top + crop_height

    # 根据脸部中心确定左右，保证 3:4 比例，左右留白自然满足 10–15%
    crop_left = face_center_x - crop_width / 2.0
    crop_right = crop_left + crop_width

    # 将裁剪框坐标转换为整数
    crop_left_f = float(crop_left)
    crop_top_f = float(crop_top)
    crop_right_f = float(crop_right)
    crop_bottom_f = float(crop_bottom)

    dest_width = int(round(crop_width))
    dest_height = int(round(crop_height))

    # 如果计算结果异常，返回原图
    if dest_width <= 0 or dest_height <= 0:
        return pil_img

    # 若裁剪框过小（小于目标成品尺寸），采用扩白而非拉伸放大
    if dest_width < MIN_CROP_W or dest_height < MIN_CROP_H:
        dest_width = max(dest_width, MIN_CROP_W)
        dest_height = max(dest_height, MIN_CROP_H)

    # 生成目标画布：保持原图模式
    if img_mode == "RGBA":
        bg_color = (0, 0, 0, 0)  # 透明
    else:
        bg_color = (255, 255, 255)

    dst_img = Image.new(img_mode, (dest_width, dest_height), bg_color)

    # 计算与原图的交集区域，用于裁剪
    src_left = max(0, int(round(crop_left_f)))
    src_top = max(0, int(round(crop_top_f)))
    src_right = min(w, int(round(crop_right_f)))
    src_bottom = min(h, int(round(crop_bottom_f)))

    if src_right <= src_left or src_bottom <= src_top:
        # 裁剪区域完全落在图像外，返回原图
        return pil_img
    # 从原图中裁剪实际可见部分
    src_crop = pil_img.crop((src_left, src_top, src_right, src_bottom))

    # 计算将在这一块贴到目标画布上的位置
    paste_x = src_left - int(round(crop_left_f))
    paste_y = src_top - int(round(crop_top_f))

    dst_img.paste(src_crop, (paste_x, paste_y))

    return dst_img

    # 若检测不到人脸，回退：原图等比缩放放入 3:4 画布（白底）

    if not results.multi_face_landmarks:
        return _fallback_center_canvas(pil_img)

    # 计算关键点同之前...
    # head_top_y, head_bottom_y, face_height, face_center_x, eye_y_est

    # 头顶留白 5%~12%，下巴到底部 25%~35%，左右留白 >=5%
    # 先按头部比例算裁剪框，再施加约束
    crop_height = face_height / max(target_head_ratio, 1e-6)
    crop_width = crop_height *(3.0 / 4.0)

    #眼睛位置反推 top/bottom
    crop_top = eye_y_est - eye_height_ratio * crop_height
    crop_bottom = crop_top + crop_height

    # 头顶留白约束
    desired_headroom_min = 0.05*crop_height
    desired_headroom_max = 0.12*crop_height


def _fall_back_center_canvas(pil_img:Image.Image) ->Image.Image:
    """人脸检测失败的安全回退：将原图等比缩放放入 3:4 画布白底"""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    target_h = max(MIN_CROP_H,int(h))
    target_w = int(target_h * 3 / 4)
    canvas = Image.new("RGB",(target_w, target_h),(255,255,255))
    scale = min(target_w / w,target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def resize_to_size(pil_img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """统一使用 LANCZOS，避免像素化/锯齿"""
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
    img_1.save(path_1, format="JPEG", quality=95, subsampling=0, optimize=True)
    img_2.save(path_2, format="JPEG", quality=95, subsampling=0, optimize=True)

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
    """完整流程：抠图 + 自动裁剪 + 换底色 + 生成 1 寸、2 寸 + 打包 zip"""
    # 0. 最小分辨率保护（原图太小直接报错）
    with Image.open(input_path) as _src_check:
        _ensure_min_resolution(_src_check)

    # 1. 抠图
    pil_transparent = remove_background(input_path)

    # 2. 自动标准构图裁剪（以人脸关键点为基准）
    pil_transparent = auto_crop_face(pil_transparent)

    # 3. 换底色
    pil_bg = apply_background(pil_transparent, bg_color)

    # 4. 保存 1 寸 / 2 寸（固定像素，不拉伸到更低分辨率）
    path_1, path_2 = save_versions(pil_bg, output_dir, prefix)

    # 5. 打包 zip（顺便把抠完图的 PNG 原图也一起打包）
    alpha_path = os.path.join(output_dir, f"{prefix}_alpha.png")
    pil_transparent.save(alpha_path, format="PNG")

    zip_path = make_zip([path_1, path_2, alpha_path], output_dir, prefix)
    return zip_path


