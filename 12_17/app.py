from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import uuid
import base64
from PIL import Image
from io import BytesIO

from utils.image_tools import process_id_photo
from utils.composition_params import IDPHOTO_RULES, SIZE_PRESETS

app = Flask(__name__)
app.secret_key = "ai_id_photo_demo_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "upload")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "output")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# 从规则源中读取展示用的基础参数（默认 2 寸）
OUTPUT_W, OUTPUT_H = IDPHOTO_RULES["output_size"]
OUTPUT_DPI = IDPHOTO_RULES["dpi"]


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    处理上传图片，生成合规证件照成片。
    假定图片已由外部 AI（如豆包）完成换背景/换衣服，本服务只做尺寸/DPI/清晰度等交付级处理。
    """
    if "photo" not in request.files:
        flash("请先选择要上传的图片文件。")
        return redirect(url_for("index"))

    file = request.files["photo"]
    # 证件照尺寸规格（默认 2 寸）
    spec = request.form.get("spec", "2inch")
    if spec not in SIZE_PRESETS:
        spec = "2inch"

    # 照片类型：AI 成品 / 原始自拍
    photo_type = request.form.get("photo_type", "ai")
    is_ai = (photo_type == "ai")

    # 文件大小限制
    size_limit = request.form.get("size_limit", "none")
    max_kb = None
    if size_limit == "200":
        max_kb = 200
    elif size_limit == "100":
        max_kb = 100

    if file.filename == "":
        flash("未选择文件，请重新上传。")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("仅支持 jpg、jpeg、png 格式的图片。")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    unique_id = uuid.uuid4().hex
    upload_name = f"{unique_id}_{filename}"
    upload_path = os.path.join(UPLOAD_FOLDER, upload_name)

    try:
        # 保存上传文件
        file.save(upload_path)
    except Exception as e:
        print(f"[错误] 文件保存失败: {e}")
        flash("文件上传失败，请重试。")
        return redirect(url_for("index"))

    # 根据照片类型设置处理策略
    # AI 成品：不动构图，只改尺寸 / DPI / KB，默认不锐化
    # 原始照片：开启标准构图裁剪 + 允许轻微锐化
    enable_background = False
    enable_composition = False if is_ai else True
    enable_sharpness = False if is_ai else True

    result = process_id_photo(
        input_path=upload_path,
        output_dir=OUTPUT_FOLDER,
        prefix=unique_id,
        spec=spec,
        enable_background=enable_background,
        enable_dpi=True,             # 写入 300 DPI
        enable_sharpness=enable_sharpness,
        enable_composition=enable_composition,
        max_kb=max_kb,
    )

    # 处理结果
    if result["success"]:
        # 优先返回 ZIP，否则返回单个文件
        download_path = result.get("zip_path") or result.get("file_path")
        if download_path:
            zip_name = os.path.basename(download_path)
            return render_template(
                "index.html",
                download_url=url_for("download", zip_name=zip_name),
            )
        else:
            flash("处理完成，但文件路径异常。")
            return redirect(url_for("index"))
    else:
        # 处理失败，返回友好提示
        error_msg = result.get("error", "图片处理失败，请确认图片内容清晰，然后稍后重试。")
        flash(error_msg)
        return redirect(url_for("index"))


@app.route("/preview", methods=["POST"])
def preview():
    """
    预览功能：生成预览图，不保存最终文件。
    所见即所得：与正式处理走同一条“标准构图 + 尺寸 + DPI + 清晰度”链路。
    """
    if "photo" not in request.files:
        return {"error": "请先选择要上传的图片文件。"}, 400

    file = request.files["photo"]
    # 预览使用与正式处理相同的规格，做到所见即所得
    spec = request.form.get("spec", "2inch")
    if spec not in SIZE_PRESETS:
        spec = "2inch"

    # 照片类型与体积限制，与正式处理保持一致
    photo_type = request.form.get("photo_type", "ai")
    is_ai = (photo_type == "ai")

    size_limit = request.form.get("size_limit", "none")
    max_kb = None
    if size_limit == "200":
        max_kb = 200
    elif size_limit == "100":
        max_kb = 100

    if file.filename == "":
        return {"error": "未选择文件，请重新上传。"}, 400

    if not allowed_file(file.filename):
        return {"error": "仅支持 jpg、jpeg、png 格式的图片。"}, 400

    # 临时保存文件（只用短 ID，避免 Windows 路径过长）
    temp_id = uuid.uuid4().hex[:8]
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "jpg"
    temp_upload_path = os.path.join(UPLOAD_FOLDER, f"tmp_{temp_id}.{ext}")

    try:
        file.save(temp_upload_path)
    except Exception as e:
        print(f"[错误] 临时文件保存失败: {e}")
        return {"error": "文件上传失败。"}, 500

    try:
        # 根据照片类型设置处理策略（与正式处理保持一致，预览即所见即所得）
        enable_background = False
        enable_composition = False if is_ai else True
        enable_sharpness = False if is_ai else True

        result = process_id_photo(
            input_path=temp_upload_path,
            output_dir=OUTPUT_FOLDER,
            prefix=f"tmp_{temp_id}",
            spec=spec,
            enable_background=enable_background,
            enable_dpi=True,             # 预览也写入 DPI，方便展示合规信息
            enable_sharpness=enable_sharpness,
            enable_composition=enable_composition,
            max_kb=max_kb,
        )

        if not result["success"]:
            return {"error": result.get("error", "预览生成失败。")}, 500

        # 读取生成的文件并转为 base64，同时返回合规信息
        preview_path = result.get("file_path")
        if preview_path and os.path.exists(preview_path):
            # 直接读取最终成片文件的字节与信息，保证预览与下载完全一致
            with open(preview_path, "rb") as f:
                img_bytes = f.read()
            preview_base64 = base64.b64encode(img_bytes).decode("utf-8")

            width = OUTPUT_W
            height = OUTPUT_H
            dpi_value = OUTPUT_DPI
            img_format = "JPG"
            try:
                with Image.open(BytesIO(img_bytes)) as img:
                    width, height = img.size
                    dpi_info = img.info.get("dpi")
                    if isinstance(dpi_info, tuple) and len(dpi_info) >= 1 and dpi_info[0]:
                        dpi_value = int(dpi_info[0])
                    img_format = (img.format or "JPEG").upper()
            except Exception:
                pass

            size_kb = len(img_bytes) // 1024

            # 清理临时文件
            try:
                if os.path.exists(temp_upload_path):
                    os.remove(temp_upload_path)
                if os.path.exists(preview_path):
                    os.remove(preview_path)
            except Exception:
                pass

            return {
                "success": True,
                "preview_base64": preview_base64,
                "preview_mime_type": "image/jpeg",
                "meta": {
                    "width": width,
                    "height": height,
                    "dpi": dpi_value,
                    "file_size_kb": size_kb,
                    "format": img_format,
                    "spec": spec,
                    "size_limit_kb": max_kb,
                    "photo_type": photo_type,
                },
            }
        else:
            return {"error": "预览文件生成失败。"}, 500

    except Exception as e:
        print(f"[错误] 预览处理失败: {e}")
        # 清理临时文件
        try:
            if os.path.exists(temp_upload_path):
                os.remove(temp_upload_path)
        except Exception:
            pass
        return {"error": "预览生成失败，请稍后重试。"}, 500


@app.route("/download/<zip_name>", methods=["GET"])
def download(zip_name):
    """提供文件下载（ZIP 或单个文件）"""
    file_path = os.path.join(OUTPUT_FOLDER, zip_name)
    if not os.path.exists(file_path):
        flash("要下载的文件不存在，请重新生成证件照。")
        return redirect(url_for("index"))

    return send_file(file_path, as_attachment=True, download_name=zip_name)


if __name__ == "__main__":
    # 开发环境下使用 debug 模式，方便调试
    app.run(host="0.0.0.0", port=5000, debug=True)