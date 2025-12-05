from flask import Flask, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import uuid
import base64
from PIL import Image

from utils.image_tools import process_id_photo_set

app = Flask(__name__)
app.secret_key = "ai_id_photo_demo_secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "upload")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "output")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """处理上传图片，生成 1 寸、2 寸证件照，并返回下载链接"""
    if "photo" not in request.files:
        flash("请先选择要上传的图片文件。")
        return redirect(url_for("index"))

    file = request.files["photo"]
    bg_color = request.form.get("bg_color", "white")

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
    file.save(upload_path)

    # 最小分辨率保护，避免低清原图导致放大模糊
    try:
        with Image.open(upload_path) as _img_check:
            if _img_check.width < 500 or _img_check.height < 600:
                flash("原图分辨率太低（至少 500x600），请更换清晰原图。")
                return redirect(url_for("index"))
    except Exception:
        flash("图片文件读取失败，请更换清晰原图。")
        return redirect(url_for("index"))

    try:
        zip_path = process_id_photo_set(
            input_path=upload_path,
            output_dir=OUTPUT_FOLDER,
            bg_color=bg_color,
            prefix=unique_id,
        )

    except Exception as e:
        print("处理图片时出错：", e)
        flash("图片处理失败，请确认图片内容清晰，然后稍后重试。")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        download_url=url_for("download", zip_name=os.path.basename(zip_path)),
    )

@app.route("/preview", methods=["POST"])
def preview():
    """预览功能：只生成预览图，不保存文件"""
    if "photo" not in request.files:
        return {"error": "请先选择要上传的图片文件。"}, 400
    file = request.files["photo"]
    bg_color = request.form.get("bg_color", "white")

    if file.filename == "":
        return {"error": "未选择文件，请重新上传。"}, 400
    if not allowed_file(file.filename):
        return {"error": "仅支持 jpg、jpeg、png 格式的图片。"}, 400

    # 临时保存文件用于处理
    filename = secure_filename(file.filename)
    temp_id = uuid.uuid4().hex
    temp_upload_path = os.path.join(UPLOAD_FOLDER, f"temp_{temp_id}_{filename}")
    file.save(temp_upload_path)

    # 最小分辨率保护
    try:
        with Image.open(temp_upload_path) as _img_check:
            if _img_check.width < 500 or _img_check.height < 600:
                if os.path.exists(temp_upload_path):
                    os.remove(temp_upload_path)
                return {"error": "原图分辨率太低（至少 500x600），请更换清晰原图。"}, 400
    except Exception:
        if os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
        return {"error": "图片文件读取失败，请更换清晰原图。"}, 400

    try:
        # 只做抠图、自动裁剪和换底色，不保存文件
        from utils.image_tools import remove_background, auto_crop_face, apply_background, resize_to_size

        # 1. 抠图
        pil_transparent = remove_background(temp_upload_path)

        # 2. 自动标准构图裁剪（与最终生成流程一致）
        pil_transparent = auto_crop_face(pil_transparent)

        # 3. 换底色
        pil_bg = apply_background(pil_transparent, bg_color)

        # 4. 生成 1 寸预览图（内存中不保存）
        size_1inch = (295, 413)
        pil_preview = resize_to_size(pil_bg, size_1inch)

        # 5. 转为 base64
        from io import BytesIO
        buffer = BytesIO()
        pil_preview.save(buffer, format="JPEG", quality=95)
        img_bytes = buffer.getvalue()
        preview_base64 = base64.b64encode(img_bytes).decode('utf-8')

        # 清理临时文件
        if os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
        return {
            "success": True,
            "preview_base64": preview_base64,
            "preview_mime_type": "image/jpeg",
        }
    except Exception as e:
        print("预览处理时出错: ", e)
        # 清理临时文件
        if os.path.exists(temp_upload_path):
            os.remove(temp_upload_path)
        return {"error": "图片处理失败，请确认图片内容清晰，然后稍后重试。"}, 500





@app.route("/download/<zip_name>", methods=["GET"])
def download(zip_name):
    """提供 zip 文件下载"""
    zip_path = os.path.join(OUTPUT_FOLDER, zip_name)
    if not os.path.exists(zip_path):
        flash("要下载的文件不存在，请重新生成证件照。")
        return redirect(url_for("index"))

    return send_file(zip_path, as_attachment=True, download_name=zip_name)


if __name__ == "__main__":
    # 开发环境下使用 debug 模式，方便调试
    app.run(host="0.0.0.0", port=5000, debug=True)
