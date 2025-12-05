from PIL import Image
from utils.image_tools import auto_crop_face

def main():
    # 确保项目根目录里有 test.jpg（随便放一张人像）
    img = Image.open("test.jpg")
    out = auto_crop_face(img)
    out.save("cropped_test.jpg")
    print("完成，输出 cropped_test.jpg")

if __name__ == "__main__":
    main()


