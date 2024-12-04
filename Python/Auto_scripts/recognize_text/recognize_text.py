import pytesseract  # 导入 pytesseract 库，用于图像文字识别（OCR）
from PIL import Image  # 从 PIL 库（Pillow）中导入 Image 模块，用于处理图像

# 从图像中识别文本的函数
def recognize_text(image_path):
    """
    识别指定图像中的文本内容。

    参数:
    image_path (str): 图像文件的路径。

    返回:
    str: 识别出的文本内容。
    """
    # 打开输入的图像文件
    image = Image.open(image_path)

    # 使用 pytesseract 对图像进行 OCR，提取文本
    text = pytesseract.image_to_string(image, lang='chi_sim')  # 使用简体中文语言识别

    return text  # 返回识别出的文本内容

# 使用示例
if __name__ == "__main__":
    # 调用 recognize_text 函数，识别指定图像中的文本内容
    text = recognize_text('/path/to/image.jpg')  # 请将 '/path/to/image.jpg' 替换为实际的图像路径
    # 打印提取出的文本内容
    print(text)
