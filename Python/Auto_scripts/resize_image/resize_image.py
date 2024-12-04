from PIL import Image  # 从 PIL 库（Pillow）中导入 Image 模块，用于处理图像

# 调整图像尺寸的函数
def resize_image(input_path, output_path, width, height):
    """
    将输入的图像调整为指定宽度和高度，并保存到输出路径。

    参数:
    input_path (str): 输入图像的文件路径。
    output_path (str): 调整后图像的保存路径。
    width (int): 调整后图像的宽度（像素）。
    height (int): 调整后图像的高度（像素）。
    """
    # 打开输入图像文件
    image = Image.open(input_path)

    # 调整图像的大小，使用 ANTIALIAS 算法提高质量
    resized_image = image.resize((width, height), Image.ANTIALIAS)

    # 保存调整后的图像到指定输出路径
    resized_image.save(output_path)

# 使用示例
if __name__ == "__main__":
    # 调用 resize_image 函数，将输入图像调整为800x600像素
    resize_image('/path/to/input.jpg', '/path/to/output.jpg', 800, 600)  # 请将路径替换为实际的输入和输出图像路径
