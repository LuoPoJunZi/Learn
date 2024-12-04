import requests  # 导入requests库，用于发送HTTP请求

def download_images(url, save_directory):
    """
    从指定的API获取图片URL列表并下载图片到本地目录。
    
    参数:
    url (str): API的URL地址，假设返回一个包含图片URL的JSON数组。
    save_directory (str): 图片保存的本地目录路径。
    """
    try:
        # 发送HTTP GET请求到指定的API URL
        response = requests.get(url)
        # 检查请求是否成功（状态码200）
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # 如果请求过程中发生异常，打印错误信息并退出函数
        print(f"请求失败: {e}")
        return

    try:
        # 假设API返回一个包含图片URL的JSON数组
        images = response.json()
    except ValueError as e:
        # 如果解析JSON失败，打印错误信息并退出函数
        print(f"解析JSON失败: {e}")
        return

    # 检查保存目录是否存在，如果不存在则创建
    if not os.path.exists(save_directory):
        try:
            os.makedirs(save_directory)
            print(f"已创建保存目录: {save_directory}")
        except OSError as e:
            print(f"无法创建目录: {save_directory}. 错误: {e}")
            return

    # 遍历每个图片URL并下载图片
    for index, image_url in enumerate(images):
        try:
            # 发送HTTP GET请求下载图片
            image_response = requests.get(image_url)
            # 检查图片请求是否成功
            image_response.raise_for_status()
        except requests.exceptions.RequestException as e:
            # 如果下载图片失败，打印错误信息并跳过当前图片
            print(f"下载图片失败: {image_url}. 错误: {e}")
            continue

        # 构建图片的保存路径，格式为 image_索引.jpg
        image_path = os.path.join(save_directory, f"image_{index}.jpg")
        try:
            # 以二进制写入模式打开文件并保存图片内容
            with open(image_path, "wb") as f:
                f.write(image_response.content)
            print(f"已保存图片: {image_path}")
        except OSError as e:
            # 如果保存图片失败，打印错误信息
            print(f"无法保存图片: {image_path}. 错误: {e}")

# 使用示例
if __name__ == "__main__":
    # 定义API的URL地址和图片保存的本地目录路径
    api_url = 'https://api.example.com/images'  # 请替换为实际的API URL
    save_dir = '/path/to/save'  # 请替换为实际的保存目录路径

    # 调用download_images函数下载图片
    download_images(api_url, save_dir)
