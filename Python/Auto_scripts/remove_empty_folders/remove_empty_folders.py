import os  # 导入操作系统相关的模块，用于文件和目录操作

def remove_empty_folders(directory_path):
    """
    递归删除指定目录及其子目录中的所有空文件夹。

    参数:
    directory_path (str): 需要清理的目标目录的路径。
    """
    # 使用os.walk遍历目录树，设置topdown=False以从底部开始遍历
    for root, dirs, files in os.walk(directory_path, topdown=False):
        # 遍历当前目录中的所有子目录
        for folder in dirs:
            folder_path = os.path.join(root, folder)  # 构建子目录的完整路径
            if not os.listdir(folder_path):  # 检查子目录是否为空
                try:
                    os.rmdir(folder_path)  # 删除空的子目录
                    print(f"已删除空文件夹: {folder_path}")  # 打印删除操作（可选）
                except OSError as e:
                    print(f"无法删除文件夹: {folder_path}. 错误: {e}")  # 打印错误信息（可选）

# 使用示例
if __name__ == "__main__":
    # 调用remove_empty_folders函数，传入要清理的目录路径
    remove_empty_folders('/path/to/directory')  # 请将'/path/to/directory'替换为实际路径
