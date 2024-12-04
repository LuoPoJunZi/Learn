import os  # 导入操作系统相关的模块，用于文件和目录操作
from shutil import move  # 从shutil模块导入move函数，用于移动文件

def sort_files(directory_path):
    """
    按照文件扩展名对指定目录中的文件进行分类整理。

    参数:
    directory_path (str): 需要整理的目标目录的路径。
    """
    # 遍历目标目录中的所有条目（文件和子目录）
    for filename in os.listdir(directory_path):
        # 构建完整的文件路径
        file_path = os.path.join(directory_path, filename)
        
        # 检查当前条目是否是文件（而非子目录）
        if os.path.isfile(file_path):
            # 通过最后一个点分割文件名，获取扩展名
            file_extension = filename.split('.')[-1]
            
            # 构建目标子目录的路径，子目录名为扩展名
            destination_directory = os.path.join(directory_path, file_extension)
            
            # 如果目标子目录不存在，则创建它
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
                # 打印创建子目录的操作（可选）
                # print(f"创建目录: {destination_directory}")
            
            # 构建目标文件的完整路径
            destination_path = os.path.join(destination_directory, filename)
            
            # 移动文件到目标子目录
            move(file_path, destination_path)
            # 打印移动文件的操作（可选）
            # print(f"移动文件: {file_path} --> {destination_path}")

# 使用示例
if __name__ == "__main__":
    # 调用sort_files函数，传入要整理的目录路径
    sort_files('/path/to/directory')  # 请将'/path/to/directory'替换为实际路径
