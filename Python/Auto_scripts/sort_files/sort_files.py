import os  # 导入操作系统相关的模块，用于文件和目录操作
from shutil import move  # 从shutil模块导入move函数，用于移动文件

def sort_files(directory_path, dry_run=True):
    """
    按照文件扩展名对指定目录中的文件进行分类整理。

    参数:
    directory_path (str): 需要整理的目标目录的路径。
    dry_run (bool): 为 True 时只显示移动计划，不修改文件。
    """
    # 遍历目标目录中的所有条目（文件和子目录）
    for filename in os.listdir(directory_path):
        # 构建完整的文件路径
        file_path = os.path.join(directory_path, filename)

        # 检查当前条目是否是文件（而非子目录）
        if os.path.isfile(file_path) and not os.path.islink(file_path):
            # 通过最后一个点分割文件名，获取扩展名
            file_extension = os.path.splitext(filename)[1].lstrip('.').lower() or 'no_extension'

            # 构建目标子目录的路径，子目录名为扩展名
            destination_directory = os.path.join(directory_path, file_extension)

            # 不跟随已有的符号链接，也不把普通文件当作分类目录。
            if os.path.islink(destination_directory) or (
                os.path.lexists(destination_directory) and not os.path.isdir(destination_directory)
            ):
                print(f"跳过，不安全或非目录的分类目标: {destination_directory}")
                continue

            # 构建目标文件的完整路径
            destination_path = os.path.join(destination_directory, filename)

            if os.path.lexists(destination_path):
                print(f"跳过，目标文件已存在: {destination_path}")
                continue

            if dry_run:
                print(f"预演：将移动 {file_path} -> {destination_path}")
                continue

            os.makedirs(destination_directory, exist_ok=True)

            # 移动文件到目标子目录
            move(file_path, destination_path)
            # 打印移动文件的操作（可选）
            # print(f"移动文件: {file_path} --> {destination_path}")

# 使用示例
if __name__ == "__main__":
    # 默认只预演。确认列表无误后，再传入 dry_run=False。
    sort_files('/path/to/directory')
