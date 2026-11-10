import os  # 导入操作系统相关的模块，用于文件和目录操作

def _append_log(log_path, message):
    """将删除结果写入日志文件，方便后续审查。"""
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

def remove_empty_folders(directory_path, log_path=None, dry_run=True):
    """
    递归删除指定目录及其子目录中的所有空文件夹。

    参数:
    directory_path (str): 需要清理的目标目录的路径。
    log_path (str): 删除日志文件路径，默认保存在目标目录下。
    dry_run (bool): 为 True 时只列出空目录，不执行删除。
    """
    if log_path is None:
        log_path = os.path.join(directory_path, "remove_empty_folders.log")

    # 预演也记住拟删除子目录，才能展示删除子目录后变空的父目录。
    planned = set()
    # 使用os.walk遍历目录树，设置topdown=False以从底部开始遍历
    for root, dirs, files in os.walk(directory_path, topdown=False):
        # 遍历当前目录中的所有子目录
        for folder in dirs:
            folder_path = os.path.join(root, folder)  # 构建子目录的完整路径
            if os.path.islink(folder_path):
                continue
            try:
                remaining = [
                    name for name in os.listdir(folder_path)
                    if os.path.join(folder_path, name) not in planned
                ]
            except OSError as e:
                print(f"无法检查文件夹: {folder_path}. 错误: {e}")
                continue
            if not remaining:
                if dry_run:
                    planned.add(folder_path)
                    print(f"预演：将删除空文件夹 {folder_path}")
                    continue
                try:
                    os.rmdir(folder_path)  # 删除空的子目录
                    print(f"已删除空文件夹: {folder_path}")  # 打印删除操作（可选）
                    _append_log(log_path, f"DELETED\t{folder_path}")
                except OSError as e:
                    print(f"无法删除文件夹: {folder_path}. 错误: {e}")  # 打印错误信息（可选）
                    _append_log(log_path, f"FAILED\t{folder_path}\t{e}")

# 使用示例
if __name__ == "__main__":
    # 默认只预演。确认列表无误后，再传入 dry_run=False。
    remove_empty_folders('/path/to/directory')
