import os  # 导入操作系统相关的模块，用于文件和目录操作

def _append_log(log_path, message):
    """将重命名结果写入日志文件，方便后续审查。"""
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(message + "\n")

def rename_files(directory_path, old_name, new_name, log_path=None):
    """
    在指定目录中查找包含特定子字符串的文件，并将其重命名。

    参数:
    directory_path (str): 需要操作的目标目录的路径。
    old_name (str): 文件名中需要被替换的旧子字符串。
    new_name (str): 用于替换的新子字符串。
    log_path (str): 重命名日志文件路径，默认保存在目标目录下。
    """
    if log_path is None:
        log_path = os.path.join(directory_path, "rename_files.log")

    # 遍历目标目录中的所有文件和子目录
    for filename in os.listdir(directory_path):
        if os.path.abspath(os.path.join(directory_path, filename)) == os.path.abspath(log_path):
            continue

        # 检查当前项是否为文件，避免对目录进行重命名
        if os.path.isfile(os.path.join(directory_path, filename)):
            # 检查文件名中是否包含旧的子字符串
            if old_name in filename:
                # 创建新的文件名，将旧子字符串替换为新的子字符串
                new_filename = filename.replace(old_name, new_name)
                
                # 构建完整的旧文件路径和新文件路径
                old_file = os.path.join(directory_path, filename)
                new_file = os.path.join(directory_path, new_filename)

                if os.path.exists(new_file):
                    message = f"跳过重命名，目标文件已存在: '{new_filename}'"
                    print(message)
                    _append_log(log_path, f"SKIPPED_EXISTS\t{old_file}\t{new_file}")
                    continue
                
                try:
                    # 执行重命名操作
                    os.rename(old_file, new_file)
                    print(f"已重命名: '{filename}' -> '{new_filename}'")  # 打印成功信息
                    _append_log(log_path, f"RENAMED\t{old_file}\t{new_file}")
                except OSError as e:
                    # 如果重命名失败，打印错误信息
                    print(f"无法重命名文件: '{filename}'. 错误: {e}")
                    _append_log(log_path, f"FAILED\t{old_file}\t{new_file}\t{e}")

# 使用示例
if __name__ == "__main__":
    # 调用rename_files函数，传入要操作的目录路径、旧子字符串和新子字符串
    rename_files('/path/to/directory', 'old', 'new')  # 请将'/path/to/directory'、'old'和'new'替换为实际值
