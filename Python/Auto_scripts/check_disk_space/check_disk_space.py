import shutil  # 导入 shutil 模块，用于检查磁盘的使用情况

# 检查磁盘剩余空间的函数
def check_disk_space(path, threshold):
    """
    检查指定路径所在磁盘的剩余空间，并与指定的阈值进行比较。

    参数:
    path (str): 要检查的路径，例如根目录 '/'。
    threshold (int): 剩余磁盘空间的阈值（以GB为单位）。
    """
    # 使用 shutil.disk_usage() 获取磁盘的总空间、已用空间和剩余空间（单位为字节）
    total, used, free = shutil.disk_usage(path)

    # 将剩余空间从字节转换为GB
    free_gb = free // (2**30)

    # 比较剩余空间与阈值，输出相应的消息
    if free_gb < threshold:
        print(f"Warning: Free disk space is below {threshold} GB.")
    else:
        print(f"Free disk space: {free_gb} GB.")

# 使用示例
if __name__ == "__main__":
    # 调用 check_disk_space 函数，检查根目录的剩余磁盘空间是否低于 10 GB
    check_disk_space('/', 10)
