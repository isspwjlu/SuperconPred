"""共享工具函数。"""
import os


def ensure_dir(path: str) -> None:
    """确保目录存在，如不存在则创建。"""
    os.makedirs(path, exist_ok=True)
