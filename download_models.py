#!/usr/bin/env python3
"""
ComfyUI 模型下载脚本 - 多镜像加速版

使用方法:
    # 下载单个文件
    python download_models.py <url> [保存路径]

    # 示例
    python download_models.py https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
    python download_models.py https://huggingface.co/xxx/model.safetensors Z:/models/checkpoints/model.safetensors

功能:
    1. 多镜像并行分片下载（加速）
    2. 自动测速选择最快镜像
    3. 支持断点续传
    4. 检查文件大小，已存在且大小正确则跳过
"""

import re
import os
import sys
import argparse
import requests
import threading
import time
from pathlib import Path
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# 默认保存目录
MODELS_DIR = Path("Z:/models")

# 多镜像源（会自动测速选择最快的）
MIRRORS = [
    "huggingface.co",      # 原始源
    "hf-mirror.com",       # 国内镜像1
    # "huggingface.bytedance.net",  # 字节镜像（如果可用）
]

# 下载配置
TIMEOUT = (10, 60)  # (连接超时, 读取超时)
CHUNK_SIZE = 1024 * 1024  # 1MB
NUM_THREADS = 4  # 每个文件的并行下载线程数
MAX_RETRIES = 5


def get_filename_from_url(url: str) -> str:
    """从 URL 提取文件名"""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    return os.path.basename(path)


def get_save_path_from_url(url: str, models_dir: Path = None) -> Path:
    """根据 URL 智能推断保存路径"""
    if models_dir is None:
        models_dir = MODELS_DIR

    filename = get_filename_from_url(url)

    # 根据 URL 路径推断目录
    url_lower = url.lower()
    if 'text_encoder' in url_lower or 'clip' in url_lower:
        subdir = 'text_encoders'
    elif 'diffusion_model' in url_lower or 'unet' in url_lower:
        subdir = 'diffusion_models'
    elif 'vae' in url_lower:
        subdir = 'vae'
    elif 'lora' in url_lower:
        subdir = 'loras'
    elif 'controlnet' in url_lower:
        subdir = 'controlnet'
    elif 'checkpoint' in url_lower or 'ckpt' in url_lower:
        subdir = 'checkpoints'
    else:
        # 默认放 checkpoints
        subdir = 'checkpoints'

    return models_dir / subdir / filename


def test_mirror_speed(url: str, mirror: str) -> tuple[str, float]:
    """测试镜像速度，返回 (镜像, 速度MB/s)"""
    test_url = url.replace("huggingface.co", mirror)
    try:
        start = time.time()
        response = requests.get(
            test_url,
            headers={'Range': 'bytes=0-1048575'},  # 下载1MB测速
            timeout=TIMEOUT,
            stream=True
        )
        if response.status_code in (200, 206):
            data = response.content
            elapsed = time.time() - start
            speed = len(data) / elapsed / 1024 / 1024  # MB/s
            return (mirror, speed)
    except:
        pass
    return (mirror, 0)


def select_best_mirrors(url: str, count: int = 2) -> list[str]:
    """测速并选择最快的镜像"""
    print("  [测速] 测试镜像速度...")

    results = []
    with ThreadPoolExecutor(max_workers=len(MIRRORS)) as executor:
        futures = {executor.submit(test_mirror_speed, url, m): m for m in MIRRORS}
        for future in as_completed(futures):
            mirror, speed = future.result()
            if speed > 0:
                results.append((mirror, speed))
                print(f"    {mirror}: {speed:.2f} MB/s")

    # 按速度排序
    results.sort(key=lambda x: x[1], reverse=True)
    best = [r[0] for r in results[:count]]

    if not best:
        print("  [警告] 所有镜像不可用，使用原始源")
        return ["huggingface.co"]

    print(f"  [选择] 使用镜像: {', '.join(best)}")
    return best


def get_file_info(url: str, mirrors: list[str]) -> tuple[int, bool]:
    """获取文件大小和是否支持分片下载"""
    for mirror in mirrors:
        test_url = url.replace("huggingface.co", mirror)
        try:
            response = requests.head(test_url, timeout=TIMEOUT[0], allow_redirects=True)
            if response.status_code == 200:
                size = int(response.headers.get('content-length', 0))
                accept_ranges = response.headers.get('accept-ranges', '').lower() == 'bytes'
                return (size, accept_ranges)
        except:
            continue
    return (0, False)


def check_file_complete(save_path: Path, expected_size: int) -> bool:
    """检查本地文件是否完整"""
    if not save_path.exists():
        return False
    return save_path.stat().st_size == expected_size


def download_range(url: str, start: int, end: int, mirrors: list[str],
                   result: bytearray, offset: int, pbar: tqdm, lock: threading.Lock) -> bool:
    """下载指定范围的数据"""
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Range': f'bytes={start}-{end}'
    }

    for retry in range(MAX_RETRIES):
        # 轮流尝试不同镜像
        mirror = mirrors[retry % len(mirrors)]
        download_url = url.replace("huggingface.co", mirror)

        try:
            response = requests.get(
                download_url,
                headers=headers,
                timeout=TIMEOUT,
                stream=True
            )

            if response.status_code not in (200, 206):
                continue

            pos = 0
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    result[offset + pos:offset + pos + len(chunk)] = chunk
                    pos += len(chunk)
                    with lock:
                        pbar.update(len(chunk))

            return True

        except Exception as e:
            if retry < MAX_RETRIES - 1:
                time.sleep(1)
            continue

    return False


def download_file_multithread(url: str, save_path: Path, mirrors: list[str],
                               file_size: int, num_threads: int = NUM_THREADS) -> bool:
    """多线程分片下载"""
    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 计算分片
    part_size = file_size // num_threads
    ranges = []
    for i in range(num_threads):
        start = i * part_size
        end = file_size - 1 if i == num_threads - 1 else (i + 1) * part_size - 1
        ranges.append((start, end))

    # 预分配内存
    result = bytearray(file_size)
    lock = threading.Lock()

    print(f"  [下载] {num_threads} 线程并行下载...")

    with tqdm(total=file_size, unit='B', unit_scale=True,
              desc=save_path.name[:30], ncols=80) as pbar:

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, (start, end) in enumerate(ranges):
                # 分配不同镜像给不同线程
                thread_mirrors = mirrors[i % len(mirrors):] + mirrors[:i % len(mirrors)]
                future = executor.submit(
                    download_range, url, start, end,
                    thread_mirrors, result, start, pbar, lock
                )
                futures.append(future)

            # 等待所有完成
            success = all(f.result() for f in futures)

    if success:
        # 写入文件
        with open(save_path, 'wb') as f:
            f.write(result)
        print(f"  [完成] {save_path.name}")
        return True
    else:
        print(f"  [失败] 部分分片下载失败")
        return False


def download_file_simple(url: str, save_path: Path, mirrors: list[str]) -> bool:
    """简单单线程下载（用于小文件或不支持分片的情况）"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.with_suffix(save_path.suffix + '.downloading')

    # 断点续传
    downloaded_size = 0
    if temp_path.exists():
        downloaded_size = temp_path.stat().st_size
        print(f"  [续传] 已下载 {downloaded_size / 1024 / 1024:.1f} MB")

    headers = {'User-Agent': 'Mozilla/5.0'}
    if downloaded_size > 0:
        headers['Range'] = f'bytes={downloaded_size}-'

    for retry in range(MAX_RETRIES):
        mirror = mirrors[retry % len(mirrors)]
        download_url = url.replace("huggingface.co", mirror)

        try:
            print(f"  [连接] {mirror}")
            response = requests.get(
                download_url,
                headers=headers,
                stream=True,
                timeout=TIMEOUT
            )

            if response.status_code == 416:
                if temp_path.exists():
                    temp_path.rename(save_path)
                return True

            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            if downloaded_size > 0:
                total_size += downloaded_size

            mode = 'ab' if downloaded_size > 0 else 'wb'
            with open(temp_path, mode) as f:
                with tqdm(total=total_size, initial=downloaded_size,
                          unit='B', unit_scale=True,
                          desc=save_path.name[:30], ncols=80) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            temp_path.rename(save_path)
            print(f"  [完成] {save_path.name}")
            return True

        except Exception as e:
            print(f"  [重试 {retry + 1}/{MAX_RETRIES}] {e}")
            if temp_path.exists():
                downloaded_size = temp_path.stat().st_size
                headers['Range'] = f'bytes={downloaded_size}-'
            time.sleep(2 ** retry)

    return False


def download_file(url: str, save_path: Path, num_threads: int = NUM_THREADS) -> bool:
    """智能下载：根据情况选择多线程或单线程"""

    # 测速选择最佳镜像
    mirrors = select_best_mirrors(url, count=min(num_threads, len(MIRRORS)))

    # 获取文件信息
    file_size, supports_range = get_file_info(url, mirrors)

    if file_size == 0:
        print("  [错误] 无法获取文件大小")
        return False

    print(f"  [信息] 文件大小: {file_size / 1024 / 1024:.1f} MB, 支持分片: {supports_range}")

    # 检查是否已存在
    if check_file_complete(save_path, file_size):
        print(f"  [跳过] 文件已存在且完整")
        return True

    # 大于 100MB 且支持分片，使用多线程
    if file_size > 100 * 1024 * 1024 and supports_range and len(mirrors) > 1:
        return download_file_multithread(url, save_path, mirrors, file_size, num_threads)
    else:
        return download_file_simple(url, save_path, mirrors)


def main():
    parser = argparse.ArgumentParser(
        description='ComfyUI 模型下载工具 - 多镜像加速版',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
  %(prog)s https://huggingface.co/xxx/model.safetensors Z:/models/checkpoints/model.safetensors
  %(prog)s -t 8 https://huggingface.co/xxx/model.safetensors
        '''
    )
    parser.add_argument('url', help='HuggingFace 模型下载 URL')
    parser.add_argument('save_path', nargs='?', help='保存路径（可选，默认根据 URL 自动推断）')
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='并行下载线程数（默认 4）')
    parser.add_argument('-o', '--output-dir', type=str, default="Z:/models",
                        help='模型保存根目录（默认 Z:/models）')

    args = parser.parse_args()

    # 更新配置
    num_threads = args.threads
    models_dir = Path(args.output_dir)

    # 确定保存路径
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        save_path = get_save_path_from_url(args.url, models_dir)

    print("=" * 60)
    print("ComfyUI 模型下载工具 (多镜像加速版)")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"保存: {save_path}")
    print(f"镜像: {', '.join(MIRRORS)}")
    print(f"线程: {num_threads}")
    print("=" * 60)

    # 执行下载
    if download_file(args.url, save_path, num_threads):
        print("\n下载成功!")
        sys.exit(0)
    else:
        print("\n下载失败!")
        sys.exit(1)


if __name__ == "__main__":
    main()
