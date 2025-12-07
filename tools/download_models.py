#!/usr/bin/env python3
"""
ComfyUI 模型下载脚本 - 多镜像加速版

使用方法:
    # 下载单个文件
    python download_models.py <url> [保存路径]

    # 断点续传上次未完成的下载
    python download_models.py -r

    # 示例
    python download_models.py https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
    python download_models.py https://huggingface.co/xxx/model.safetensors Z:/models/checkpoints/model.safetensors

功能:
    1. 多镜像并行分片下载（加速）
    2. 自动测速选择最快镜像
    3. 支持断点续传（进度保存到 .progress 文件）
    4. 检查文件大小，已存在且大小正确则跳过
"""

import os
import sys
import json
import signal
import argparse
import requests
import threading
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional

# 全局中断标志
interrupted = False

def signal_handler(signum, frame):
    """处理 Ctrl+C 信号"""
    global interrupted
    interrupted = True
    print("\n\n  [中断] 收到中断信号，正在保存进度...")


# 注册信号处理器
signal.signal(signal.SIGINT, signal_handler)
if hasattr(signal, 'SIGTERM'):
    signal.signal(signal.SIGTERM, signal_handler)

# 默认保存目录
MODELS_DIR = Path("Z:/models")

# 进度文件目录
PROGRESS_DIR = Path("~/.comfy_download").expanduser()

# 多镜像源（会自动测速选择最快的）
# 参考: https://hf-mirror.com/, https://aifasthub.com/
MIRRORS = [
    "hf-mirror.com",       # 国内镜像（推荐）
    "aifasthub.com",       # AI快站镜像
    "huggingface.co",      # 原始源（需要梯子）
]

# 下载配置
TIMEOUT = (10, 30)  # (连接超时, 读取超时) - 缩短读取超时
CHUNK_SIZE = 1024 * 1024  # 1MB
NUM_THREADS = 4  # 每个文件的并行下载线程数
MAX_RETRIES = 10  # 增加重试次数，因为会更频繁地重连
MIN_SPEED = 50 * 1024  # 最低速度 50KB/s，低于此值重连
SPEED_CHECK_INTERVAL = 5  # 每5秒检查一次速度


@dataclass
class DownloadProgress:
    """下载进度信息"""
    url: str
    save_path: str
    file_size: int
    downloaded_size: int
    chunk_states: dict  # {chunk_index: downloaded_bytes}
    mirrors: list
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'DownloadProgress':
        data = json.loads(json_str)
        return cls(**data)


def get_progress_file(url: str, save_path: Path) -> Path:
    """获取进度文件路径"""
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    # 使用 URL 和保存路径的哈希作为文件名
    key = f"{url}:{save_path}"
    hash_name = hashlib.md5(key.encode()).hexdigest()[:16]
    return PROGRESS_DIR / f"{hash_name}.progress"


def load_progress(url: str, save_path: Path) -> Optional[DownloadProgress]:
    """加载下载进度"""
    progress_file = get_progress_file(url, save_path)
    if progress_file.exists():
        try:
            content = progress_file.read_text(encoding='utf-8')
            progress = DownloadProgress.from_json(content)
            # 验证 URL 和保存路径匹配
            if progress.url == url and progress.save_path == str(save_path):
                return progress
        except Exception as e:
            print(f"  [警告] 无法加载进度文件: {e}")
    return None


def save_progress(progress: DownloadProgress):
    """保存下载进度"""
    progress_file = get_progress_file(progress.url, Path(progress.save_path))
    progress.timestamp = time.time()
    progress_file.write_text(progress.to_json(), encoding='utf-8')


def delete_progress(url: str, save_path: Path):
    """删除进度文件"""
    progress_file = get_progress_file(url, save_path)
    if progress_file.exists():
        progress_file.unlink()


def list_pending_downloads() -> list[DownloadProgress]:
    """列出所有未完成的下载"""
    pending = []
    if not PROGRESS_DIR.exists():
        return pending

    for progress_file in PROGRESS_DIR.glob("*.progress"):
        try:
            content = progress_file.read_text(encoding='utf-8')
            progress = DownloadProgress.from_json(content)
            pending.append(progress)
        except Exception:
            continue

    # 按时间戳排序，最近的在前
    pending.sort(key=lambda x: x.timestamp, reverse=True)
    return pending


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
    """获取文件大小和是否支持分片下载，从多个镜像验证尺寸一致性"""
    sizes = []
    accept_ranges = False

    for mirror in mirrors:
        test_url = url.replace("huggingface.co", mirror)
        try:
            response = requests.head(test_url, timeout=TIMEOUT[0], allow_redirects=True)
            if response.status_code == 200:
                size = int(response.headers.get('content-length', 0))
                if size > 0:
                    sizes.append((mirror, size))
                    if response.headers.get('accept-ranges', '').lower() == 'bytes':
                        accept_ranges = True
        except:
            continue

    if not sizes:
        return (0, False)

    # 验证所有镜像返回相同的文件大小
    first_size = sizes[0][1]
    for mirror, size in sizes[1:]:
        if size != first_size:
            print(f"  [警告] 镜像文件大小不一致: {sizes[0][0]}={first_size}, {mirror}={size}")
            # 使用原始源的大小（如果有）
            for m, s in sizes:
                if 'huggingface.co' in m:
                    return (s, accept_ranges)
            # 否则使用第一个
            return (first_size, accept_ranges)

    return (first_size, accept_ranges)


def check_file_complete(save_path: Path, expected_size: int) -> bool:
    """检查本地文件是否完整"""
    if not save_path.exists():
        return False
    return save_path.stat().st_size == expected_size


class SlowSpeedError(Exception):
    """速度过慢异常，用于触发重连"""
    pass


def download_chunk(url: str, start: int, end: int, mirrors: list[str],
                   chunk_index: int, temp_dir: Path, progress: DownloadProgress,
                   pbar: tqdm, lock: threading.Lock) -> bool:
    """下载单个分片到临时文件，带速度监测和自动重连"""
    global interrupted
    chunk_file = temp_dir / f"chunk_{chunk_index:04d}"

    # 检查已下载的部分
    chunk_downloaded = progress.chunk_states.get(str(chunk_index), 0)
    chunk_size = end - start + 1

    if chunk_downloaded >= chunk_size:
        # 分片已完成
        return True

    mirror_index = chunk_index % len(mirrors)  # 初始镜像，每个线程用不同的

    for retry in range(MAX_RETRIES):
        if interrupted:
            return False

        # 选择镜像：每次重试切换到下一个镜像
        mirror = mirrors[(mirror_index + retry) % len(mirrors)]
        download_url = url.replace("huggingface.co", mirror)

        # 更新 Range header
        current_start = start + chunk_downloaded
        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Range': f'bytes={current_start}-{end}'
        }

        try:
            response = requests.get(
                download_url,
                headers=headers,
                timeout=TIMEOUT,
                stream=True
            )

            if response.status_code not in (200, 206):
                continue

            mode = 'ab' if chunk_downloaded > 0 else 'wb'

            # 速度监测变量
            speed_check_start = time.time()
            speed_check_bytes = 0

            with open(chunk_file, mode) as f:
                for data in response.iter_content(chunk_size=CHUNK_SIZE):
                    if interrupted:
                        # 保存当前进度
                        with lock:
                            progress.chunk_states[str(chunk_index)] = chunk_downloaded
                            progress.downloaded_size = sum(progress.chunk_states.values())
                            save_progress(progress)
                        return False

                    if data:
                        f.write(data)
                        data_len = len(data)
                        chunk_downloaded += data_len
                        speed_check_bytes += data_len

                        with lock:
                            pbar.update(data_len)
                            progress.chunk_states[str(chunk_index)] = chunk_downloaded
                            progress.downloaded_size = sum(progress.chunk_states.values())

                        # 每隔一段时间检查速度
                        elapsed = time.time() - speed_check_start
                        if elapsed >= SPEED_CHECK_INTERVAL:
                            speed = speed_check_bytes / elapsed
                            if speed < MIN_SPEED:
                                # 速度太慢，抛异常触发重连
                                raise SlowSpeedError(
                                    f"速度过慢: {speed/1024:.1f} KB/s < {MIN_SPEED/1024:.0f} KB/s, 切换镜像"
                                )
                            # 重置计数器
                            speed_check_start = time.time()
                            speed_check_bytes = 0

            # 下载完成，保存进度
            with lock:
                save_progress(progress)

            return True

        except SlowSpeedError as e:
            # 速度过慢，切换镜像重试
            with lock:
                # 不用打印太多，进度条已经能看出来卡住了
                pass
            continue

        except Exception as e:
            # 其他错误，短暂等待后重试
            if retry < MAX_RETRIES - 1:
                time.sleep(0.5)
            continue

    return False


def download_file_multithread(url: str, save_path: Path, mirrors: list[str],
                               file_size: int, num_threads: int = NUM_THREADS,
                               existing_progress: Optional[DownloadProgress] = None) -> bool:
    """多线程分片下载，支持断点续传"""
    global interrupted

    # 确保目录存在
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建临时目录
    temp_dir = save_path.parent / f".{save_path.name}.parts"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 计算分片
    part_size = file_size // num_threads
    chunks = []
    for i in range(num_threads):
        start = i * part_size
        end = file_size - 1 if i == num_threads - 1 else (i + 1) * part_size - 1
        chunks.append((i, start, end))

    # 初始化或恢复进度
    if existing_progress and existing_progress.file_size == file_size:
        progress = existing_progress
        # 从临时文件实际大小重新计算已下载量（更可靠）
        actual_downloaded = 0
        for i in range(num_threads):
            chunk_file = temp_dir / f"chunk_{i:04d}"
            if chunk_file.exists():
                chunk_size = chunk_file.stat().st_size
                progress.chunk_states[str(i)] = chunk_size
                actual_downloaded += chunk_size
        progress.downloaded_size = actual_downloaded
        initial_downloaded = actual_downloaded
        if initial_downloaded > 0:
            print(f"  [续传] 已下载 {initial_downloaded / 1024 / 1024:.1f} MB / {file_size / 1024 / 1024:.1f} MB ({initial_downloaded * 100 / file_size:.1f}%)")
    else:
        progress = DownloadProgress(
            url=url,
            save_path=str(save_path),
            file_size=file_size,
            downloaded_size=0,
            chunk_states={},
            mirrors=mirrors,
            timestamp=time.time()
        )
        initial_downloaded = 0

    lock = threading.Lock()

    # 显示线程和镜像分配
    print(f"  [下载] {num_threads} 线程并行下载 (速度 < {MIN_SPEED//1024}KB/s 自动切换镜像)")
    for i in range(min(num_threads, len(mirrors))):
        print(f"    线程 {i}: {mirrors[i % len(mirrors)]}")

    with tqdm(total=file_size, initial=initial_downloaded, unit='B', unit_scale=True,
              desc=save_path.name[:30], ncols=80) as pbar:

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for chunk_index, start, end in chunks:
                future = executor.submit(
                    download_chunk, url, start, end,
                    mirrors, chunk_index, temp_dir, progress, pbar, lock
                )
                futures.append(future)

            # 等待所有完成
            results = [f.result() for f in futures]
            success = all(results)

    if success and not interrupted:
        # 验证所有分片文件
        all_chunks_exist = True
        for i in range(num_threads):
            chunk_file = temp_dir / f"chunk_{i:04d}"
            if not chunk_file.exists():
                all_chunks_exist = False
                print(f"  [错误] 分片 {i} 不存在")
                break

        if all_chunks_exist:
            # 合并分片
            print("  [合并] 合并分片文件...")
            with open(save_path, 'wb') as outfile:
                for i in range(num_threads):
                    chunk_file = temp_dir / f"chunk_{i:04d}"
                    with open(chunk_file, 'rb') as infile:
                        outfile.write(infile.read())
                    chunk_file.unlink()

            # 验证最终文件大小
            final_size = save_path.stat().st_size
            if final_size == file_size:
                # 清理
                try:
                    temp_dir.rmdir()
                except:
                    pass
                delete_progress(url, save_path)
                print(f"  [完成] {save_path.name}")
                return True
            else:
                print(f"  [错误] 文件大小不匹配: {final_size} != {file_size}")
                save_path.unlink()
                return False
        else:
            save_progress(progress)
            return False
    else:
        # 保存进度以便下次续传
        save_progress(progress)
        print(f"  [中断] 进度已保存到 ~/.comfy_download/，下次运行将自动续传")
        return False


def download_file_simple(url: str, save_path: Path, mirrors: list[str],
                          existing_progress: Optional[DownloadProgress] = None) -> bool:
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
            delete_progress(url, save_path)
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

    # 检查是否有保存的进度
    existing_progress = load_progress(url, save_path)
    if existing_progress:
        print(f"  [恢复] 发现之前的下载进度")
        mirrors = existing_progress.mirrors
    else:
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
        delete_progress(url, save_path)
        return True

    # 大于 100MB 且支持分片，使用多线程
    if file_size > 100 * 1024 * 1024 and supports_range and len(mirrors) > 1:
        return download_file_multithread(url, save_path, mirrors, file_size, num_threads, existing_progress)
    else:
        return download_file_simple(url, save_path, mirrors, existing_progress)


def main():
    parser = argparse.ArgumentParser(
        description='ComfyUI 模型下载工具 - 多镜像加速版（支持断点续传）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
  %(prog)s https://huggingface.co/xxx/model.safetensors Z:/models/checkpoints/model.safetensors
  %(prog)s -t 8 https://huggingface.co/xxx/model.safetensors

断点续传:
  %(prog)s -r              # 续传最近一次未完成的下载
  %(prog)s -r --list       # 列出所有未完成的下载
  进度信息保存在 ~/.comfy_download/ 目录。
        '''
    )
    parser.add_argument('url', nargs='?', help='HuggingFace 模型下载 URL')
    parser.add_argument('save_path', nargs='?', help='保存路径（可选，默认根据 URL 自动推断）')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='续传上次未完成的下载')
    parser.add_argument('--list', action='store_true',
                        help='列出所有未完成的下载（配合 -r 使用）')
    parser.add_argument('-t', '--threads', type=int, default=4,
                        help='并行下载线程数（默认 4）')
    parser.add_argument('-o', '--output-dir', type=str, default="Z:/models",
                        help='模型保存根目录（默认 Z:/models）')

    args = parser.parse_args()

    # 处理 -r 参数：续传模式
    if args.resume or args.list:
        pending = list_pending_downloads()

        if not pending:
            print("没有未完成的下载任务。")
            sys.exit(0)

        if args.list:
            # 列出所有未完成的下载
            print("=" * 60)
            print("未完成的下载任务:")
            print("=" * 60)
            for i, p in enumerate(pending, 1):
                percent = p.downloaded_size * 100 / p.file_size if p.file_size > 0 else 0
                time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(p.timestamp))
                print(f"\n[{i}] {Path(p.save_path).name}")
                print(f"    进度: {p.downloaded_size / 1024 / 1024:.1f} MB / {p.file_size / 1024 / 1024:.1f} MB ({percent:.1f}%)")
                print(f"    路径: {p.save_path}")
                print(f"    时间: {time_str}")
            print("\n" + "=" * 60)
            print("使用 -r 续传最近的任务，或运行原始命令续传指定任务")
            sys.exit(0)

        # 续传最近的一个任务
        latest = pending[0]
        args.url = latest.url
        args.save_path = latest.save_path
        percent = latest.downloaded_size * 100 / latest.file_size if latest.file_size > 0 else 0
        print(f"续传任务: {Path(latest.save_path).name}")
        print(f"进度: {latest.downloaded_size / 1024 / 1024:.1f} MB / {latest.file_size / 1024 / 1024:.1f} MB ({percent:.1f}%)")

    # 检查是否提供了 URL
    if not args.url:
        parser.print_help()
        sys.exit(1)

    # 更新配置
    num_threads = args.threads
    models_dir = Path(args.output_dir)

    # 确定保存路径
    if args.save_path:
        save_path = Path(args.save_path)
    else:
        save_path = get_save_path_from_url(args.url, models_dir)

    print("=" * 60)
    print("ComfyUI 模型下载工具 (多镜像加速版 + 断点续传)")
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
        print("\n下载失败或中断!")
        sys.exit(1)


if __name__ == "__main__":
    main()
