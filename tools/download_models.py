#!/usr/bin/env python3
"""
ComfyUI 模型下载脚本 - FlashGet 多线程模式

使用方法:
    # 下载单个文件
    python download_models.py <url> [保存路径]

    # 断点续传上次未完成的下载
    python download_models.py -r

    # 使用更多线程 (默认8线程)
    python download_models.py -t 16 <url>

    # 指定期望的 SHA256 哈希值进行验证
    python download_models.py --sha256 abc123... <url>

    # 自动从 HuggingFace 获取哈希值验证
    python download_models.py --verify <url>

    # 示例
    python download_models.py https://huggingface.co/xxx/model.safetensors

功能:
    1. FlashGet 风格多线程分块下载（同一镜像也多线程）
    2. 动态任务分配（快的线程多下载）
    3. 自动测速选择最快镜像
    4. 断点续传（进度保存到 ~/.comfy_download/）
    5. 慢速自动切换镜像
    6. SHA256 哈希验证（支持自动获取或手动指定）
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
from dataclasses import dataclass, asdict, field
from typing import Optional
from queue import Queue, Empty

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

# 进度文件目录 (也用于临时下载)
PROGRESS_DIR = Path("~/.comfy_download").expanduser()
TEMP_DOWNLOAD_DIR = PROGRESS_DIR / "downloading"  # 本地临时下载目录

# 多镜像源
MIRRORS = [
    "hf-mirror.com",       # 国内镜像
    "aifasthub.com",       # AI快站镜像
    "huggingface.co",      # 原始源
]

# 下载配置
TIMEOUT = (10, 30)
CHUNK_SIZE = 64 * 1024       # 64KB 读取块
BLOCK_SIZE = 4 * 1024 * 1024  # 4MB 每个下载块 (FlashGet 风格小块)
NUM_THREADS = 8              # 默认8线程 (FlashGet 风格)
MAX_RETRIES = 5
MIN_SPEED = 100 * 1024       # 最低速度 100KB/s
SPEED_CHECK_INTERVAL = 3     # 每3秒检查速度


@dataclass
class BlockInfo:
    """下载块信息"""
    index: int
    start: int
    end: int
    downloaded: int = 0
    status: str = 'pending'  # pending, downloading, completed, failed

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    @property
    def remaining(self) -> int:
        return self.size - self.downloaded

    @property
    def is_complete(self) -> bool:
        return self.downloaded >= self.size


@dataclass
class DownloadProgress:
    """下载进度信息"""
    url: str
    save_path: str
    file_size: int
    downloaded_size: int
    blocks: list  # List of BlockInfo as dicts
    mirrors: list
    timestamp: float
    block_size: int = BLOCK_SIZE

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'DownloadProgress':
        data = json.loads(json_str)
        return cls(**data)


def get_progress_file(url: str, save_path: Path) -> Path:
    """获取进度文件路径"""
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
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
        subdir = 'checkpoints'

    return models_dir / subdir / filename


def test_mirror_speed(url: str, mirror: str) -> tuple[str, float]:
    """测试镜像速度"""
    test_url = url.replace("huggingface.co", mirror)
    try:
        start = time.time()
        response = requests.get(
            test_url,
            headers={'Range': 'bytes=0-1048575'},
            timeout=TIMEOUT,
            stream=True
        )
        if response.status_code in (200, 206):
            data = response.content
            elapsed = time.time() - start
            speed = len(data) / elapsed / 1024 / 1024
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

    results.sort(key=lambda x: x[1], reverse=True)
    best = [r[0] for r in results[:count]]

    if not best:
        print("  [警告] 所有镜像不可用，使用原始源")
        return ["huggingface.co"]

    print(f"  [选择] 使用镜像: {', '.join(best)}")
    return best


def get_file_info(url: str, mirrors: list[str]) -> tuple[int, bool]:
    """获取文件大小和是否支持分片下载"""
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

    first_size = sizes[0][1]
    for mirror, size in sizes[1:]:
        if size != first_size:
            print(f"  [警告] 镜像文件大小不一致: {sizes[0][0]}={first_size}, {mirror}={size}")
            for m, s in sizes:
                if 'huggingface.co' in m:
                    return (s, accept_ranges)
            return (first_size, accept_ranges)

    return (first_size, accept_ranges)


def check_file_complete(save_path: Path, expected_size: int) -> bool:
    """检查本地文件是否完整"""
    if not save_path.exists():
        return False
    return save_path.stat().st_size == expected_size


def fetch_hf_sha256(url: str) -> Optional[str]:
    """从 HuggingFace 获取文件的 SHA256 哈希值"""
    # 将下载 URL 转换为 blob 页面 URL
    # https://huggingface.co/xxx/resolve/main/file.safetensors
    # -> https://huggingface.co/xxx/blob/main/file.safetensors
    if '/resolve/' not in url:
        return None

    blob_url = url.replace('/resolve/', '/blob/')
    # 确保使用原始 huggingface.co
    for mirror in MIRRORS:
        blob_url = blob_url.replace(mirror, 'huggingface.co')

    try:
        print(f"  [哈希] 从 HuggingFace 获取 SHA256...")
        response = requests.get(blob_url, timeout=TIMEOUT[0])
        if response.status_code == 200:
            # 在页面中搜索 SHA256 哈希值 (64位十六进制)
            import re
            # HuggingFace 页面中的哈希格式
            patterns = [
                r'"oid":"([a-f0-9]{64})"',  # JSON 格式
                r'sha256["\s:]+([a-f0-9]{64})',  # 通用格式
                r'>([a-f0-9]{64})<',  # HTML 标签内
            ]
            for pattern in patterns:
                match = re.search(pattern, response.text, re.IGNORECASE)
                if match:
                    sha256 = match.group(1).lower()
                    print(f"  [哈希] 获取到: {sha256[:16]}...")
                    return sha256
    except Exception as e:
        print(f"  [哈希] 获取失败: {e}")

    return None


def calculate_file_sha256(filepath: Path, show_progress: bool = True) -> str:
    """计算文件的 SHA256 哈希值"""
    sha256 = hashlib.sha256()
    file_size = filepath.stat().st_size

    if show_progress:
        with tqdm(total=file_size, unit='B', unit_scale=True,
                  desc="SHA256 校验", ncols=80) as pbar:
            with open(filepath, 'rb') as f:
                while True:
                    data = f.read(8 * 1024 * 1024)  # 8MB chunks
                    if not data:
                        break
                    sha256.update(data)
                    pbar.update(len(data))
    else:
        with open(filepath, 'rb') as f:
            while True:
                data = f.read(8 * 1024 * 1024)
                if not data:
                    break
                sha256.update(data)

    return sha256.hexdigest()


def verify_file_hash(filepath: Path, expected_hash: str) -> bool:
    """验证文件哈希值"""
    print(f"  [验证] 计算文件哈希...")
    actual_hash = calculate_file_sha256(filepath)

    if actual_hash.lower() == expected_hash.lower():
        print(f"  [验证] SHA256 校验通过!")
        return True
    else:
        print(f"  [验证] SHA256 校验失败!")
        print(f"    期望: {expected_hash}")
        print(f"    实际: {actual_hash}")
        return False


class SlowSpeedError(Exception):
    """速度过慢异常"""
    pass


class BlockDownloader:
    """FlashGet 风格的块下载管理器"""

    def __init__(self, url: str, save_path: Path, file_size: int,
                 mirrors: list[str], num_threads: int,
                 existing_progress: Optional[DownloadProgress] = None):
        self.url = url
        self.save_path = save_path
        self.file_size = file_size
        self.mirrors = mirrors
        self.num_threads = num_threads

        # 使用本地临时目录下载 (SSD 快)，完成后再复制到目标
        TEMP_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        # 用文件名的 hash 创建唯一目录，避免冲突
        file_hash = hashlib.md5(f"{url}:{save_path}".encode()).hexdigest()[:12]
        self.temp_dir = TEMP_DOWNLOAD_DIR / f"{save_path.stem}_{file_hash}"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # 初始化块列表
        self.blocks: list[BlockInfo] = []
        self.block_queue = Queue()
        self.lock = threading.Lock()
        self.downloaded_bytes = 0
        self.active_downloads = 0

        # 统计信息
        self.thread_stats = {}  # thread_id -> {'bytes': 0, 'blocks': 0}

        # 初始化或恢复块
        self._init_blocks(existing_progress)

    def _init_blocks(self, existing_progress: Optional[DownloadProgress]):
        """初始化下载块"""
        # 计算块数量
        num_blocks = (self.file_size + BLOCK_SIZE - 1) // BLOCK_SIZE

        if existing_progress and existing_progress.file_size == self.file_size:
            # 从进度恢复
            for block_data in existing_progress.blocks:
                block = BlockInfo(**block_data)
                # 验证临时文件实际大小
                block_file = self.temp_dir / f"block_{block.index:06d}"
                if block_file.exists():
                    actual_size = block_file.stat().st_size
                    block.downloaded = min(actual_size, block.size)
                    if block.is_complete:
                        block.status = 'completed'
                    else:
                        block.status = 'pending'
                else:
                    block.downloaded = 0
                    block.status = 'pending'
                self.blocks.append(block)
                self.downloaded_bytes += block.downloaded
        else:
            # 新建块
            for i in range(num_blocks):
                start = i * BLOCK_SIZE
                end = min((i + 1) * BLOCK_SIZE - 1, self.file_size - 1)
                block = BlockInfo(index=i, start=start, end=end)
                self.blocks.append(block)

        # 将未完成的块加入队列
        pending_blocks = [b for b in self.blocks if not b.is_complete]
        for block in pending_blocks:
            self.block_queue.put(block)

        if self.downloaded_bytes > 0:
            pct = self.downloaded_bytes * 100 / self.file_size
            print(f"  [续传] 已下载 {self.downloaded_bytes / 1024 / 1024:.1f} MB / "
                  f"{self.file_size / 1024 / 1024:.1f} MB ({pct:.1f}%)")
            print(f"  [续传] {len(pending_blocks)} / {len(self.blocks)} 块待下载")

    def _get_progress(self) -> DownloadProgress:
        """获取当前进度"""
        return DownloadProgress(
            url=self.url,
            save_path=str(self.save_path),
            file_size=self.file_size,
            downloaded_size=self.downloaded_bytes,
            blocks=[asdict(b) for b in self.blocks],
            mirrors=self.mirrors,
            timestamp=time.time()
        )

    def _download_block(self, thread_id: int, pbar: tqdm) -> bool:
        """工作线程：从队列获取块并下载"""
        global interrupted

        # 初始化线程统计
        with self.lock:
            self.thread_stats[thread_id] = {'bytes': 0, 'blocks': 0}

        mirror_index = thread_id % len(self.mirrors)

        while not interrupted:
            # 从队列获取块
            try:
                block = self.block_queue.get(timeout=0.5)
            except Empty:
                # 队列空了，检查是否还有下载中的
                with self.lock:
                    if self.active_downloads == 0:
                        return True
                continue

            if block.is_complete:
                self.block_queue.task_done()
                continue

            with self.lock:
                block.status = 'downloading'
                self.active_downloads += 1

            success = False
            for retry in range(MAX_RETRIES):
                if interrupted:
                    break

                mirror = self.mirrors[(mirror_index + retry) % len(self.mirrors)]
                download_url = self.url.replace("huggingface.co", mirror)

                current_start = block.start + block.downloaded
                headers = {
                    'User-Agent': 'Mozilla/5.0',
                    'Range': f'bytes={current_start}-{block.end}'
                }

                try:
                    response = requests.get(
                        download_url, headers=headers,
                        timeout=TIMEOUT, stream=True
                    )

                    if response.status_code not in (200, 206):
                        continue

                    block_file = self.temp_dir / f"block_{block.index:06d}"
                    mode = 'ab' if block.downloaded > 0 else 'wb'

                    speed_check_start = time.time()
                    speed_check_bytes = 0

                    with open(block_file, mode) as f:
                        for data in response.iter_content(chunk_size=CHUNK_SIZE):
                            if interrupted:
                                break

                            if data:
                                f.write(data)
                                data_len = len(data)
                                block.downloaded += data_len
                                speed_check_bytes += data_len

                                with self.lock:
                                    self.downloaded_bytes += data_len
                                    self.thread_stats[thread_id]['bytes'] += data_len
                                    pbar.update(data_len)

                                # 速度检查
                                elapsed = time.time() - speed_check_start
                                if elapsed >= SPEED_CHECK_INTERVAL:
                                    speed = speed_check_bytes / elapsed
                                    if speed < MIN_SPEED:
                                        raise SlowSpeedError(f"速度过慢: {speed/1024:.0f} KB/s")
                                    speed_check_start = time.time()
                                    speed_check_bytes = 0

                    if block.is_complete:
                        success = True
                        with self.lock:
                            block.status = 'completed'
                            self.thread_stats[thread_id]['blocks'] += 1
                        break

                except SlowSpeedError:
                    # 切换镜像重试
                    continue
                except Exception as e:
                    if retry < MAX_RETRIES - 1:
                        time.sleep(0.5)
                    continue

            with self.lock:
                self.active_downloads -= 1
                if not success and not interrupted:
                    block.status = 'pending'
                    # 重新放回队列
                    self.block_queue.put(block)

            self.block_queue.task_done()

        return False

    def download(self) -> bool:
        """执行下载"""
        global interrupted

        num_blocks = len(self.blocks)
        pending = sum(1 for b in self.blocks if not b.is_complete)

        print(f"  [下载] FlashGet 模式: {self.num_threads} 线程, {num_blocks} 块 (每块 {BLOCK_SIZE // 1024 // 1024}MB)")
        print(f"  [下载] 临时目录: {self.temp_dir}")
        print(f"  [下载] 速度 < {MIN_SPEED // 1024}KB/s 自动切换镜像")

        with tqdm(total=self.file_size, initial=self.downloaded_bytes,
                  unit='B', unit_scale=True, desc=self.save_path.name[:30],
                  ncols=80) as pbar:

            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                futures = []
                for i in range(self.num_threads):
                    future = executor.submit(self._download_block, i, pbar)
                    futures.append(future)

                # 定期保存进度
                while not all(f.done() for f in futures):
                    time.sleep(2)
                    with self.lock:
                        save_progress(self._get_progress())

                results = [f.result() for f in futures]

        # 保存最终进度
        save_progress(self._get_progress())

        if interrupted:
            print(f"  [中断] 进度已保存，下次运行 -r 续传")
            self._print_stats()
            return False

        # 检查是否全部完成
        incomplete = [b for b in self.blocks if not b.is_complete]
        if incomplete:
            print(f"  [错误] {len(incomplete)} 块未完成")
            return False

        # 合并文件
        return self._merge_blocks()

    def _merge_blocks(self) -> bool:
        """合并所有块到本地临时文件，然后复制到目标目录"""
        import shutil

        # 本地临时合并文件
        local_merged = self.temp_dir / self.save_path.name
        print(f"  [合并] 合并分块到本地临时文件...")

        try:
            # 第一步：合并到本地
            with open(local_merged, 'wb') as outfile:
                for block in sorted(self.blocks, key=lambda b: b.index):
                    block_file = self.temp_dir / f"block_{block.index:06d}"
                    with open(block_file, 'rb') as infile:
                        outfile.write(infile.read())
                    block_file.unlink()

            # 验证本地文件大小
            local_size = local_merged.stat().st_size
            if local_size != self.file_size:
                print(f"  [错误] 文件大小不匹配: {local_size} != {self.file_size}")
                local_merged.unlink()
                return False

            # 第二步：复制到目标目录
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [复制] 复制到目标: {self.save_path}")

            # 使用 shutil.copy2 保留元数据，带进度显示
            with tqdm(total=self.file_size, unit='B', unit_scale=True,
                      desc="复制中", ncols=80) as pbar:
                with open(local_merged, 'rb') as src:
                    with open(self.save_path, 'wb') as dst:
                        while True:
                            buf = src.read(1024 * 1024)  # 1MB chunks
                            if not buf:
                                break
                            dst.write(buf)
                            pbar.update(len(buf))

            # 验证目标文件大小
            final_size = self.save_path.stat().st_size
            if final_size != self.file_size:
                print(f"  [错误] 目标文件大小不匹配: {final_size} != {self.file_size}")
                self.save_path.unlink()
                return False

            # 清理本地临时文件
            local_merged.unlink()
            try:
                self.temp_dir.rmdir()
            except:
                pass

            delete_progress(self.url, self.save_path)
            print(f"  [完成] {self.save_path.name}")
            self._print_stats()
            return True

        except Exception as e:
            print(f"  [错误] 合并/复制失败: {e}")
            # 保留本地已合并文件以便恢复
            if local_merged.exists():
                print(f"  [提示] 本地已合并文件保留在: {local_merged}")
            return False

    def _print_stats(self):
        """打印线程统计"""
        print("\n  [统计] 各线程下载量:")
        total_bytes = 0
        for tid, stats in sorted(self.thread_stats.items()):
            mb = stats['bytes'] / 1024 / 1024
            total_bytes += stats['bytes']
            print(f"    线程 {tid}: {mb:.1f} MB ({stats['blocks']} 块)")


def download_file_simple(url: str, save_path: Path, mirrors: list[str]) -> bool:
    """简单单线程下载（用于小文件）"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = save_path.with_suffix(save_path.suffix + '.downloading')

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
            response = requests.get(download_url, headers=headers,
                                    stream=True, timeout=TIMEOUT)

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


def download_file(url: str, save_path: Path, num_threads: int = NUM_THREADS,
                  expected_sha256: Optional[str] = None, auto_verify: bool = False,
                  no_mirror: bool = False) -> bool:
    """智能下载入口

    Args:
        url: 下载 URL
        save_path: 保存路径
        num_threads: 线程数
        expected_sha256: 期望的 SHA256 哈希值（手动指定）
        auto_verify: 是否自动从 HuggingFace 获取哈希值验证
        no_mirror: 不使用镜像，只从原始源下载
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取期望的哈希值
    sha256_to_verify = expected_sha256
    if auto_verify and not sha256_to_verify:
        sha256_to_verify = fetch_hf_sha256(url)

    # 检查是否有保存的进度
    existing_progress = load_progress(url, save_path)
    if existing_progress:
        print(f"  [恢复] 发现之前的下载进度")
        mirrors = existing_progress.mirrors
    elif no_mirror:
        print(f"  [直连] 只使用 huggingface.co 原始源")
        mirrors = ["huggingface.co"]
    else:
        mirrors = select_best_mirrors(url, count=min(3, len(MIRRORS)))

    # 获取文件信息
    file_size, supports_range = get_file_info(url, mirrors)

    if file_size == 0:
        print("  [错误] 无法获取文件大小")
        return False

    print(f"  [信息] 文件大小: {file_size / 1024 / 1024:.1f} MB, 支持分片: {supports_range}")

    # 检查是否已存在
    if check_file_complete(save_path, file_size):
        print(f"  [跳过] 文件已存在且大小匹配")
        # 即使文件存在，如果指定了哈希也要验证
        if sha256_to_verify:
            if not verify_file_hash(save_path, sha256_to_verify):
                print(f"  [警告] 现有文件哈希不匹配，建议重新下载")
                return False
        delete_progress(url, save_path)
        return True

    # 大于 50MB 且支持分片，使用 FlashGet 模式
    if file_size > 50 * 1024 * 1024 and supports_range:
        downloader = BlockDownloader(
            url, save_path, file_size, mirrors, num_threads, existing_progress
        )
        success = downloader.download()
    else:
        success = download_file_simple(url, save_path, mirrors)

    # 下载完成后验证哈希
    if success and sha256_to_verify:
        if not verify_file_hash(save_path, sha256_to_verify):
            print(f"  [错误] 下载的文件哈希不匹配，文件可能损坏")
            # 可选：删除损坏的文件
            # save_path.unlink()
            return False

    return success


def main():
    parser = argparse.ArgumentParser(
        description='ComfyUI 模型下载工具 - FlashGet 多线程模式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  %(prog)s https://huggingface.co/xxx/model.safetensors
  %(prog)s -t 16 https://huggingface.co/xxx/model.safetensors
  %(prog)s -r              # 续传上次未完成的下载
  %(prog)s -r --list       # 列出所有未完成的下载
        '''
    )
    parser.add_argument('url', nargs='?', help='HuggingFace 模型下载 URL')
    parser.add_argument('save_path', nargs='?', help='保存路径（可选）')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='续传上次未完成的下载')
    parser.add_argument('--list', action='store_true',
                        help='列出所有未完成的下载')
    parser.add_argument('-t', '--threads', type=int, default=NUM_THREADS,
                        help=f'并行下载线程数（默认 {NUM_THREADS}）')
    parser.add_argument('-o', '--output-dir', type=str, default="Z:/models",
                        help='模型保存根目录（默认 Z:/models）')
    parser.add_argument('--sha256', type=str, default=None,
                        help='期望的 SHA256 哈希值（用于验证）')
    parser.add_argument('--verify', action='store_true',
                        help='自动从 HuggingFace 获取并验证 SHA256')
    parser.add_argument('--no-mirror', action='store_true',
                        help='不使用镜像，只从 huggingface.co 原始源下载')

    args = parser.parse_args()

    # 处理 -r 参数
    if args.resume or args.list:
        pending = list_pending_downloads()

        if not pending:
            print("没有未完成的下载任务。")
            sys.exit(0)

        if args.list:
            print("=" * 60)
            print("未完成的下载任务:")
            print("=" * 60)
            for i, p in enumerate(pending, 1):
                percent = p.downloaded_size * 100 / p.file_size if p.file_size > 0 else 0
                time_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(p.timestamp))
                print(f"\n[{i}] {Path(p.save_path).name}")
                print(f"    进度: {p.downloaded_size / 1024 / 1024:.1f} MB / "
                      f"{p.file_size / 1024 / 1024:.1f} MB ({percent:.1f}%)")
                print(f"    路径: {p.save_path}")
                print(f"    时间: {time_str}")
            print("\n" + "=" * 60)
            sys.exit(0)

        latest = pending[0]
        args.url = latest.url
        args.save_path = latest.save_path
        percent = latest.downloaded_size * 100 / latest.file_size if latest.file_size > 0 else 0
        print(f"续传任务: {Path(latest.save_path).name}")
        print(f"进度: {latest.downloaded_size / 1024 / 1024:.1f} MB / "
              f"{latest.file_size / 1024 / 1024:.1f} MB ({percent:.1f}%)")

    if not args.url:
        parser.print_help()
        sys.exit(1)

    num_threads = args.threads
    models_dir = Path(args.output_dir)

    if args.save_path:
        save_path = Path(args.save_path)
    else:
        save_path = get_save_path_from_url(args.url, models_dir)

    print("=" * 60)
    print("ComfyUI 模型下载工具 (FlashGet 多线程模式)")
    print("=" * 60)
    print(f"URL: {args.url}")
    print(f"保存: {save_path}")
    print(f"线程: {num_threads}")
    print("=" * 60)

    if download_file(args.url, save_path, num_threads,
                     expected_sha256=args.sha256, auto_verify=args.verify,
                     no_mirror=args.no_mirror):
        print("\n下载成功!")
        sys.exit(0)
    else:
        print("\n下载失败或中断!")
        sys.exit(1)


if __name__ == "__main__":
    main()
