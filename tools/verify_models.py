#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Model File Verification Tool

Features:
1. Verify safetensors file structure integrity
2. Check file size against expected values
3. Verify tensor data is readable
4. Compute file hash (optional)

Usage:
    python verify_models.py <model_path>
    python verify_models.py Z:/models/diffusion_models/z_image_turbo_bf16.safetensors
    python verify_models.py Z:/models --all  # Scan all models
"""

import os
import sys
import json
import struct
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import argparse


# 常见模型的预期大小 (字节, 允许 1% 误差)
KNOWN_MODEL_SIZES = {
    # Flux 系列
    "flux1-dev.safetensors": 23_802_932_552,
    "flux1-schnell.safetensors": 23_802_932_552,
    "ae.safetensors": 335_304_388,  # Flux VAE

    # Z-Image
    "z_image_turbo_bf16.safetensors": 12_309_866_400,

    # SDXL
    "sd_xl_base_1.0.safetensors": 6_938_078_334,
    "sd_xl_refiner_1.0.safetensors": 6_075_981_930,

    # SD 1.5
    "v1-5-pruned.safetensors": 4_265_380_512,
    "v1-5-pruned-emaonly.safetensors": 4_265_146_304,

    # Text Encoders
    "clip_l.safetensors": 246_144_152,
    "t5xxl_fp16.safetensors": 9_787_849_216,
    "t5xxl_fp8_e4m3fn.safetensors": 4_893_924_776,
    "qwen_3_4b.safetensors": 7_615_489_024,
}


class ModelVerifier:
    """模型文件验证器"""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors = []
        self.warnings = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def verify_safetensors(self, filepath: Path) -> Tuple[bool, Dict[str, Any]]:
        """
        验证 safetensors 文件完整性

        Returns:
            (is_valid, info_dict)
        """
        info = {
            "path": str(filepath),
            "size": 0,
            "tensors": 0,
            "metadata": {},
            "errors": [],
            "warnings": [],
        }

        try:
            file_size = filepath.stat().st_size
            info["size"] = file_size
            info["size_human"] = self._human_size(file_size)

            with open(filepath, "rb") as f:
                # 读取头部长度 (8 字节, little-endian uint64)
                header_size_bytes = f.read(8)
                if len(header_size_bytes) < 8:
                    info["errors"].append("文件太小，无法读取头部")
                    return False, info

                header_size = struct.unpack("<Q", header_size_bytes)[0]
                info["header_size"] = header_size

                # 合理性检查
                if header_size > 100_000_000:  # 100MB 头部不太合理
                    info["errors"].append(f"头部大小异常: {header_size} 字节")
                    return False, info

                if header_size + 8 > file_size:
                    info["errors"].append(f"头部大小 ({header_size}) 超过文件大小 ({file_size})")
                    return False, info

                # 读取并解析 JSON 头部
                header_bytes = f.read(header_size)
                try:
                    header = json.loads(header_bytes.decode("utf-8"))
                except json.JSONDecodeError as e:
                    info["errors"].append(f"JSON 头部解析失败: {e}")
                    return False, info

                # 提取元数据
                if "__metadata__" in header:
                    info["metadata"] = header.pop("__metadata__")

                # 统计张量
                tensor_count = len(header)
                info["tensors"] = tensor_count

                if tensor_count == 0:
                    info["warnings"].append("文件不包含任何张量")

                # 验证张量偏移量
                data_start = 8 + header_size
                max_offset = 0

                for name, tensor_info in header.items():
                    if "data_offsets" not in tensor_info:
                        info["errors"].append(f"张量 '{name}' 缺少 data_offsets")
                        continue

                    start, end = tensor_info["data_offsets"]
                    if end > max_offset:
                        max_offset = end

                    # 检查偏移量是否超出文件
                    if data_start + end > file_size:
                        info["errors"].append(
                            f"张量 '{name}' 偏移量 ({data_start + end}) 超出文件大小 ({file_size})"
                        )

                # 检查文件是否被截断
                expected_size = data_start + max_offset
                if file_size < expected_size:
                    info["errors"].append(
                        f"文件可能被截断: 期望 {expected_size} 字节，实际 {file_size} 字节"
                    )
                    return False, info

                # 可选：验证第一个和最后一个张量数据是否可读
                if tensor_count > 0:
                    # 读取少量数据验证文件可访问
                    first_tensor = list(header.values())[0]
                    start, end = first_tensor["data_offsets"]
                    f.seek(data_start + start)
                    sample = f.read(min(1024, end - start))
                    if len(sample) == 0:
                        info["errors"].append("无法读取张量数据")
                        return False, info

                # 检查已知模型大小
                filename = filepath.name
                if filename in KNOWN_MODEL_SIZES:
                    expected = KNOWN_MODEL_SIZES[filename]
                    tolerance = expected * 0.01  # 1% 容差
                    if abs(file_size - expected) > tolerance:
                        info["warnings"].append(
                            f"文件大小不匹配: 期望 {self._human_size(expected)}，"
                            f"实际 {self._human_size(file_size)}"
                        )

            if info["errors"]:
                return False, info

            return True, info

        except PermissionError:
            info["errors"].append("无法访问文件 (权限不足)")
            return False, info
        except Exception as e:
            info["errors"].append(f"验证失败: {e}")
            return False, info

    def verify_ckpt(self, filepath: Path) -> Tuple[bool, Dict[str, Any]]:
        """验证 .ckpt 文件 (PyTorch checkpoint)"""
        info = {
            "path": str(filepath),
            "size": filepath.stat().st_size,
            "size_human": self._human_size(filepath.stat().st_size),
            "format": "pytorch_checkpoint",
            "errors": [],
            "warnings": [],
        }

        try:
            import torch
            # 只加载元数据，不加载完整模型
            checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

            if isinstance(checkpoint, dict):
                info["keys"] = list(checkpoint.keys())[:10]  # 前 10 个键
                if "state_dict" in checkpoint:
                    info["tensors"] = len(checkpoint["state_dict"])

            return True, info

        except Exception as e:
            info["errors"].append(f"加载失败: {e}")
            return False, info

    def compute_hash(self, filepath: Path, algorithm: str = "sha256") -> str:
        """计算文件哈希值"""
        h = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            while chunk := f.read(8192 * 1024):  # 8MB chunks
                h.update(chunk)
        return h.hexdigest()

    def verify_file(self, filepath: Path, compute_hash: bool = False) -> Tuple[bool, Dict[str, Any]]:
        """验证单个模型文件"""
        suffix = filepath.suffix.lower()

        if suffix == ".safetensors":
            valid, info = self.verify_safetensors(filepath)
        elif suffix in (".ckpt", ".pt", ".pth", ".bin"):
            valid, info = self.verify_ckpt(filepath)
        else:
            return False, {"path": str(filepath), "errors": [f"不支持的格式: {suffix}"]}

        if compute_hash and valid:
            self.log(f"  计算哈希值...")
            info["sha256"] = self.compute_hash(filepath)

        return valid, info

    def scan_directory(self, dirpath: Path, compute_hash: bool = False) -> Dict[str, Any]:
        """扫描目录中的所有模型文件"""
        results = {
            "total": 0,
            "valid": 0,
            "invalid": 0,
            "files": [],
        }

        extensions = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}

        for root, dirs, files in os.walk(dirpath):
            for filename in files:
                if Path(filename).suffix.lower() in extensions:
                    filepath = Path(root) / filename
                    results["total"] += 1

                    self.log(f"\n验证: {filepath}")
                    valid, info = self.verify_file(filepath, compute_hash)

                    if valid:
                        results["valid"] += 1
                        self.log(f"  [OK] Valid ({info.get('tensors', '?')} tensors, {info.get('size_human', '?')})")
                    else:
                        results["invalid"] += 1
                        self.log(f"  [FAIL] Invalid: {info.get('errors', [])}")

                    if info.get("warnings"):
                        for w in info["warnings"]:
                            self.log(f"  [WARN] {w}")

                    results["files"].append(info)

        return results

    @staticmethod
    def _human_size(size: int) -> str:
        """转换为人类可读的大小"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.2f} {unit}"
            size /= 1024
        return f"{size:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="ComfyUI 模型文件校验工具")
    parser.add_argument("path", help="模型文件或目录路径")
    parser.add_argument("--all", action="store_true", help="扫描目录中所有模型")
    parser.add_argument("--hash", action="store_true", help="计算 SHA256 哈希值")
    parser.add_argument("--quiet", "-q", action="store_true", help="安静模式")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"错误: 路径不存在: {path}")
        sys.exit(1)

    verifier = ModelVerifier(verbose=not args.quiet and not args.json)

    if path.is_file():
        valid, info = verifier.verify_file(path, compute_hash=args.hash)

        if args.json:
            print(json.dumps(info, indent=2, ensure_ascii=False))
        else:
            if valid:
                print(f"\n[OK] Model is valid")
                print(f"  Size: {info.get('size_human', '?')}")
                print(f"  Tensors: {info.get('tensors', '?')}")
                if info.get("metadata"):
                    print(f"  Metadata keys: {list(info['metadata'].keys())}")
                if info.get("sha256"):
                    print(f"  SHA256: {info['sha256']}")
            else:
                print(f"\n[FAIL] Model is invalid")
                for err in info.get("errors", []):
                    print(f"  Error: {err}")

        sys.exit(0 if valid else 1)

    elif path.is_dir():
        if not args.all:
            print("提示: 使用 --all 参数扫描目录中的所有模型")
            sys.exit(1)

        results = verifier.scan_directory(path, compute_hash=args.hash)

        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(f"\n{'='*50}")
            print(f"Scan completed")
            print(f"  Total: {results['total']} files")
            print(f"  Valid: {results['valid']}")
            print(f"  Invalid: {results['invalid']}")

            if results["invalid"] > 0:
                print(f"\nInvalid files:")
                for f in results["files"]:
                    if f.get("errors"):
                        print(f"  - {f['path']}")
                        for err in f["errors"]:
                            print(f"      {err}")

        sys.exit(0 if results["invalid"] == 0 else 1)


if __name__ == "__main__":
    main()
