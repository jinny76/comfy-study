# ComfyUI 源码解析 | 从零开始理解 AI 绘画引擎

> 最全面的 ComfyUI 中文源码分析，深入讲解节点系统、执行引擎、模型管理等核心模块。适合想要深入理解 Stable Diffusion 工作流原理的开发者。

[![GitHub stars](https://img.shields.io/github/stars/jinny76/comfy-study?style=social)](https://github.com/jinny76/comfy-study)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 这个项目是什么？

这是一份 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 源码学习笔记，逐行解析核心模块的实现原理：

- 节点是如何定义和注册的？
- 工作流是如何被执行的？
- 模型是如何加载和管理显存的？
- LoRA、ControlNet 是如何注入的？

如果你想：
- **开发自定义节点** - 了解节点 API 规范
- **优化工作流性能** - 理解缓存和执行机制
- **排查疑难问题** - 深入源码找答案
- **学习 AI 工程实践** - 参考优秀的代码架构

那这份笔记适合你。

## 源码解析笔记

| # | 模块 | 源文件 | 内容 |
|---|------|--------|------|
| 00 | [Desktop 代理设置](study/00-ComfyUI-Desktop代理设置.md) | - | 配置网络代理下载模型 |
| 01 | [启动流程](study/01-启动流程-main.py.md) | `main.py` | 程序入口、参数解析、服务启动 |
| 02 | [服务器架构](study/02-服务器架构-server.py.md) | `server.py` | aiohttp 服务、WebSocket、REST API |
| 03 | [节点系统](study/03-节点系统-nodes.py.md) | `nodes.py` | 节点定义、INPUT_TYPES、类型验证 |
| 04 | [执行引擎](study/04-执行引擎-execution.py.md) | `execution.py` | DAG 拓扑排序、缓存机制、PromptExecutor |
| 05 | [模型管理](study/05-模型管理-model_management.py.md) | `model_management.py` | 显存管理、模型加载/卸载策略 |
| 06 | [采样器系统](study/06-采样器系统-samplers.py.md) | `samplers.py` | 采样算法、调度器、CFG 引导 |
| 07 | [模型加载](study/07-模型加载-sd.py.md) | `sd.py` | Checkpoint 加载、CLIP、VAE、模型检测 |
| 08 | [模型修补器](study/08-模型修补-model_patcher.py.md) | `model_patcher.py` | LoRA 注入、动态权重修改 |
| 09 | [ControlNet](study/09-ControlNet-controlnet.py.md) | `controlnet.py` | ControlNet、T2I-Adapter 实现 |
| 10 | [Hook 系统](study/10-Hook系统-hooks.py.md) | `hooks.py` | 动态权重切换、关键帧调度 |
| 11 | [Latent 格式](study/11-Latent格式-latent_formats.py.md) | `latent_formats.py` | 潜空间规格、缩放因子、RGB 预览 |
| 12 | [条件系统](study/12-条件系统-conds.py.md) | `conds.py` | 条件封装、批处理、区域蒙版 |
| 13 | [操作封装](study/13-操作封装-ops.py.md) | `ops.py` | 权重转换、FP8 算子、LoRA 注入点 |
| 14 | [插件系统](study/14-插件系统-custom_nodes.md) | `nodes.py` | 自定义节点加载、V1/V3 API |
| 15 | [前端架构](study/15-前端架构-frontend.md) | `web/` | Vue 3 + LiteGraph.js、服务层、状态管理 |
| 16 | [文本编码器](study/16-文本编码器-clip.md) | `clip_model.py` | CLIP、T5、Tokenizer、权重语法、Embedding |
| 17 | [模型检测](study/17-模型检测-model_detection.md) | `model_detection.py` | 自动识别 60+ 种模型、UNet 配置推断 |

## 用户指南

面向普通用户的使用教程：

| 指南 | 内容 |
|------|------|
| [安装配置](docs/01-安装配置.md) | 安装步骤、环境配置、代理设置、常见问题 |
| [工作流入门](docs/02-工作流入门.md) | 节点系统、基础工作流、模板使用、资源网站 |
| [模型指南](docs/03-模型指南.md) | 模型下载、存放位置、LoRA、ControlNet |
| [性能优化](docs/04-性能优化.md) | 显存管理、速度优化、批量处理 |
| [自定义节点](docs/05-自定义节点.md) | 节点安装、常用节点推荐 |

## 工作流文档

详细的工作流节点参数说明：

| 工作流 | 说明 |
|--------|------|
| [Z-Image-Turbo](workflow-docs/z-image-turbo.md) | 高效文生图，支持中英文，每个节点参数详解 |
| [Hunyuan3D v2.1](workflow-docs/hunyuan3d-v2.1.md) | 图片转 3D 模型，完整流程说明 |

## 实用工具

### 模型下载器

```bash
# 多线程下载 + 自动 SHA256 校验
python tools/download_models.py --verify <huggingface_url>

# 只从 HuggingFace 原始源下载（不走镜像）
python tools/download_models.py --no-mirror --verify <url>

# 断点续传
python tools/download_models.py -r
```

特性：
- FlashGet 风格多线程下载（8 线程，4MB 分块）
- 多镜像自动测速（hf-mirror.com、aifasthub.com）
- SHA256 哈希校验（`--verify` 自动获取，`--sha256` 手动指定）
- 断点续传，Ctrl+C 安全中断

### 模型校验器

```bash
# 验证单个模型
python tools/verify_models.py model.safetensors

# 扫描整个目录
python tools/verify_models.py Z:\models --all
```

## 架构概览

```
ComfyUI 架构
├── main.py              # 入口：参数解析、服务启动
├── server.py            # 服务：aiohttp、WebSocket、REST API
├── nodes.py             # 节点：70+ 内置节点定义
├── execution.py         # 执行：DAG 引擎、智能缓存
├── comfy/
│   ├── model_management.py  # 显存/内存管理
│   ├── samplers.py          # 采样算法
│   ├── sd.py                # 模型加载（CLIP、VAE、UNet）
│   ├── model_detection.py   # 模型类型检测
│   ├── model_patcher.py     # LoRA、动态补丁
│   └── ldm/                 # 扩散模型实现
├── custom_nodes/            # 第三方插件
└── web/                     # 前端（LiteGraph.js）
```

## 支持的模型

| 类型 | 模型 |
|------|------|
| 图像 | SD 1.5、SD 2.x、SDXL、SD3、Flux、Z-Image、PixArt、AuraFlow |
| 视频 | Mochi、LTXV、HunyuanVideo、Wan、Cosmos |
| 音频 | Stable Audio、ACE Step |
| 3D | Hunyuan3D |

## 环境要求

- ComfyUI 0.3.x
- Python 3.10+
- PyTorch 2.x
- NVIDIA GPU（推荐 8GB+ 显存）

## 相关链接

- [ComfyUI 官方仓库](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI 官方文档](https://docs.comfy.org/)
- [Stable Diffusion 论文](https://arxiv.org/abs/2112.10752)

## 贡献

欢迎提 Issue 和 PR！如果这个项目对你有帮助，请点个 Star 支持一下。

## 许可证

MIT License

---

**关键词**: ComfyUI 教程, ComfyUI 源码, Stable Diffusion 工作流, AI 绘画, 文生图, SDXL, Flux, 自定义节点开发
