# ComfyUI Source Code Study Notes

A deep dive into [ComfyUI](https://github.com/comfyanonymous/ComfyUI) source code, documenting the architecture, design patterns, and implementation details of this powerful Stable Diffusion GUI.

## About

This repository contains my study notes while learning the ComfyUI codebase. Each document focuses on a specific module, explaining:

- Core architecture and design decisions
- Key classes and functions
- Data flow and execution pipelines
- Code examples with annotations

## Study Notes

| # | Module | File | Description |
|---|--------|------|-------------|
| 00 | [Desktop Proxy Settings](study/00-ComfyUI-Desktop代理设置.md) | - | ComfyUI Desktop proxy configuration |
| 01 | [Startup Flow](study/01-启动流程-main.py.md) | `main.py` | Application entry point and initialization |
| 02 | [Server Architecture](study/02-服务器架构-server.py.md) | `server.py` | aiohttp server, WebSocket, REST API |
| 03 | [Node System](study/03-节点系统-nodes.py.md) | `nodes.py` | Node definitions, INPUT_TYPES, validation |
| 04 | [Execution Engine](study/04-执行引擎-execution.py.md) | `execution.py` | DAG execution, caching, PromptExecutor |
| 05 | [Model Management](study/05-模型管理-model_management.py.md) | `model_management.py` | VRAM management, model loading/unloading |
| 06 | [Sampler System](study/06-采样器系统-samplers.py.md) | `samplers.py` | Sampling algorithms, schedulers, CFG |
| 07 | [Model Loading](study/07-模型加载-sd.py.md) | `sd.py` | Checkpoint loading, CLIP, VAE, model detection |
| 08 | [Model Patcher](study/08-模型修补-model_patcher.py.md) | `model_patcher.py` | LoRA, hooks, dynamic weight patching |
| 09 | [ControlNet](study/09-ControlNet-controlnet.py.md) | `controlnet.py` | ControlNet, T2I-Adapter, control signals |
| 10 | [Hook System](study/10-Hook系统-hooks.py.md) | `hooks.py` | Dynamic weight switching, keyframe scheduling |
| 11 | [Latent Formats](study/11-Latent格式-latent_formats.py.md) | `latent_formats.py` | Latent space specs, scale factors, RGB preview |
| 12 | [Conditioning](study/12-条件系统-conds.py.md) | `conds.py` | Condition wrappers, batching, area masks |
| 13 | [Operations](study/13-操作封装-ops.py.md) | `ops.py` | Weight casting, FP8 ops, LoRA injection |

## User Guides

📚 **[docs/](docs/)** - User-friendly guides for ComfyUI beginners

| Guide | Description |
|-------|-------------|
| [安装配置](docs/01-安装配置.md) | Installation, environment setup, proxy configuration |
| [工作流入门](docs/02-工作流入门.md) | Node system basics, workflows, tips |
| [模型指南](docs/03-模型指南.md) | Model downloads, types, LoRA, ControlNet |
| [性能优化](docs/04-性能优化.md) | VRAM management, speed optimization |

## Tools

- [`tools/download_models.py`](tools/download_models.py) - Multi-mirror accelerated model downloader
  - Multi-mirror parallel download (hf-mirror.com, aifasthub.com)
  - **Robust resume support** - progress saved to `~/.comfy_download/`
  - Ctrl+C safe - gracefully saves progress on interrupt
  - File size verification across mirrors
  - Auto speed test to select fastest mirror

## Architecture Overview

```
ComfyUI Architecture
├── main.py              # Entry point, CLI args, server startup
├── server.py            # aiohttp server, WebSocket, REST API
├── nodes.py             # 50+ built-in node definitions
├── execution.py         # DAG execution engine, caching
├── comfy/
│   ├── model_management.py  # VRAM/RAM management
│   ├── samplers.py          # Sampling algorithms
│   ├── sd.py                # Model loading (CLIP, VAE, UNet)
│   ├── model_detection.py   # Auto-detect model types
│   ├── model_patcher.py     # LoRA, dynamic patching
│   └── ldm/                 # Diffusion model implementations
└── web/                     # Frontend (LiteGraph.js based)
```

## Key Concepts

### Node-Based Workflow
ComfyUI uses a node graph system where each node performs a specific operation. Nodes are connected to form a workflow DAG (Directed Acyclic Graph).

### Execution Engine
The execution engine topologically sorts the node graph and executes nodes in order, with intelligent caching to skip unchanged nodes.

### Model Management
Smart VRAM management automatically loads/unloads models based on available memory, supporting model switching without restarts.

### Supported Models
- **Image**: SD 1.x, SD 2.x, SDXL, SD3, Flux, PixArt, AuraFlow
- **Video**: Mochi, LTXV, HunyuanVideo, Wan, Cosmos
- **Audio**: Stable Audio, ACE Step
- **3D**: Hunyuan3D

## Requirements

These notes are based on:
- ComfyUI version 0.3.x
- Python 3.10+
- PyTorch 2.x

## Contributing

Contributions are welcome! If you find any errors or want to add more detailed explanations, please open an issue or submit a pull request.

## References

- [ComfyUI GitHub](https://github.com/comfyanonymous/ComfyUI)
- [ComfyUI Documentation](https://docs.comfy.org/)
- [Stable Diffusion Paper](https://arxiv.org/abs/2112.10752)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Notes are written in Chinese (Simplified) for personal study purposes.*
