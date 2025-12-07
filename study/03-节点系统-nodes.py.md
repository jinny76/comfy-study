# ComfyUI 学习笔记 03：节点系统 (nodes.py)

## 概述

`nodes.py` 是 ComfyUI 的节点注册中心，定义了所有核心节点并管理节点的加载。

文件位置：`ComfyUI/nodes.py` (约 2453 行)

---

## 节点定义规范

每个节点是一个 Python 类，必须包含以下属性：

### 必须属性

```python
class MyNode:
    @classmethod
    def INPUT_TYPES(cls):
        """定义输入参数"""
        return {
            "required": { ... },   # 必填输入
            "optional": { ... },   # 可选输入
            "hidden": { ... }      # 隐藏输入（系统自动填充）
        }

    RETURN_TYPES = ("TYPE1", "TYPE2")  # 输出类型元组
    FUNCTION = "execute"               # 执行函数名

    def execute(self, **inputs):       # 实际执行的函数
        return (output1, output2)      # 返回元组
```

### 可选属性

```python
class MyNode:
    CATEGORY = "sampling"              # 节点分类（在UI中的位置）
    DESCRIPTION = "节点描述"            # 节点说明
    RETURN_NAMES = ("name1", "name2")  # 输出名称
    OUTPUT_TOOLTIPS = ("tip1", "tip2") # 输出提示
    OUTPUT_NODE = True                 # 是否是输出节点（如SaveImage）
    DEPRECATED = True                  # 标记为废弃
    EXPERIMENTAL = True                # 标记为实验性
```

---

## 输入类型定义详解

### INPUT_TYPES 返回格式

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "参数名": ("类型", {配置选项}),
        },
        "optional": {
            "可选参数": ("类型", {配置选项}),
        },
        "hidden": {
            "prompt": "PROMPT",           # 当前工作流
            "extra_pnginfo": "EXTRA_PNGINFO",  # 额外PNG信息
            "unique_id": "UNIQUE_ID",     # 节点唯一ID
        }
    }
```

### 基础数据类型

| 类型 | 说明 | 配置选项示例 |
|------|------|-------------|
| `INT` | 整数 | `{"default": 512, "min": 64, "max": 4096, "step": 8}` |
| `FLOAT` | 浮点数 | `{"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}` |
| `STRING` | 字符串 | `{"default": "", "multiline": True}` |
| `BOOLEAN` | 布尔值 | `{"default": True}` |

### 选择列表

```python
# 静态列表
"sampler_name": (["euler", "euler_a", "dpm++"], {"default": "euler"})

# 动态列表（从文件夹读取）
"ckpt_name": (folder_paths.get_filename_list("checkpoints"), )
```

### AI 相关类型

| 类型 | 说明 |
|------|------|
| `MODEL` | 扩散模型 |
| `CLIP` | 文本编码器 |
| `VAE` | 变分自编码器 |
| `CONDITIONING` | 条件（正向/负向提示词编码） |
| `LATENT` | 潜空间图像 |
| `IMAGE` | 像素图像 (B, H, W, C) 格式 |
| `MASK` | 遮罩 |
| `CONTROL_NET` | ControlNet 模型 |

### 输入配置选项

```python
{
    "default": 值,              # 默认值
    "min": 最小值,              # 数值最小值
    "max": 最大值,              # 数值最大值
    "step": 步长,               # 调整步长
    "multiline": True,          # 多行文本
    "dynamicPrompts": True,     # 支持动态提示词
    "tooltip": "帮助文本",       # 鼠标悬停提示
    "control_after_generate": True,  # 生成后控制（用于seed）
}
```

---

## 核心节点示例分析

### 1. CLIPTextEncode - 文本编码

```python
class CLIPTextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True}),
                "clip": (IO.CLIP, )
            }
        }

    RETURN_TYPES = (IO.CONDITIONING,)
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(self, clip, text):
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens), )
```

**作用**：将文本提示词编码为 conditioning 向量

### 2. KSampler - 采样器

```python
class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "seed": ("INT", {"default": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler,
               positive, negative, latent_image, denoise=1.0):
        return common_ksampler(model, seed, steps, cfg, sampler_name,
                               scheduler, positive, negative, latent_image,
                               denoise=denoise)
```

**作用**：核心采样节点，执行扩散模型的去噪过程

### 3. VAEDecode - 解码

```python
class VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", ),
                "vae": ("VAE", )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, vae, samples):
        images = vae.decode(samples["samples"])
        return (images, )
```

**作用**：将潜空间图像解码为像素图像

### 4. SaveImage - 保存图片（输出节点）

```python
class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"})
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ()          # 无输出
    OUTPUT_NODE = True         # 标记为输出节点
    FUNCTION = "save_images"
    CATEGORY = "image"
```

**作用**：保存生成的图片，`OUTPUT_NODE = True` 表示这是一个终端节点

---

## 节点注册机制

### 核心注册字典

```python
# 节点类映射（类名 -> 类）
NODE_CLASS_MAPPINGS = {
    "KSampler": KSampler,
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "CLIPTextEncode": CLIPTextEncode,
    # ... 约 70 个核心节点
}

# 显示名称映射（类名 -> UI显示名）
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSampler": "KSampler",
    "CheckpointLoaderSimple": "Load Checkpoint",
    "CLIPTextEncode": "CLIP Text Encode (Prompt)",
    # ...
}

# Web 扩展目录
EXTENSION_WEB_DIRS = {}

# 已加载模块目录
LOADED_MODULE_DIRS = {}
```

### 节点加载流程

```
init_extra_nodes()
    │
    ├── init_builtin_extra_nodes()     # 加载 comfy_extras/ 下的扩展节点
    │       └── nodes_flux.py, nodes_sd3.py, nodes_video.py 等 60+ 文件
    │
    ├── init_builtin_api_nodes()       # 加载 comfy_api_nodes/ 下的 API 节点
    │       └── nodes_openai.py, nodes_stability.py 等
    │
    └── init_external_custom_nodes()   # 加载 custom_nodes/ 下的自定义节点
            └── 用户安装的第三方节点
```

### 自定义节点加载

```python
async def load_custom_node(module_path, base_node_names, module_parent="custom_nodes"):
    # 1. 导入模块
    module = importlib.import_module(module_name)

    # 2. 检查是否有 NODE_CLASS_MAPPINGS
    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)

        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

    # 3. 检查是否有 WEB_DIRECTORY（前端扩展）
    if hasattr(module, "WEB_DIRECTORY"):
        EXTENSION_WEB_DIRS[module_name] = module.WEB_DIRECTORY
```

---

## 自定义节点开发模板

### 基本模板

```python
# my_custom_nodes.py

class MyCustomNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1,
                    "tooltip": "Effect strength"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("output_image",)
    FUNCTION = "process"
    CATEGORY = "my_nodes/image"

    def process(self, input_image, strength):
        # input_image: torch.Tensor, shape (B, H, W, C)
        output = input_image * strength
        return (output,)


# 必须导出这两个字典
NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNode": "My Custom Node",
}
```

### 带前端扩展的模板

```python
# __init__.py

import os

# 指定前端 JS 文件目录
WEB_DIRECTORY = os.path.join(os.path.dirname(__file__), "js")

NODE_CLASS_MAPPINGS = { ... }
NODE_DISPLAY_NAME_MAPPINGS = { ... }
```

---

## 扩展节点目录 (comfy_extras/)

| 文件 | 功能 |
|------|------|
| `nodes_flux.py` | Flux 模型节点 |
| `nodes_sd3.py` | SD3 模型节点 |
| `nodes_video.py` | 视频处理节点 |
| `nodes_audio.py` | 音频处理节点 |
| `nodes_custom_sampler.py` | 自定义采样器 |
| `nodes_controlnet.py` | ControlNet 节点 |
| `nodes_model_merging.py` | 模型合并节点 |
| `nodes_mask.py` | 遮罩操作节点 |
| `nodes_images.py` | 图像处理节点 |
| `nodes_latent.py` | 潜空间操作节点 |

共 60+ 个扩展节点文件。

---

## 数据流类型图

```
文本提示词 (STRING)
     │
     ▼
┌─────────────┐
│ CLIPTextEncode │
└──────┬──────┘
       │
       ▼
  CONDITIONING ──────────────────────┐
                                     │
                                     ▼
模型文件 ──▶ CheckpointLoader ──▶ MODEL ──▶ KSampler ──▶ LATENT
                  │                           ▲            │
                  ▼                           │            ▼
                 VAE ─────────────────────────┘       VAEDecode
                                                          │
                                                          ▼
                                                       IMAGE
                                                          │
                                                          ▼
                                                     SaveImage
```

---

## 关键函数

### before_node_execution()

```python
def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()
```

每个节点执行前调用，检查是否被用户中断。

### interrupt_processing()

```python
def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)
```

中断当前执行（响应用户的中断请求）。

---

## 下一步学习

- `execution.py`：了解节点如何被调度执行
- `comfy_extras/nodes_custom_sampler.py`：了解高级采样器
- 实践：写一个简单的自定义节点
