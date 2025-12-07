# ComfyUI 插件系统源码分析

> 源码文件: `nodes.py` (节点加载部分，约 2400+ 行)

## 概述

ComfyUI 的"插件系统"本质上是**自定义节点（Custom Nodes）机制**。它允许第三方开发者通过 Python 模块扩展 ComfyUI 的功能，无需修改核心代码。

系统支持三类节点来源：
1. **内置节点** - `nodes.py` 中定义的 ~70 个基础节点
2. **内置扩展节点** - `comfy_extras/` 目录下的 ~80 个官方扩展
3. **外部自定义节点** - `custom_nodes/` 目录下的第三方插件

## 核心注册机制

### 全局注册表

```python
# 节点类映射：节点ID → 节点类
NODE_CLASS_MAPPINGS = {
    "KSampler": KSampler,
    "CheckpointLoaderSimple": CheckpointLoaderSimple,
    "CLIPTextEncode": CLIPTextEncode,
    # ... 约70个内置节点
}

# 显示名映射：节点ID → UI显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "KSampler": "KSampler",
    "CheckpointLoaderSimple": "Load Checkpoint",
    "CLIPTextEncode": "CLIP Text Encode (Prompt)",
    # ...
}

# 扩展的 Web 目录（用于前端扩展）
EXTENSION_WEB_DIRS = {}

# 已加载模块及其目录
LOADED_MODULE_DIRS = {}
```

### 节点加载函数

```python
async def load_custom_node(module_path: str, ignore=set(), module_parent="custom_nodes") -> bool:
    """
    动态加载自定义节点模块

    Args:
        module_path: 模块路径（.py 文件或包目录）
        ignore: 要忽略的节点名集合（避免覆盖）
        module_parent: 父模块名（用于日志和追踪）

    Returns:
        bool: 是否成功加载
    """
    module_name = get_module_name(module_path)

    # 使用 importlib 动态加载模块
    if os.path.isfile(module_path):
        module_spec = importlib.util.spec_from_file_location(sys_module_name, module_path)
    else:
        module_spec = importlib.util.spec_from_file_location(
            sys_module_name,
            os.path.join(module_path, "__init__.py")
        )

    module = importlib.util.module_from_spec(module_spec)
    sys.modules[sys_module_name] = module
    module_spec.loader.exec_module(module)

    # 记录已加载模块
    LOADED_MODULE_DIRS[module_name] = os.path.abspath(module_dir)

    # 处理 Web 扩展目录
    if hasattr(module, "WEB_DIRECTORY"):
        web_dir = os.path.abspath(os.path.join(module_dir, module.WEB_DIRECTORY))
        if os.path.isdir(web_dir):
            EXTENSION_WEB_DIRS[module_name] = web_dir

    # V1 节点定义方式
    if hasattr(module, "NODE_CLASS_MAPPINGS"):
        for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
            if name not in ignore:
                NODE_CLASS_MAPPINGS[name] = node_cls
                node_cls.RELATIVE_PYTHON_MODULE = f"{module_parent}.{module_name}"

        if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS"):
            NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        return True

    # V3 扩展定义方式（新版 API）
    elif hasattr(module, "comfy_entrypoint"):
        entrypoint = getattr(module, "comfy_entrypoint")
        extension = await entrypoint()  # 支持异步
        node_list = await extension.get_node_list()
        for node_cls in node_list:
            schema = node_cls.GET_SCHEMA()
            NODE_CLASS_MAPPINGS[schema.node_id] = node_cls
        return True

    else:
        logging.warning(f"Skip {module_path}: no NODE_CLASS_MAPPINGS or comfy_entrypoint")
        return False
```

## 节点定义规范

### V1 规范（传统方式）

```python
# my_custom_node/__init__.py

class MyCustomNode:
    """自定义节点类"""

    @classmethod
    def INPUT_TYPES(cls):
        """定义输入类型"""
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)           # 输出类型
    RETURN_NAMES = ("processed_image",) # 输出名称（可选）
    FUNCTION = "process"                 # 执行函数名
    CATEGORY = "My Nodes/Processing"     # 节点分类路径

    def process(self, image, strength, mask=None):
        # 节点逻辑
        result = do_something(image, strength, mask)
        return (result,)  # 必须返回元组


# 必须导出的映射
NODE_CLASS_MAPPINGS = {
    "MyCustomNode": MyCustomNode,
}

# 可选的显示名映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "MyCustomNode": "My Custom Node",
}

# 可选的 Web 扩展目录
WEB_DIRECTORY = "./js"
```

### V3 规范（新版 API）

```python
# my_custom_node/__init__.py

from comfy_api.latest import io, ComfyExtension

class MyNodeV3(io.ComfyNode):
    """V3 节点定义"""

    @classmethod
    def GET_SCHEMA(cls):
        return io.NodeSchema(
            node_id="MyNodeV3",
            display_name="My Node V3",
            category="My Nodes",
            inputs=[
                io.Input("image", io.IMAGE),
                io.Input("strength", io.FLOAT, default=1.0),
            ],
            outputs=[
                io.Output("image", io.IMAGE),
            ]
        )

    def execute(self, image, strength):
        return (process(image, strength),)


def comfy_entrypoint() -> ComfyExtension:
    """V3 入口点"""
    extension = ComfyExtension()
    extension.add_node(MyNodeV3)
    return extension
```

## 节点加载流程

### 初始化顺序

```python
async def init_extra_nodes(init_custom_nodes=True, init_api_nodes=True):
    """节点初始化入口"""

    # 1. 注册公共 API
    await init_public_apis()

    # 2. 加载内置扩展节点 (comfy_extras/)
    import_failed = await init_builtin_extra_nodes()

    # 3. 加载 API 节点 (comfy_api_nodes/) - 可选
    if init_api_nodes:
        import_failed += await init_builtin_api_nodes()

    # 4. 加载外部自定义节点 (custom_nodes/)
    if init_custom_nodes:
        await init_external_custom_nodes()
```

### 外部节点加载

```python
async def init_external_custom_nodes():
    """加载 custom_nodes 目录下的所有节点"""

    # 获取基础节点名，用于检测冲突
    base_node_names = set(NODE_CLASS_MAPPINGS.keys())

    # 获取所有 custom_nodes 路径（支持多路径）
    node_paths = folder_paths.get_folder_paths("custom_nodes")

    for custom_node_path in node_paths:
        possible_modules = os.listdir(custom_node_path)

        for possible_module in possible_modules:
            module_path = os.path.join(custom_node_path, possible_module)

            # 跳过非 Python 文件
            if os.path.isfile(module_path) and not module_path.endswith(".py"):
                continue

            # 跳过 .disabled 后缀的模块
            if module_path.endswith(".disabled"):
                continue

            # 检查白名单（如果启用了禁用所有自定义节点）
            if args.disable_all_custom_nodes:
                if possible_module not in args.whitelist_custom_nodes:
                    continue

            # 记录加载时间
            time_before = time.perf_counter()
            success = await load_custom_node(module_path, base_node_names)
            node_import_times.append((time.perf_counter() - time_before, module_path, success))
```

## 节点目录结构

### 内置扩展 (comfy_extras/)

约 80 个官方扩展节点模块：

```
comfy_extras/
├── nodes_flux.py              # Flux 模型支持
├── nodes_sd3.py               # SD3 模型支持
├── nodes_hunyuan.py           # 混元模型支持
├── nodes_video_model.py       # 视频模型支持
├── nodes_custom_sampler.py    # 自定义采样器
├── nodes_controlnet.py        # ControlNet 扩展
├── nodes_model_merging.py     # 模型合并
├── nodes_upscale_model.py     # 放大模型
├── nodes_mask.py              # 蒙版操作
├── nodes_audio.py             # 音频处理
└── ... (约 80 个文件)
```

### API 节点 (comfy_api_nodes/)

云端 API 集成节点：

```
comfy_api_nodes/
├── nodes_openai.py            # OpenAI API
├── nodes_stability.py         # Stability AI
├── nodes_runway.py            # Runway
├── nodes_kling.py             # 可灵
├── nodes_bfl.py               # Black Forest Labs
├── nodes_luma.py              # Luma AI
└── ... (约 20 个文件)
```

### 外部自定义节点 (custom_nodes/)

第三方开发者的插件：

```
custom_nodes/
├── ComfyUI-Manager/           # 节点管理器
│   ├── __init__.py
│   ├── js/                    # Web 扩展
│   └── requirements.txt
├── ComfyUI-Impact-Pack/       # 检测分割包
├── ComfyUI_IPAdapter_plus/    # IP-Adapter
└── ...
```

## Web 扩展机制

自定义节点可以通过 `WEB_DIRECTORY` 扩展前端：

```python
# __init__.py
WEB_DIRECTORY = "./js"

# 或使用 pyproject.toml
# [tool.comfy]
# web = "js"
```

前端扩展文件会被自动注册到 `EXTENSION_WEB_DIRS`，服务器启动时会加载这些 JS 文件。

```javascript
// js/my_extension.js
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "MyExtension",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 节点注册前的钩子
    },

    async nodeCreated(node) {
        // 节点创建后的钩子
    }
});
```

## 禁用与白名单

### 禁用单个节点

将目录重命名为 `.disabled` 后缀：

```bash
mv custom_nodes/SomeNode custom_nodes/SomeNode.disabled
```

### 全局禁用与白名单

启动参数：

```bash
# 禁用所有自定义节点
python main.py --disable-all-custom-nodes

# 禁用所有，但白名单例外
python main.py --disable-all-custom-nodes --whitelist-custom-nodes ComfyUI-Manager
```

## 节点冲突处理

```python
# load_custom_node 中的冲突检测
for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
    if name not in ignore:  # ignore 包含已注册的基础节点名
        NODE_CLASS_MAPPINGS[name] = node_cls
```

如果自定义节点与内置节点同名，自定义节点会被跳过（不覆盖）。

## 加载时间统计

```python
# 记录每个模块的加载时间
node_import_times = []
time_before = time.perf_counter()
success = await load_custom_node(module_path, base_node_names)
node_import_times.append((time.perf_counter() - time_before, module_path, success))

# 最后按时间排序输出
if len(googbye_times) > 0:
    print("\nImport times for custom nodes:")
    for t, n, s in sorted(node_import_times, reverse=True):
        print(f"  {t:.2f}s {n} {'(failed)' if not s else ''}")
```

## 数据流图

```
启动时
    │
    ▼
init_extra_nodes()
    │
    ├─► init_public_apis()           # 注册 API 版本
    │
    ├─► init_builtin_extra_nodes()   # 加载 comfy_extras/
    │       │
    │       └─► load_custom_node() × 80
    │
    ├─► init_builtin_api_nodes()     # 加载 comfy_api_nodes/
    │       │
    │       └─► load_custom_node() × 20
    │
    └─► init_external_custom_nodes() # 加载 custom_nodes/
            │
            └─► load_custom_node() × N
                    │
                    ├─► importlib 动态加载
                    │
                    ├─► 检测 NODE_CLASS_MAPPINGS (V1)
                    │   或 comfy_entrypoint (V3)
                    │
                    ├─► 注册到全局 NODE_CLASS_MAPPINGS
                    │
                    └─► 注册 WEB_DIRECTORY (如果有)
```

## 开发自定义节点

### 最小示例

```python
# custom_nodes/my_node/__init__.py

class AddNumbers:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "a": ("FLOAT", {"default": 0}),
                "b": ("FLOAT", {"default": 0}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "add"
    CATEGORY = "Math"

    def add(self, a, b):
        return (a + b,)

NODE_CLASS_MAPPINGS = {"AddNumbers": AddNumbers}
NODE_DISPLAY_NAME_MAPPINGS = {"AddNumbers": "Add Numbers"}
```

### 支持的数据类型

| 类型 | 说明 | Python 类型 |
|------|------|-------------|
| IMAGE | 图像 | torch.Tensor [B,H,W,C] |
| LATENT | 潜空间 | dict {"samples": Tensor} |
| CONDITIONING | 条件 | list |
| MODEL | 模型 | ModelPatcher |
| CLIP | 文本编码器 | CLIP |
| VAE | VAE | VAE |
| MASK | 蒙版 | torch.Tensor [B,H,W] |
| INT | 整数 | int |
| FLOAT | 浮点数 | float |
| STRING | 字符串 | str |
| BOOLEAN | 布尔值 | bool |

### 输入参数选项

```python
"param_name": ("TYPE", {
    "default": value,           # 默认值
    "min": 0,                   # 最小值
    "max": 100,                 # 最大值
    "step": 1,                  # 步进
    "multiline": True,          # 多行文本 (STRING)
    "dynamicPrompts": True,     # 动态提示词 (STRING)
    "tooltip": "说明文字",       # 提示信息
})
```

## 总结

ComfyUI 的插件系统特点：

1. **简单直接** - 只需实现 `NODE_CLASS_MAPPINGS` 即可
2. **动态加载** - 使用 `importlib` 运行时加载
3. **无侵入性** - 不修改核心代码
4. **前后端扩展** - 同时支持 Python 后端和 JS 前端扩展
5. **版本演进** - V1 (字典映射) → V3 (声明式 Schema)
6. **加载时间统计** - 方便排查慢启动问题
7. **禁用机制** - 支持 .disabled 后缀和白名单

这套机制使得 ComfyUI 拥有了极其活跃的插件生态，目前社区已有数百个自定义节点包。
