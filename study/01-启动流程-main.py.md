# ComfyUI 学习笔记 01：启动流程 (main.py)

## 概述

`main.py` 是 ComfyUI 的入口文件，负责初始化整个系统并启动 Web 服务器。

文件位置：`ComfyUI/main.py` (约 414 行)

---

## 启动流程图

```
python main.py
      │
      ├── 1. enable_args_parsing()     # 解析命令行参数
      │
      ├── 2. setup_logger()            # 初始化日志系统
      │
      ├── 3. apply_custom_paths()      # 配置模型路径
      │       └── 加载 extra_model_paths.yaml
      │
      ├── 4. execute_prestartup_script()  # 执行自定义节点预启动脚本
      │
      ├── 5. 导入核心模块
      │       ├── execution (执行引擎)
      │       ├── server (Web服务器)
      │       ├── nodes (节点系统)
      │       └── comfy.model_management (显存管理)
      │
      └── 6. start_comfyui()
              ├── PromptServer()         # 创建服务器实例
              ├── init_extra_nodes()     # 加载所有节点
              ├── add_routes()           # 注册API路由
              ├── prompt_worker 线程      # 启动工作流执行线程
              └── run()                  # 启动HTTP/WebSocket服务
```

---

## 分段详解

### 第一段：参数解析 (第 1-2 行)

```python
import comfy.options
comfy.options.enable_args_parsing()
```

**作用**：在任何其他模块导入之前，先启用命令行参数解析。

**为什么要这么早？** 因为很多模块的行为取决于命令行参数（如 `--cpu`、`--cuda-device` 等），必须在导入这些模块之前就解析好参数。

### 第二段：环境变量设置 (第 19-23 行)

```python
if __name__ == "__main__":
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
```

**作用**：禁用 HuggingFace 的遥测数据收集，保护隐私。

### 第三段：模型路径配置 (第 43-75 行)

```python
def apply_custom_paths():
    # 加载 extra_model_paths.yaml 配置文件
    extra_model_paths_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "extra_model_paths.yaml"
    )
    if os.path.isfile(extra_model_paths_config_path):
        utils.extra_config.load_extra_path_config(extra_model_paths_config_path)
```

**作用**：
- 加载自定义模型路径配置
- 设置输出目录、输入目录、用户目录
- 添加默认的模型保存路径（checkpoints、clip、vae、loras 等）

### 第四段：预启动脚本 (第 78-124 行)

```python
def execute_prestartup_script():
    node_paths = folder_paths.get_folder_paths("custom_nodes")
    for custom_node_path in node_paths:
        # 遍历每个自定义节点目录
        script_path = os.path.join(module_path, "prestartup_script.py")
        if os.path.exists(script_path):
            execute_script(script_path)  # 执行预启动脚本
```

**作用**：在加载节点之前，执行每个自定义节点的 `prestartup_script.py`。

**用途**：自定义节点可以用这个脚本做一些初始化工作，比如下载依赖、检查环境等。

### 第五段：GPU 设备配置 (第 141-172 行)

```python
if os.name == "nt":
    os.environ['MIMALLOC_PURGE_DELAY'] = '0'  # Windows 内存优化

if args.cuda_device is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
```

**作用**：
- Windows 系统的内存分配优化
- 设置使用哪个 GPU（通过 `--cuda-device` 参数）
- 支持 AMD ROCm、Intel OneAPI、Ascend NPU 等

### 第六段：核心模块导入 (第 177-186 行)

```python
import comfy.utils
import execution          # 工作流执行引擎
import server            # Web 服务器
from protocol import BinaryEventTypes  # WebSocket 二进制协议
import nodes             # 节点注册系统
import comfy.model_management  # 模型/显存管理
```

**关键点**：这些导入必须在参数解析和环境变量设置之后，因为它们的行为依赖于这些配置。

### 第七段：工作流执行线程 (第 200-276 行)

```python
def prompt_worker(q, server_instance):
    # 选择缓存策略
    cache_type = execution.CacheType.CLASSIC
    if args.cache_lru > 0:
        cache_type = execution.CacheType.LRU
    elif args.cache_ram > 0:
        cache_type = execution.CacheType.RAM_PRESSURE

    # 创建执行器
    e = execution.PromptExecutor(server_instance, cache_type=cache_type)

    while True:
        queue_item = q.get(timeout=timeout)  # 从队列获取任务
        if queue_item is not None:
            e.execute(item[2], prompt_id, extra_data, item[4])  # 执行工作流

        # 内存管理
        if need_gc:
            gc.collect()
            comfy.model_management.soft_empty_cache()
```

**作用**：
- 这是一个独立的后台线程
- 不断从队列中获取工作流任务并执行
- 执行完成后进行垃圾回收和显存清理

### 第八段：服务器启动 (第 333-394 行)

```python
def start_comfyui(asyncio_loop=None):
    # 1. 创建服务器实例
    prompt_server = server.PromptServer(asyncio_loop)

    # 2. 加载所有节点（核心节点 + 自定义节点 + API节点）
    asyncio_loop.run_until_complete(nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes),
        init_api_nodes=not args.disable_api_nodes
    ))

    # 3. 注册 API 路由
    prompt_server.add_routes()

    # 4. 设置进度回调
    hijack_progress(prompt_server)

    # 5. 启动工作流执行线程
    threading.Thread(target=prompt_worker, daemon=True,
                    args=(prompt_server.prompt_queue, prompt_server,)).start()

    # 6. 返回事件循环和启动函数
    return asyncio_loop, prompt_server, start_all
```

### 第九段：主入口 (第 397-413 行)

```python
if __name__ == "__main__":
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {}".format(comfyui_version.__version__))

    event_loop, _, start_all_func = start_comfyui()
    event_loop.run_until_complete(start_all_func())  # 阻塞运行服务器
```

---

## 重要概念

### 1. 双线程架构

ComfyUI 使用两个主要执行环境：
- **主线程（asyncio）**：运行 HTTP/WebSocket 服务器，处理 API 请求
- **工作线程（prompt_worker）**：执行实际的 AI 推理任务

### 2. 四种缓存策略

通过命令行参数选择：
- `--cache-classic`：经典模式，基于输入签名缓存
- `--cache-lru N`：LRU 模式，保留最近 N 个执行结果
- `--cache-ram N`：RAM 压力模式，根据内存压力自动清理
- `--cache-none`：禁用缓存

### 3. 模块加载顺序的重要性

```
命令行参数 → 环境变量 → 日志系统 → 路径配置 → 预启动脚本 → 核心模块 → 节点系统
```

这个顺序确保后面的模块可以正确读取配置。

---

## 常用命令行参数

| 参数 | 说明 |
|------|------|
| `--listen 0.0.0.0` | 监听所有网络接口 |
| `--port 8188` | 服务端口 |
| `--cuda-device 0` | 使用指定 GPU |
| `--cpu` | 仅使用 CPU |
| `--enable-manager` | 启用节点管理器 |
| `--disable-all-custom-nodes` | 禁用所有自定义节点 |
| `--preview-method auto` | 启用预览 |
| `--verbose` | 详细日志 |

---

## 下一步学习

- `server.py`：了解 Web 服务器和 API 路由
- `nodes.py`：了解节点系统如何定义和注册
- `execution.py`：了解工作流如何被执行
