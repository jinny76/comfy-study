# ComfyUI 学习笔记 02：服务器架构 (server.py)

## 概述

`server.py` 是 ComfyUI 的 Web 服务器，基于 **aiohttp** 框架实现，提供：
- HTTP REST API（提交工作流、查询状态等）
- WebSocket 实时通信（进度更新、预览图像等）
- 静态文件服务（前端界面）

文件位置：`ComfyUI/server.py` (约 1096 行)

---

## 核心类：PromptServer

```python
class PromptServer():
    def __init__(self, loop):
        PromptServer.instance = self  # 单例模式

        # 管理器
        self.user_manager = UserManager()
        self.model_file_manager = ModelFileManager()
        self.custom_node_manager = CustomNodeManager()

        # 执行队列
        self.prompt_queue = execution.PromptQueue(self)

        # WebSocket 连接管理
        self.sockets = dict()           # sid -> WebSocket
        self.sockets_metadata = dict()  # sid -> 元数据

        # aiohttp 应用
        self.app = web.Application(middlewares=middlewares)
```

---

## 中间件（Middleware）

中间件是请求处理的"拦截器"，在请求到达路由之前/之后执行。

### 1. 缓存控制 (cache_control)
```python
from middleware.cache_middleware import cache_control
```
控制静态资源的缓存策略。

### 2. 废弃警告 (deprecation_warning)
```python
@web.middleware
async def deprecation_warning(request, handler):
    if path.startswith("/scripts/ui") or path.startswith("/extensions/core/"):
        logging.warning(f"Detected import of deprecated legacy API: {path}")
    return await handler(request)
```
警告使用过时 API 的自定义节点。

### 3. 响应压缩 (compress_body)
```python
@web.middleware
async def compress_body(request, handler):
    if "gzip" in accept_encoding:
        response.enable_compression()
    return response
```
对 JSON 和文本响应启用 gzip 压缩。

### 4. CORS 跨域 (cors_middleware)
```python
def create_cors_middleware(allowed_origin):
    response.headers['Access-Control-Allow-Origin'] = allowed_origin
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, DELETE, PUT, OPTIONS'
```
允许跨域请求（通过 `--enable-cors-header` 启用）。

### 5. Origin 安全检查 (origin_only_middleware)
```python
def create_origin_only_middleware():
    # 防止恶意网站向 localhost 发送请求
    if host_domain != origin_domain:
        return web.Response(status=403)
```
**安全功能**：防止 CSRF 攻击，确保请求来源与目标主机匹配。

---

## REST API 路由

### 工作流相关

| 端点 | 方法 | 功能 |
|------|------|------|
| `/prompt` | GET | 获取队列状态 |
| `/prompt` | POST | **提交工作流执行** |
| `/queue` | GET | 获取当前队列 |
| `/queue` | POST | 清空/删除队列项 |
| `/interrupt` | POST | 中断执行 |
| `/history` | GET | 获取执行历史 |
| `/history/{prompt_id}` | GET | 获取特定执行记录 |
| `/free` | POST | 释放内存/卸载模型 |

### 节点信息

| 端点 | 方法 | 功能 |
|------|------|------|
| `/object_info` | GET | 获取所有节点信息 |
| `/object_info/{node_class}` | GET | 获取特定节点信息 |

### 模型和资源

| 端点 | 方法 | 功能 |
|------|------|------|
| `/models` | GET | 列出模型类型 |
| `/models/{folder}` | GET | 列出指定类型的模型 |
| `/embeddings` | GET | 列出 embeddings |
| `/extensions` | GET | 列出前端扩展 |

### 文件操作

| 端点 | 方法 | 功能 |
|------|------|------|
| `/upload/image` | POST | 上传图片 |
| `/upload/mask` | POST | 上传遮罩 |
| `/view` | GET | 查看图片 |
| `/view_metadata/{folder}` | GET | 查看 safetensors 元数据 |

### 系统信息

| 端点 | 方法 | 功能 |
|------|------|------|
| `/system_stats` | GET | 系统状态（RAM、VRAM等） |
| `/features` | GET | 服务器特性标志 |

---

## 核心 API 详解

### 1. 提交工作流 POST /prompt

```python
@routes.post("/prompt")
async def post_prompt(request):
    json_data = await request.json()

    # 验证工作流
    valid = await execution.validate_prompt(prompt_id, prompt)

    if valid[0]:
        # 放入执行队列
        self.prompt_queue.put((number, prompt_id, prompt, extra_data, outputs_to_execute))
        return {"prompt_id": prompt_id, "number": number}
    else:
        return {"error": valid[1]}, status=400
```

**请求格式**：
```json
{
    "prompt": {
        "1": {
            "class_type": "CheckpointLoader",
            "inputs": {"ckpt_name": "model.safetensors"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "a beautiful landscape",
                "clip": ["1", 1]
            }
        }
    },
    "client_id": "uuid-string",
    "extra_data": {}
}
```

**关键点**：
- `prompt` 是节点图的 JSON 表示
- 输入引用其他节点：`["node_id", output_index]`
- `client_id` 用于 WebSocket 消息定向

### 2. 获取节点信息 GET /object_info

```python
def node_info(node_class):
    obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
    info = {
        'input': obj_class.INPUT_TYPES(),
        'output': obj_class.RETURN_TYPES,
        'output_name': obj_class.RETURN_NAMES,
        'category': obj_class.CATEGORY,
        'display_name': NODE_DISPLAY_NAME_MAPPINGS.get(node_class),
        'description': obj_class.DESCRIPTION
    }
    return info
```

**返回示例**：
```json
{
    "CheckpointLoader": {
        "input": {
            "required": {
                "ckpt_name": [["model.safetensors", "other.ckpt"], {}]
            }
        },
        "output": ["MODEL", "CLIP", "VAE"],
        "category": "loaders",
        "display_name": "Load Checkpoint"
    }
}
```

### 3. 系统状态 GET /system_stats

```python
@routes.get("/system_stats")
async def system_stats(request):
    return {
        "system": {
            "os": sys.platform,
            "ram_total": ram_total,
            "ram_free": ram_free,
            "comfyui_version": __version__,
            "python_version": sys.version,
            "pytorch_version": torch_version
        },
        "devices": [{
            "name": "NVIDIA GeForce RTX 4090",
            "vram_total": 24576,
            "vram_free": 20000
        }]
    }
```

---

## WebSocket 通信

### 连接建立

```python
@routes.get('/ws')
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # 获取或生成客户端ID
    sid = request.rel_url.query.get('clientId', '') or uuid.uuid4().hex

    # 保存连接
    self.sockets[sid] = ws

    # 发送初始状态
    await self.send("status", {"status": self.get_queue_info(), "sid": sid}, sid)
```

**连接URL**：`ws://localhost:8188/ws?clientId=your-uuid`

### 特性协商

客户端连接后可以发送特性标志：
```json
{
    "type": "feature_flags",
    "data": {
        "supports_preview_metadata": true
    }
}
```

服务器响应支持的特性。

### 消息类型

**JSON 消息**：
```python
await self.send_json(event, data, sid)
# 格式: {"type": "event_name", "data": {...}}
```

| 事件 | 说明 |
|------|------|
| `status` | 队列状态更新 |
| `executing` | 正在执行的节点 |
| `executed` | 节点执行完成 |
| `progress` | 采样进度 |
| `execution_start` | 开始执行 |
| `execution_cached` | 使用缓存 |
| `execution_error` | 执行错误 |

**二进制消息**：
```python
await self.send_bytes(event, data, sid)
# 格式: [4字节事件类型][数据]
```

| 事件类型 | 值 | 说明 |
|----------|---|------|
| PREVIEW_IMAGE | 1 | 预览图片 |
| PREVIEW_IMAGE_WITH_METADATA | 2 | 带元数据的预览 |
| TEXT | 3 | 文本消息 |

### 发送消息的两种方式

```python
# 1. 异步发送（在 async 函数中）
await self.send("progress", {"value": 50, "max": 100}, sid)

# 2. 同步发送（从其他线程，如 prompt_worker）
self.send_sync("progress", {"value": 50, "max": 100}, sid)
```

`send_sync` 是线程安全的，通过消息队列传递给事件循环：
```python
def send_sync(self, event, data, sid=None):
    self.loop.call_soon_threadsafe(
        self.messages.put_nowait, (event, data, sid))
```

---

## 图片预览机制

```python
async def send_image(self, image_data, sid=None):
    image_type = image_data[0]  # "JPEG" 或 "PNG"
    image = image_data[1]        # PIL Image
    max_size = image_data[2]     # 最大尺寸

    # 缩放图片
    if max_size:
        image = ImageOps.contain(image, (max_size, max_size))

    # 编码并发送
    bytesIO = BytesIO()
    image.save(bytesIO, format=image_type)
    await self.send_bytes(BinaryEventTypes.PREVIEW_IMAGE, bytesIO.getvalue(), sid)
```

**二进制格式**：
```
[4字节: 图片类型 1=JPEG, 2=PNG][图片数据]
```

---

## 文件上传安全

```python
def image_upload(post):
    # 1. 检查路径安全
    filepath = os.path.abspath(os.path.join(upload_dir, filename))
    if os.path.commonpath((upload_dir, filepath)) != upload_dir:
        return web.Response(status=400)  # 防止路径遍历攻击

    # 2. 处理文件名冲突
    while os.path.exists(filepath):
        if compare_image_hash(filepath, image):  # 重复文件
            break
        filename = f"{name} ({i}){ext}"  # 重命名
        i += 1

    # 3. 保存文件
    with open(filepath, "wb") as f:
        f.write(image.file.read())
```

---

## 路由注册流程

```python
def add_routes(self):
    # 1. 注册各管理器的路由
    self.user_manager.add_routes(self.routes)
    self.model_file_manager.add_routes(self.routes)
    self.custom_node_manager.add_routes(self.routes, self.app)

    # 2. 添加 /api 前缀的路由（新版本）
    for route in self.routes:
        api_routes.route(route.method, "/api" + route.path)(route.handler)

    # 3. 添加自定义节点的 web 扩展
    for name, dir in nodes.EXTENSION_WEB_DIRS.items():
        self.app.add_routes([web.static('/extensions/' + name, dir)])

    # 4. 静态文件服务（前端）
    self.app.add_routes([web.static('/', self.web_root)])
```

**API 路由兼容**：
- 旧版：`/prompt`, `/queue`, `/object_info`
- 新版：`/api/prompt`, `/api/queue`, `/api/object_info`

---

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                       PromptServer                          │
├─────────────────────────────────────────────────────────────┤
│  Middlewares: cache → deprecation → compress → cors/origin  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │  REST API   │   │  WebSocket   │   │  Static Files   │  │
│  │  /prompt    │   │  /ws         │   │  /              │  │
│  │  /queue     │   │              │   │  /extensions    │  │
│  │  /object_info│   │              │   │  /templates     │  │
│  └──────┬──────┘   └──────┬───────┘   └─────────────────┘  │
│         │                 │                                 │
│         ▼                 ▼                                 │
│  ┌─────────────────────────────────────┐                   │
│  │         PromptQueue                 │                   │
│  │  (执行队列，连接工作线程)              │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  prompt_worker  │
                    │  (执行线程)      │
                    └─────────────────┘
```

---

## 关键设计总结

1. **单例模式**：`PromptServer.instance` 全局可访问
2. **异步架构**：基于 asyncio + aiohttp
3. **双通道通信**：REST API + WebSocket
4. **线程安全**：`send_sync` 跨线程消息传递
5. **安全机制**：Origin 检查、路径验证、CORS 控制
6. **可扩展性**：中间件链、路由注册、管理器模块化

---

## 下一步学习

- `nodes.py`：节点系统如何定义
- `execution.py`：工作流如何执行
- `comfy_execution/graph.py`：图依赖解析
