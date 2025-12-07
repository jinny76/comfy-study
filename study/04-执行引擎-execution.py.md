# ComfyUI 学习笔记 04：执行引擎 (execution.py)

## 概述

`execution.py` 是 ComfyUI 的工作流执行引擎，负责：
- 验证工作流（prompt）的合法性
- 解析节点依赖关系
- 按拓扑排序执行节点
- 管理执行缓存
- 处理异步和动态子图

文件位置：`ComfyUI/execution.py` (约 1239 行)

---

## 核心类关系图

```
                    PromptQueue
                        │
                        │ put()/get()
                        ▼
                  PromptExecutor
                        │
                        │ execute_async()
                        ▼
    ┌───────────────────┼───────────────────┐
    │                   │                   │
    ▼                   ▼                   ▼
DynamicPrompt      CacheSet           ExecutionList
(工作流图)          (缓存管理)          (执行调度)
```

---

## PromptQueue - 任务队列

任务队列管理待执行的工作流，线程安全设计。

```python
class PromptQueue:
    def __init__(self, server):
        self.mutex = threading.RLock()      # 可重入锁
        self.not_empty = threading.Condition(self.mutex)
        self.queue = []                      # 优先级队列（堆）
        self.currently_running = {}          # 正在执行的任务
        self.history = {}                    # 执行历史

    def put(self, item):
        """添加任务到队列"""
        with self.mutex:
            heapq.heappush(self.queue, item)
            self.not_empty.notify()

    def get(self, timeout=None):
        """获取下一个任务（阻塞）"""
        with self.not_empty:
            while len(self.queue) == 0:
                self.not_empty.wait(timeout=timeout)
            item = heapq.heappop(self.queue)
            self.currently_running[self.task_counter] = item
            return (item, self.task_counter)

    def task_done(self, item_id, history_result, status):
        """标记任务完成，存入历史"""
        with self.mutex:
            prompt = self.currently_running.pop(item_id)
            self.history[prompt[1]] = {
                "prompt": prompt,
                "outputs": {},
                "status": status
            }
```

### 任务项格式

```python
# queue 中的 item 格式
(
    priority,        # 优先级（数字越小越优先）
    prompt_id,       # 唯一标识（UUID）
    prompt,          # 工作流图（节点字典）
    extra_data,      # 额外数据（client_id, 元数据等）
    outputs_to_execute  # 要执行的输出节点列表
)
```

---

## PromptExecutor - 执行器

核心执行器，负责实际执行工作流。

```python
class PromptExecutor:
    def __init__(self, server, cache_type=False, cache_args=None):
        self.server = server
        self.caches = CacheSet(cache_type, cache_args)

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        """同步执行入口"""
        asyncio.run(self.execute_async(prompt, prompt_id, extra_data, execute_outputs))

    async def execute_async(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        """异步执行主循环"""
        # 1. 初始化
        nodes.interrupt_processing(False)
        dynamic_prompt = DynamicPrompt(prompt)

        # 2. 设置缓存
        is_changed_cache = IsChangedCache(prompt_id, dynamic_prompt, self.caches.outputs)
        for cache in self.caches.all:
            await cache.set_prompt(dynamic_prompt, prompt.keys(), is_changed_cache)
            cache.clean_unused()

        # 3. 构建执行列表
        execution_list = ExecutionList(dynamic_prompt, self.caches.outputs)
        for node_id in execute_outputs:
            execution_list.add_node(node_id)

        # 4. 执行循环
        while not execution_list.is_empty():
            node_id, error, ex = await execution_list.stage_node_execution()
            if error is not None:
                self.handle_execution_error(...)
                break

            result, error, ex = await execute(
                self.server, dynamic_prompt, self.caches,
                node_id, extra_data, executed, prompt_id,
                execution_list, pending_subgraph_results,
                pending_async_nodes, ui_node_outputs
            )

            if result == ExecutionResult.FAILURE:
                break
            elif result == ExecutionResult.PENDING:
                execution_list.unstage_node_execution()
            else:  # SUCCESS
                execution_list.complete_node_execution()
```

---

## CacheSet - 缓存管理

管理节点执行结果的缓存，避免重复计算。

### 四种缓存策略

```python
class CacheType(Enum):
    CLASSIC = 0      # 经典模式：基于输入签名
    LRU = 1          # LRU模式：保留最近N个结果
    NONE = 2         # 禁用缓存
    RAM_PRESSURE = 3 # RAM压力模式：根据内存自动清理
```

### 缓存初始化

```python
class CacheSet:
    def __init__(self, cache_type=None, cache_args={}):
        if cache_type == CacheType.NONE:
            self.init_null_cache()
        elif cache_type == CacheType.RAM_PRESSURE:
            self.init_ram_cache(cache_args.get("ram", 16.0))
        elif cache_type == CacheType.LRU:
            self.init_lru_cache(cache_args.get("lru", 0))
        else:
            self.init_classic_cache()

    def init_classic_cache(self):
        # 基于输入签名的缓存
        self.outputs = HierarchicalCache(CacheKeySetInputSignature)
        self.objects = HierarchicalCache(CacheKeySetID)

    def init_lru_cache(self, cache_size):
        # LRU 缓存，保留最近 cache_size 个结果
        self.outputs = LRUCache(CacheKeySetInputSignature, max_size=cache_size)
        self.objects = HierarchicalCache(CacheKeySetID)
```

### 缓存条目

```python
class CacheEntry(NamedTuple):
    ui: dict      # UI 输出数据（预览图等）
    outputs: list # 节点输出数据
```

---

## DynamicPrompt - 动态工作流

支持在执行过程中动态创建新节点（子图）。

```python
class DynamicPrompt:
    def __init__(self, original_prompt):
        self.original_prompt = original_prompt  # 原始工作流
        self.ephemeral_prompt = {}              # 动态创建的节点
        self.ephemeral_parents = {}             # 动态节点的父节点
        self.ephemeral_display = {}             # 显示映射

    def get_node(self, node_id):
        """获取节点（先查动态，再查原始）"""
        if node_id in self.ephemeral_prompt:
            return self.ephemeral_prompt[node_id]
        if node_id in self.original_prompt:
            return self.original_prompt[node_id]
        raise NodeNotFoundError(f"Node {node_id} not found")

    def add_ephemeral_node(self, node_id, node_info, parent_id, display_id):
        """添加动态节点"""
        self.ephemeral_prompt[node_id] = node_info
        self.ephemeral_parents[node_id] = parent_id
        self.ephemeral_display[node_id] = display_id
```

---

## ExecutionList - 执行调度

基于拓扑排序的执行调度器。

```python
class ExecutionList(TopologicalSort):
    def __init__(self, dynprompt, output_cache):
        super().__init__(dynprompt)
        self.output_cache = output_cache
        self.staged_node_id = None

    def add_node(self, node_id):
        """添加节点到执行列表（递归添加依赖）"""
        # 遍历输入，找到所有依赖的节点
        inputs = self.dynprompt.get_node(node_id)["inputs"]
        for input_name in inputs:
            value = inputs[input_name]
            if is_link(value):  # 是链接
                from_node_id = value[0]
                if not self.is_cached(from_node_id):
                    self.add_node(from_node_id)  # 递归添加依赖

    async def stage_node_execution(self):
        """选择下一个可执行的节点"""
        # 选择没有未完成依赖的节点
        for node_id in self.pendingNodes:
            if self.blockCount[node_id] == 0:
                self.staged_node_id = node_id
                return (node_id, None, None)

    def complete_node_execution(self):
        """标记节点执行完成"""
        node_id = self.staged_node_id
        # 解除被该节点阻塞的其他节点
        for blocked_node in self.blocking[node_id]:
            self.blockCount[blocked_node] -= 1
        del self.pendingNodes[node_id]
```

### 拓扑排序示意

```
原始图：
    A ──▶ B ──▶ D
          │
          ▼
          C ──▶ E

执行顺序：
1. A (无依赖)
2. B (依赖A，A完成后可执行)
3. C (依赖B)
4. D (依赖B)  ← C和D可以并行
5. E (依赖C)
```

---

## 单节点执行流程

`execute()` 函数执行单个节点：

```python
async def execute(server, dynprompt, caches, current_item, ...):
    unique_id = current_item
    class_type = dynprompt.get_node(unique_id)['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

    # 1. 检查缓存
    cached = caches.outputs.get(unique_id)
    if cached is not None:
        server.send_sync("executed", {...}, server.client_id)
        return (ExecutionResult.SUCCESS, None, None)

    # 2. 准备输入数据
    input_data_all, missing_keys, v3_data = get_input_data(
        inputs, class_def, unique_id, execution_list, dynprompt, extra_data
    )

    # 3. 发送执行状态
    server.send_sync("executing", {"node": unique_id, ...}, server.client_id)

    # 4. 获取或创建节点实例
    obj = caches.objects.get(unique_id)
    if obj is None:
        obj = class_def()
        caches.objects.set(unique_id, obj)

    # 5. 检查懒加载输入
    if hasattr(obj, "check_lazy_status"):
        required_inputs = await obj.check_lazy_status(...)
        if len(required_inputs) > 0:
            return (ExecutionResult.PENDING, None, None)

    # 6. 执行节点
    output_data, output_ui, has_subgraph = await get_output_data(
        prompt_id, unique_id, obj, input_data_all, ...
    )

    # 7. 处理子图（动态节点）
    if has_subgraph:
        # 添加动态创建的节点
        for node_id, node_info in new_graph.items():
            dynprompt.add_ephemeral_node(node_id, node_info, unique_id, display_id)
        return (ExecutionResult.PENDING, None, None)

    # 8. 缓存结果
    cache_entry = CacheEntry(ui=ui_outputs.get(unique_id), outputs=output_data)
    caches.outputs.set(unique_id, cache_entry)

    return (ExecutionResult.SUCCESS, None, None)
```

---

## 验证流程

### validate_prompt()

验证整个工作流的合法性：

```python
async def validate_prompt(prompt_id, prompt):
    outputs = set()
    # 1. 找出所有输出节点
    for node_id in prompt:
        class_def = nodes.NODE_CLASS_MAPPINGS[prompt[node_id]['class_type']]
        if hasattr(class_def, 'OUTPUT_NODE') and class_def.OUTPUT_NODE:
            outputs.add(node_id)

    # 2. 验证每个输出节点及其依赖
    for node_id in outputs:
        valid, reasons = await validate_inputs(prompt_id, prompt, node_id, validated)
        if not valid:
            errors.append((node_id, reasons))

    return (len(good_outputs) > 0, error, good_outputs, node_errors)
```

### validate_inputs()

验证单个节点的输入：

```python
async def validate_inputs(prompt_id, prompt, item, validated):
    # 1. 检查必需输入是否存在
    for x in valid_inputs:
        if x not in inputs and input_category == "required":
            error = {"type": "required_input_missing", ...}

    # 2. 检查链接类型是否匹配
    if isinstance(val, list):  # 是链接
        o_id = val[0]
        o_class_type = prompt[o_id]['class_type']
        received_type = nodes.NODE_CLASS_MAPPINGS[o_class_type].RETURN_TYPES[val[1]]
        if not validate_node_input(received_type, input_type):
            error = {"type": "return_type_mismatch", ...}

    # 3. 递归验证上游节点
    await validate_inputs(prompt_id, prompt, o_id, validated)

    return (valid, errors)
```

---

## 执行结果状态

```python
class ExecutionResult(Enum):
    SUCCESS = 0   # 执行成功
    FAILURE = 1   # 执行失败
    PENDING = 2   # 等待中（懒加载、异步任务、子图）
```

---

## 执行流程图

```
POST /prompt
     │
     ▼
validate_prompt()
     │
     ├── 验证失败 ──▶ 返回错误
     │
     ▼
PromptQueue.put()
     │
     ▼
prompt_worker 线程获取任务
     │
     ▼
PromptExecutor.execute_async()
     │
     ├── 1. 创建 DynamicPrompt
     ├── 2. 设置缓存
     ├── 3. 构建 ExecutionList (拓扑排序)
     │
     ▼
执行循环 ◀────────────────────┐
     │                        │
     ▼                        │
stage_node_execution()        │
     │                        │
     ▼                        │
execute() 单节点               │
     │                        │
     ├── SUCCESS ──▶ complete_node_execution()
     │                   │
     ├── PENDING ──▶ unstage_node_execution() ──┘
     │
     └── FAILURE ──▶ handle_execution_error()
                          │
                          ▼
                     执行结束
```

---

## IsChangedCache - 变更检测

检测节点输入是否变化，决定是否需要重新执行：

```python
class IsChangedCache:
    async def get(self, node_id):
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

        # 检查节点是否有 IS_CHANGED 方法
        if hasattr(class_def, "IS_CHANGED"):
            # 调用 IS_CHANGED 获取变更标识
            is_changed = await class_def.IS_CHANGED(**input_data)
            return is_changed

        return False  # 没有 IS_CHANGED 方法，认为没变化
```

### IS_CHANGED 用法

节点可以定义 `IS_CHANGED` 方法来控制缓存行为：

```python
class LoadImage:
    @classmethod
    def IS_CHANGED(cls, image):
        # 根据文件修改时间判断是否变化
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()
```

---

## 错误处理

```python
def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
    if isinstance(ex, comfy.model_management.InterruptProcessingException):
        # 用户中断
        self.add_message("execution_interrupted", {...})
    else:
        # 执行错误
        self.add_message("execution_error", {
            "prompt_id": prompt_id,
            "node_id": node_id,
            "exception_message": error["exception_message"],
            "exception_type": error["exception_type"],
            "traceback": error["traceback"],
        })
```

---

## WebSocket 消息流

执行过程中发送的消息：

| 消息类型 | 时机 | 数据 |
|---------|------|------|
| `execution_start` | 开始执行 | `{prompt_id}` |
| `execution_cached` | 缓存检测完成 | `{nodes: [cached_ids]}` |
| `executing` | 开始执行节点 | `{node, prompt_id}` |
| `executed` | 节点执行完成 | `{node, output, prompt_id}` |
| `execution_success` | 全部完成 | `{prompt_id}` |
| `execution_error` | 执行出错 | `{node_id, exception_message, traceback}` |
| `execution_interrupted` | 用户中断 | `{node_id}` |

---

## 关键设计总结

1. **拓扑排序执行**：保证依赖节点先执行
2. **智能缓存**：避免重复计算，支持多种策略
3. **动态子图**：支持运行时创建新节点
4. **异步执行**：支持异步节点和长时间任务
5. **中断支持**：可随时中断执行
6. **详细错误报告**：精确定位问题节点

---

## 下一步学习

- `comfy/model_management.py`：模型加载和显存管理
- `comfy_execution/caching.py`：缓存策略实现细节
- `comfy/samplers.py`：采样算法实现
