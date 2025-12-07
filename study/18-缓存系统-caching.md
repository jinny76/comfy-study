# ComfyUI 学习笔记 18：缓存系统

## 概述

ComfyUI 有三层缓存机制：

| 层级 | 文件 | 作用 |
|------|------|------|
| 执行缓存 | `comfy_execution/caching.py` | 节点输出缓存，避免重复计算 |
| HTTP 缓存 | `middleware/cache_middleware.py` | 静态资源浏览器缓存 |
| 采样缓存 | `comfy_extras/nodes_easycache.py` | 采样步骤缓存，跳过相似步骤 |

---

## 执行缓存

### 核心概念

当工作流执行时，每个节点的输出会被缓存。下次执行相同输入时，直接返回缓存结果。

```
┌─────────────────────────────────────────────────────────┐
│                    CacheSet                              │
├─────────────────────────────────────────────────────────┤
│  outputs: HierarchicalCache  ← 节点输出结果缓存          │
│  objects: HierarchicalCache  ← 节点实例对象缓存          │
└─────────────────────────────────────────────────────────┘
```

### 缓存键生成

两种策略：

**1. CacheKeySetID（简单键）**
```python
# caching.py:66
class CacheKeySetID(CacheKeySet):
    async def add_keys(self, node_ids):
        for node_id in node_ids:
            node = self.dynprompt.get_node(node_id)
            # 键 = (节点ID, 类型)
            self.keys[node_id] = (node_id, node["class_type"])
```

**2. CacheKeySetInputSignature（输入签名键）**
```python
# caching.py:81
class CacheKeySetInputSignature(CacheKeySet):
    async def get_node_signature(self, dynprompt, node_id):
        signature = []
        # 获取所有祖先节点（按输入顺序）
        ancestors, order_mapping = self.get_ordered_ancestry(dynprompt, node_id)

        # 当前节点签名
        signature.append(await self.get_immediate_node_signature(dynprompt, node_id, order_mapping))

        # 所有祖先节点签名
        for ancestor_id in ancestors:
            signature.append(await self.get_immediate_node_signature(dynprompt, ancestor_id, order_mapping))

        return to_hashable(signature)  # 转为可哈希的 frozenset
```

**节点签名内容：**
```python
# caching.py:108
async def get_immediate_node_signature(self, dynprompt, node_id, ancestor_order_mapping):
    node = dynprompt.get_node(node_id)
    class_type = node["class_type"]

    signature = [
        class_type,                              # 节点类型
        await self.is_changed_cache.get(node_id) # IS_CHANGED 返回值
    ]

    # 非幂等节点需要包含 ID
    if include_unique_id_in_input(class_type) or hasattr(class_def, "NOT_IDEMPOTENT"):
        signature.append(node_id)

    # 输入值
    for key in sorted(inputs.keys()):
        if is_link(inputs[key]):
            # 链接：记录祖先索引和输出槽
            (ancestor_id, ancestor_socket) = inputs[key]
            ancestor_index = ancestor_order_mapping[ancestor_id]
            signature.append((key, ("ANCESTOR", ancestor_index, ancestor_socket)))
        else:
            # 常量值直接记录
            signature.append((key, inputs[key]))

    return signature
```

### 缓存类型

```python
# execution.py:97
class CacheType(Enum):
    CLASSIC = 0       # 经典模式：执行后清理未使用缓存
    LRU = 1           # LRU 模式：保留最近使用的 N 个
    RAM_PRESSURE = 2  # 内存压力模式：根据可用 RAM 清理
    NONE = 3          # 禁用缓存
```

### CacheSet 初始化

```python
# execution.py:104
class CacheSet:
    def __init__(self, cache_type=None, cache_args={}):
        if cache_type == CacheType.NONE:
            self.init_null_cache()
        elif cache_type == CacheType.RAM_PRESSURE:
            cache_ram = cache_args.get("ram", 16.0)
            self.init_ram_cache(cache_ram)
        elif cache_type == CacheType.LRU:
            cache_size = cache_args.get("lru", 0)
            self.init_lru_cache(cache_size)
        else:
            self.init_classic_cache()

    def init_classic_cache(self):
        # 使用输入签名作为键
        self.outputs = HierarchicalCache(CacheKeySetInputSignature)
        # 节点对象用 ID 作为键
        self.objects = HierarchicalCache(CacheKeySetID)
```

---

## 缓存实现类

### BasicCache

基础缓存，提供核心 get/set 功能：

```python
# caching.py:149
class BasicCache:
    def __init__(self, key_class):
        self.key_class = key_class
        self.cache = {}           # 数据缓存
        self.subcaches = {}       # 子图缓存

    def _set_immediate(self, node_id, value):
        cache_key = self.cache_key_set.get_data_key(node_id)
        self.cache[cache_key] = value

    def _get_immediate(self, node_id):
        cache_key = self.cache_key_set.get_data_key(node_id)
        return self.cache.get(cache_key, None)

    def clean_unused(self):
        """清理当前工作流未使用的缓存"""
        preserve_keys = set(self.cache_key_set.get_used_keys())
        to_remove = [key for key in self.cache if key not in preserve_keys]
        for key in to_remove:
            del self.cache[key]
```

### HierarchicalCache

分层缓存，支持子图（Group Node）：

```python
# caching.py:238
class HierarchicalCache(BasicCache):
    def _get_cache_for(self, node_id):
        """获取节点所属的缓存层"""
        parent_id = self.dynprompt.get_parent_node_id(node_id)
        if parent_id is None:
            return self  # 顶层节点

        # 遍历父节点链，找到对应的子缓存
        hierarchy = []
        while parent_id is not None:
            hierarchy.append(parent_id)
            parent_id = self.dynprompt.get_parent_node_id(parent_id)

        cache = self
        for parent_id in reversed(hierarchy):
            cache = cache._get_subcache(parent_id)
            if cache is None:
                return None
        return cache

    def get(self, node_id):
        cache = self._get_cache_for(node_id)
        return cache._get_immediate(node_id) if cache else None

    def set(self, node_id, value):
        cache = self._get_cache_for(node_id)
        cache._set_immediate(node_id, value)
```

### LRUCache

LRU 淘汰策略缓存：

```python
# caching.py:299
class LRUCache(BasicCache):
    def __init__(self, key_class, max_size=100):
        super().__init__(key_class)
        self.max_size = max_size
        self.generation = 0           # 当前代数
        self.min_generation = 0       # 最小保留代数
        self.used_generation = {}     # 每个键的最后使用代数

    async def set_prompt(self, dynprompt, node_ids, is_changed_cache):
        await super().set_prompt(dynprompt, node_ids, is_changed_cache)
        self.generation += 1
        for node_id in node_ids:
            self._mark_used(node_id)

    def clean_unused(self):
        """超过 max_size 时，淘汰最老的条目"""
        while len(self.cache) > self.max_size and self.min_generation < self.generation:
            self.min_generation += 1
            to_remove = [key for key in self.cache
                        if self.used_generation[key] < self.min_generation]
            for key in to_remove:
                del self.cache[key]
                del self.used_generation[key]

    def _mark_used(self, node_id):
        """标记节点被使用"""
        cache_key = self.cache_key_set.get_data_key(node_id)
        if cache_key is not None:
            self.used_generation[cache_key] = self.generation
```

### RAMPressureCache

内存压力感知缓存：

```python
# caching.py:367
RAM_CACHE_HYSTERESIS = 1.1                    # 滞后系数
RAM_CACHE_DEFAULT_RAM_USAGE = 0.1             # 默认 RAM 使用估算（GB）
RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER = 1.3   # 旧工作流 OOM 分数乘数

class RAMPressureCache(LRUCache):
    def __init__(self, key_class):
        super().__init__(key_class, 0)  # max_size=0，不限制数量
        self.timestamps = {}

    def poll(self, ram_headroom):
        """检查并释放内存"""
        def _ram_gb():
            return psutil.virtual_memory().available / (1024**3)

        if _ram_gb() > ram_headroom:
            return

        gc.collect()
        if _ram_gb() > ram_headroom:
            return

        # 计算每个缓存项的 OOM 分数
        clean_list = []
        for key, (outputs, _) in self.cache.items():
            # 旧工作流有更高的淘汰优先级
            oom_score = RAM_CACHE_OLD_WORKFLOW_OOM_MULTIPLIER ** (self.generation - self.used_generation[key])

            # 估算 RAM 使用
            ram_usage = RAM_CACHE_DEFAULT_RAM_USAGE
            for output in outputs:
                if isinstance(output, torch.Tensor) and output.device.type == 'cpu':
                    # CPU Tensor 按 50% 计算（可能是高价值中间结果）
                    ram_usage += (output.numel() * output.element_size()) * 0.5
                elif hasattr(output, "get_ram_usage"):
                    ram_usage += output.get_ram_usage()

            oom_score *= ram_usage
            bisect.insort(clean_list, (oom_score, self.timestamps[key], key))

        # 按分数从高到低淘汰，直到内存充足
        while _ram_gb() < ram_headroom * RAM_CACHE_HYSTERESIS and clean_list:
            _, _, key = clean_list.pop()
            del self.cache[key]
            gc.collect()
```

---

## IS_CHANGED 机制

节点可以定义 `IS_CHANGED` 方法来控制缓存失效：

```python
# execution.py:48
class IsChangedCache:
    def __init__(self, prompt_id, dynprompt, outputs_cache):
        self.is_changed = {}

    async def get(self, node_id):
        if node_id in self.is_changed:
            return self.is_changed[node_id]

        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

        # 获取输入数据（不使用缓存）
        input_data_all, _, _ = get_input_data(node["inputs"], class_def, node_id, None)

        # 调用 IS_CHANGED
        if hasattr(class_def, 'IS_CHANGED'):
            is_changed = map_node_over_list(
                class_def, input_data_all, "IS_CHANGED"
            )
        else:
            is_changed = False

        self.is_changed[node_id] = is_changed
        return is_changed
```

**示例：LoadImage 节点**
```python
class LoadImage:
    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()  # 文件内容变化则缓存失效
```

---

## 执行流程中的缓存

```python
# execution.py:409
async def execute(server, dynprompt, caches, current_item, ...):
    unique_id = current_item

    # 1. 检查缓存
    cached = caches.outputs.get(unique_id)
    if cached is not None:
        # 命中缓存，直接返回
        server.send_sync("executed", {"node": unique_id, "output": cached.ui})
        execution_list.cache_update(unique_id, cached)
        return (ExecutionResult.SUCCESS, None, None)

    # 2. 获取或创建节点实例
    obj = caches.objects.get(unique_id)
    if obj is None:
        obj = class_def()
        caches.objects.set(unique_id, obj)

    # 3. 执行节点
    output_data, output_ui = execute_node(obj, input_data)

    # 4. 存入缓存
    cache_entry = CacheEntry(ui=output_ui, outputs=output_data)
    caches.outputs.set(unique_id, cache_entry)

    # 5. 检查内存压力（仅 RAM_PRESSURE 模式）
    caches.outputs.poll(ram_headroom=cache_args["ram"])
```

---

## HTTP 缓存

静态资源的浏览器缓存：

```python
# middleware/cache_middleware.py:22
@web.middleware
async def cache_control(request, handler):
    response = await handler(request)

    # JS/CSS：不缓存（可能频繁更新）
    if request.path.endswith(".js") or request.path.endswith(".css"):
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    # 图片：缓存 1 天
    if request.path.lower().endswith(IMG_EXTENSIONS):
        if response.status in (200, 201, ...):
            response.headers.setdefault("Cache-Control", f"public, max-age={ONE_DAY}")
        elif response.status == 404:
            response.headers.setdefault("Cache-Control", f"public, max-age={ONE_HOUR}")

    return response
```

---

## 采样缓存（EasyCache）

采样过程中跳过相似步骤，加速生成：

### 原理

```
步骤 1: input_1 → model → output_1
步骤 2: input_2 → model → output_2  (input_2 ≈ input_1 → 跳过，用 output_1)
步骤 3: input_3 → model → output_3  (变化超阈值 → 正常执行)
```

### EasyCacheHolder

```python
# nodes_easycache.py:175
class EasyCacheHolder:
    def __init__(self, reuse_threshold, start_percent, end_percent, ...):
        self.reuse_threshold = reuse_threshold     # 复用阈值
        self.start_percent = start_percent         # 开始百分比
        self.end_percent = end_percent             # 结束百分比

        # 状态追踪
        self.relative_transformation_rate = None   # 输入→输出变化率
        self.cumulative_change_rate = 0.0          # 累积变化率
        self.x_prev_subsampled = None              # 上次输入（降采样）
        self.output_prev_subsampled = None         # 上次输出（降采样）
        self.uuid_cache_diffs = {}                 # 缓存差值
```

### 跳过逻辑

```python
# nodes_easycache.py:12
def easycache_forward_wrapper(executor, *args, **kwargs):
    easycache = transformer_options["easycache"]
    x = args[0][:, :easycache.output_channels]

    if easycache.should_do_easycache(sigmas):
        # 计算输入变化
        if easycache.has_x_prev_subsampled():
            input_change = (easycache.subsample(x) - easycache.x_prev_subsampled).abs().mean()

            # 估算输出变化率
            approx_output_change_rate = (easycache.relative_transformation_rate * input_change) / easycache.output_prev_norm
            easycache.cumulative_change_rate += approx_output_change_rate

            # 低于阈值则跳过
            if easycache.cumulative_change_rate < easycache.reuse_threshold:
                easycache.skip_current_step = True
                return easycache.apply_cache_diff(x, uuids)  # 用缓存值
            else:
                easycache.cumulative_change_rate = 0.0  # 重置

    # 正常执行
    output = executor(*args, **kwargs)

    # 更新缓存
    easycache.update_cache_diff(output, x, uuids)
    easycache.x_prev_subsampled = easycache.subsample(x)
    easycache.output_prev_subsampled = easycache.subsample(output)

    return output
```

### 使用节点

```python
# EasyCache 节点
class EasyCacheNode(io.ComfyNode):
    @classmethod
    def execute(cls, model, reuse_threshold, start_percent, end_percent, verbose):
        model = model.clone()
        model.model_options["transformer_options"]["easycache"] = EasyCacheHolder(
            reuse_threshold, start_percent, end_percent,
            subsample_factor=8, offload_cache_diff=False
        )
        # 注册 wrapper
        model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "easycache", easycache_sample_wrapper)
        model.add_wrapper_with_key(WrappersMP.CALC_COND_BATCH, "easycache", easycache_calc_cond_batch_wrapper)
        model.add_wrapper_with_key(WrappersMP.DIFFUSION_MODEL, "easycache", easycache_forward_wrapper)
        return io.NodeOutput(model)
```

**参数说明：**
| 参数 | 默认 | 说明 |
|------|------|------|
| `reuse_threshold` | 0.2 | 复用阈值，越大跳过越多 |
| `start_percent` | 0.15 | 开始生效的采样进度 |
| `end_percent` | 0.95 | 结束生效的采样进度 |

---

## 缓存配置

### 命令行参数

```bash
# 禁用缓存
python main.py --disable-smart-memory

# LRU 缓存（保留 100 个）
python main.py --cache-lru 100

# RAM 压力缓存（保持 16GB 空闲）
python main.py --cache-classic --lowvram
```

### PromptExecutor 初始化

```python
# execution.py:618
class PromptExecutor:
    def __init__(self, server, cache_type=False, cache_args=None):
        self.cache_type = cache_type
        self.cache_args = cache_args
        self.caches = CacheSet(cache_type=self.cache_type, cache_args=self.cache_args)
```

---

## 总结

| 缓存类型 | 用途 | 特点 |
|----------|------|------|
| **Classic** | 默认 | 每次执行后清理未使用缓存 |
| **LRU** | 长期复用 | 保留最近 N 个结果 |
| **RAM Pressure** | 大工作流 | 根据内存压力动态淘汰 |
| **EasyCache** | 加速采样 | 跳过相似采样步骤 |
| **HTTP Cache** | 前端资源 | 浏览器缓存控制 |

核心机制：
1. 输入签名哈希 → 相同输入必得相同输出
2. IS_CHANGED → 节点自定义缓存失效条件
3. 分层缓存 → 支持 Group Node 子图
4. 采样缓存 → 利用相邻步骤相似性加速
