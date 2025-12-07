# ComfyUI 模型修补系统 - model_patcher.py 深度解析

## 概述

`comfy/model_patcher.py` 是 ComfyUI 动态模型修补的核心模块，实现了：
- **LoRA/LoHa/LoKR** 等权重适配器的动态应用
- **低显存模式** 下的即时权重计算
- **模型克隆** 和补丁共享机制
- **Hook 系统** 用于动态权重切换
- **部分加载/卸载** 实现显存精细管理

文件约 1346 行，是理解 ComfyUI 高效模型管理的关键。

## 核心架构

```
ModelPatcher
├── patches: dict              # LoRA 等权重补丁
├── backup: dict               # 原始权重备份
├── object_patches: dict       # 对象级补丁（如 model_sampling）
├── hook_patches: dict         # Hook 动态补丁
├── model_options: dict        # transformer_options 等配置
├── callbacks: dict            # 生命周期回调
├── wrappers: dict             # 采样包装器
└── injections: dict           # 注入补丁
```

## 一、ModelPatcher 类初始化 (line 211)

```python
class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        self.model = model                    # 被包装的模型
        self.load_device = load_device        # GPU
        self.offload_device = offload_device  # CPU

        # 权重补丁系统
        self.patches = {}                     # key -> [(strength_patch, weights, strength_model, offset, function), ...]
        self.backup = {}                      # 原始权重备份
        self.object_patches = {}              # 对象级补丁
        self.weight_wrapper_patches = {}      # 权重包装函数

        # 模型选项
        self.model_options = {"transformer_options": {}}

        # Hook 系统
        self.hook_patches = {}                # hook_ref -> {key: patches}
        self.hook_backup = {}                 # 原始权重备份
        self.cached_hook_patches = {}         # 缓存的 hook 权重
        self.current_hooks = None             # 当前激活的 hooks

        # 补丁 UUID（用于判断是否需要重新应用）
        self.patches_uuid = uuid.uuid4()
```

## 二、克隆机制 (line 283)

ModelPatcher 支持高效克隆，多个克隆共享底层模型：

```python
def clone(self):
    # 创建新的 ModelPatcher，共享同一个 model
    n = self.__class__(self.model, self.load_device, self.offload_device, ...)

    # 浅拷贝补丁列表（列表本身是新的，但补丁对象共享）
    n.patches = {}
    for k in self.patches:
        n.patches[k] = self.patches[k][:]  # 列表浅拷贝

    n.patches_uuid = self.patches_uuid     # 共享 UUID
    n.parent = self                        # 记录父级
    n.backup = self.backup                 # 共享备份

    # 深拷贝 model_options
    n.model_options = copy.deepcopy(self.model_options)

    return n

def clone_has_same_weights(self, clone: 'ModelPatcher'):
    """检查克隆是否有相同权重（优化：避免重复应用补丁）"""
    if not self.is_clone(clone):
        return False
    if self.patches_uuid == clone.patches_uuid:
        return True
    return False
```

**克隆的用途：**
- 同一模型应用不同 LoRA 强度
- ControlNet 等需要独立配置但共享基础模型
- 避免重复加载大模型到显存

## 三、补丁添加 (line 550)

### 3.1 add_patches - 添加 LoRA 等补丁

```python
def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
    """
    添加权重补丁（如 LoRA）

    Args:
        patches: dict, {key: weight_data}
        strength_patch: LoRA 强度
        strength_model: 原模型强度（通常为 1.0）
    """
    with self.use_ejected():
        p = set()
        model_sd = self.model.state_dict()

        for k in patches:
            # 支持 offset 和 function
            if isinstance(k, str):
                key = k
                offset = None
                function = None
            else:
                key = k[0]
                offset = k[1]
                function = k[2] if len(k) > 2 else None

            if key in model_sd:
                p.add(k)
                current_patches = self.patches.get(key, [])
                # 补丁格式：(strength_patch, weights, strength_model, offset, function)
                current_patches.append((strength_patch, patches[k], strength_model, offset, function))
                self.patches[key] = current_patches

        # 更新 UUID，表示补丁已更改
        self.patches_uuid = uuid.uuid4()
        return list(p)
```

### 3.2 补丁存储格式

```python
# patches[key] 是一个列表，每个元素是一个元组：
(
    strength_patch,    # float: LoRA 强度 (如 0.8)
    weights,           # tuple/WeightAdapter: 权重数据
    strength_model,    # float: 原模型强度 (通常 1.0)
    offset,           # tuple/None: 部分权重修改 (dim, start, length)
    function          # callable/None: 自定义变换函数
)
```

## 四、权重计算 (lora.py:368)

### 4.1 calculate_weight 核心函数

```python
def calculate_weight(patches, weight, key, intermediate_dtype=torch.float32, original_weights=None):
    """
    计算应用补丁后的权重

    Args:
        patches: 补丁列表
        weight: 原始权重
        key: 权重键名
        intermediate_dtype: 中间计算精度
    """
    for p in patches:
        strength = p[0]
        v = p[1]
        strength_model = p[2]
        offset = p[3]
        function = p[4]

        if function is None:
            function = lambda a: a

        # 处理 offset（部分权重修改）
        old_weight = None
        if offset is not None:
            old_weight = weight
            weight = weight.narrow(offset[0], offset[1], offset[2])

        # 应用模型强度
        if strength_model != 1.0:
            weight *= strength_model

        # 处理 WeightAdapter（LoRA/LoHa/LoKR 等）
        if isinstance(v, weight_adapter.WeightAdapterBase):
            output = v.calculate_weight(weight, key, strength, ...)
            if output is not None:
                weight = output
            continue

        # 处理简单补丁类型
        if len(v) == 1:
            patch_type = "diff"
        else:
            patch_type = v[0]
            v = v[1]

        if patch_type == "diff":
            # 差值补丁：weight += strength * diff
            diff = v[0]
            if strength != 0.0 and diff.shape == weight.shape:
                weight += function(strength * diff.to(weight.device, weight.dtype))

        elif patch_type == "set":
            # 直接设置权重
            weight.copy_(v[0])

        elif patch_type == "model_as_lora":
            # 使用另一个模型的权重差作为 LoRA
            target_weight = v[0]
            diff_weight = target_weight - original_weights[key][0][0]
            weight += function(strength * diff_weight.to(weight.dtype))

    return weight
```

### 4.2 支持的补丁类型

| 类型 | 格式 | 说明 |
|------|------|------|
| `diff` | `("diff", (tensor,))` | 差值：`W += strength * diff` |
| `set` | `("set", (tensor,))` | 直接设置权重 |
| `model_as_lora` | `("model_as_lora", (tensor,))` | 用模型差作为 LoRA |
| `WeightAdapter` | `LoRAAdapter/LoHaAdapter/...` | 各种权重适配器 |

## 五、模型加载与补丁应用 (line 674)

### 5.1 load 方法 - 智能加载

```python
def load(self, device_to=None, lowvram_model_memory=0, force_patch_weights=False, full_load=False):
    """
    加载模型到设备，智能处理低显存模式
    """
    with self.use_ejected():
        self.unpatch_hooks()
        loading = self._load_list()  # 获取所有需要加载的模块

        load_completely = []  # 完整加载到 GPU
        offloaded = []        # 保留在 CPU，即时计算

        for module_offload_mem, module_mem, n, m, params in loading:
            weight_key = "{}.weight".format(n)
            bias_key = "{}.bias".format(n)

            # 判断是否需要低显存模式
            lowvram_fits = mem_counter + module_mem < lowvram_model_memory

            if not full_load and hasattr(m, "comfy_cast_weights"):
                if not lowvram_fits:
                    # 低显存模式：设置即时计算函数
                    if weight_key in self.patches:
                        m.weight_function = [LowVramPatch(weight_key, self.patches)]
                    if bias_key in self.patches:
                        m.bias_function = [LowVramPatch(bias_key, self.patches)]
                    offloaded.append((module_mem, n, m, params))
                else:
                    # 正常模式：预计算并加载
                    load_completely.append((module_mem, n, m, params))

        # 完整加载的模块：预先计算补丁
        for x in load_completely:
            for param in params:
                key = "{}.{}".format(n, param)
                self.patch_weight_to_device(key, device_to=device_to)
            m.to(device_to)
```

### 5.2 LowVramPatch - 低显存即时计算

```python
class LowVramPatch:
    """低显存模式下的即时补丁计算"""

    def __init__(self, key, patches, convert_func=None, set_func=None):
        self.key = key
        self.patches = patches

    def __call__(self, weight):
        """每次前向传播时动态计算"""
        return comfy.lora.calculate_weight(
            self.patches[self.key],
            weight,
            self.key,
            intermediate_dtype=weight.dtype
        )
```

**工作原理：**
- 低显存模式下，模块的 `weight_function` 被设置为 `LowVramPatch`
- 每次前向传播时，模块会调用 `weight_function` 获取实际权重
- 权重在 CPU 计算后传输到 GPU，使用后丢弃

### 5.3 patch_weight_to_device - 预计算补丁

```python
def patch_weight_to_device(self, key, device_to=None, inplace_update=False):
    """预先计算补丁并应用到模型"""
    if key not in self.patches:
        return

    weight, set_func, convert_func = get_key_weight(self.model, key)

    # 备份原始权重
    if key not in self.backup:
        self.backup[key] = namedtuple('Dimension', ['weight', 'inplace_update'])(
            weight.to(device=self.offload_device, copy=inplace_update),
            inplace_update
        )

    # 计算新权重
    temp_weight = weight.to(torch.float32, copy=True)
    if convert_func is not None:
        temp_weight = convert_func(temp_weight, inplace=True)

    out_weight = comfy.lora.calculate_weight(self.patches[key], temp_weight, key)

    # 应用到模型
    out_weight = comfy.float.stochastic_rounding(out_weight, weight.dtype)
    comfy.utils.set_attr_param(self.model, key, out_weight)
```

## 六、对象级补丁 (line 466)

除了权重补丁，还支持对象级补丁：

```python
def add_object_patch(self, name, obj):
    """添加对象级补丁"""
    self.object_patches[name] = obj

# 示例用法
def set_model_compute_dtype(self, dtype):
    """设置模型计算精度"""
    self.add_object_patch("manual_cast_dtype", dtype)
    self.force_cast_weights = True
```

**常用对象补丁：**
- `model_sampling`: 采样方式（v-prediction 等）
- `manual_cast_dtype`: 强制计算精度
- 自定义模块替换

## 七、Transformer 选项系统 (line 405-463)

ModelPatcher 提供了丰富的 transformer 层修补接口：

```python
# 注意力补丁
def set_model_attn1_patch(self, patch):
    """Self-attention 补丁"""
    self.set_model_patch(patch, "attn1_patch")

def set_model_attn2_patch(self, patch):
    """Cross-attention 补丁"""
    self.set_model_patch(patch, "attn2_patch")

# 注意力替换
def set_model_attn1_replace(self, patch, block_name, number, transformer_index=None):
    """替换特定层的 self-attention"""
    self.set_model_patch_replace(patch, "attn1", block_name, number, transformer_index)

# 块级补丁
def set_model_input_block_patch(self, patch):
    """输入块补丁"""
    self.set_model_patch(patch, "input_block_patch")

def set_model_output_block_patch(self, patch):
    """输出块补丁"""
    self.set_model_patch(patch, "output_block_patch")

# CFG 相关
def set_model_sampler_cfg_function(self, sampler_cfg_function):
    """自定义 CFG 函数"""
    self.model_options["sampler_cfg_function"] = sampler_cfg_function

def set_model_sampler_post_cfg_function(self, post_cfg_function):
    """CFG 后处理"""
    self.model_options["sampler_post_cfg_function"] = [post_cfg_function]
```

**transformer_options 结构：**
```python
{
    "patches": {
        "attn1_patch": [patch1, patch2, ...],
        "attn2_patch": [...],
        "input_block_patch": [...],
        ...
    },
    "patches_replace": {
        "attn1": {
            ("input", 0): patch,
            ("output", 5, 1): patch,  # (block_name, number, transformer_index)
        }
    }
}
```

## 八、Hook 系统 (line 1126-1341)

Hook 系统用于动态切换权重（如 AnimateDiff、IP-Adapter）：

### 8.1 注册 Hook 补丁

```python
def register_all_hook_patches(self, hooks: HookGroup, target_dict: dict, ...):
    """注册所有 hook 补丁"""
    self.restore_hook_patches()

    weight_hooks_to_register = []
    for hook in hooks.get_type(EnumHookType.Weight):
        if hook.hook_ref not in self.hook_patches:
            weight_hooks_to_register.append(hook)

    if len(weight_hooks_to_register) > 0:
        # 备份当前 hook_patches
        self.hook_patches_backup = create_hook_patches_clone(self.hook_patches)
        for hook in weight_hooks_to_register:
            hook.add_hook_patches(self, model_options, target_dict, registered)
```

### 8.2 应用 Hook

```python
def apply_hooks(self, hooks: HookGroup, transformer_options=None, force_apply=False):
    """应用 hooks 到模型"""
    if self.current_hooks == hooks and not force_apply:
        return  # 已经是当前 hooks，无需重新应用

    self.patch_hooks(hooks=hooks)
    return create_transformer_options_from_hooks(self, hooks, transformer_options)

def patch_hooks(self, hooks: HookGroup):
    """实际应用 hook 权重"""
    with self.use_ejected():
        if hooks is not None:
            # 尝试使用缓存
            cached_weights = self.cached_hook_patches.get(hooks, None)
            if cached_weights is not None:
                for key in cached_weights:
                    self.patch_cached_hook_weights(cached_weights, key, memory_counter)
            else:
                # 计算并应用
                self.unpatch_hooks()
                relevant_patches = self.get_combined_hook_patches(hooks)
                for key in relevant_patches:
                    self.patch_hook_weight_to_device(hooks, relevant_patches, key, ...)
        else:
            self.unpatch_hooks()

        self.current_hooks = hooks
```

### 8.3 Hook 模式

```python
class EnumHookMode(Enum):
    MaxSpeed = "maxspeed"  # 缓存计算后的权重，占用更多显存
    MinVram = "minvram"    # 不缓存，每次重新计算
```

## 九、部分加载/卸载 (line 859-970)

### 9.1 partially_unload - 释放显存

```python
def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
    """部分卸载模型以释放显存"""
    with self.use_ejected():
        unload_list = self._load_list()
        unload_list.sort()  # 从小模块开始卸载

        for module_offload_mem, module_mem, n, m, params in unload_list:
            if memory_freed >= memory_to_free:
                break

            # 恢复原始权重
            for param in params:
                key = "{}.{}".format(n, param)
                bk = self.backup.get(key, None)
                if bk is not None:
                    comfy.utils.set_attr_param(self.model, key, bk.weight)
                    self.backup.pop(key)

            # 设置低显存模式
            if weight_key in self.patches:
                m.weight_function = [LowVramPatch(weight_key, self.patches)]

            m.to(device_to)  # 移到 CPU
            memory_freed += module_mem
```

### 9.2 partially_load - 增量加载

```python
def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
    """增量加载模型到 GPU"""
    with self.use_ejected(skip_and_inject_on_exit_only=True):
        # 检查是否需要重新应用补丁
        unpatch_weights = self.model.current_weight_patches_uuid != self.patches_uuid

        if unpatch_weights:
            self.unpatch_model(self.offload_device, unpatch_weights=True)

        # 加载更多到 GPU
        self.load(device_to, lowvram_model_memory=current_used + extra_memory)

        return self.model.model_loaded_weight_memory - current_used
```

## 十、LoRA 键名映射 (lora.py)

### 10.1 CLIP LoRA 映射

```python
LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}

def model_lora_keys_clip(model, key_map={}):
    """建立 LoRA 键名到模型键名的映射"""
    # CLIP-L
    # lora_te1_text_model_encoder_layers_0_mlp_fc1 -> clip_l.transformer.text_model.encoder.layers.0.mlp.fc1.weight

    # CLIP-G (SDXL)
    # lora_te2_text_model_encoder_layers_0_mlp_fc1 -> clip_g.transformer.text_model.encoder.layers.0.mlp.fc1.weight

    # T5 (SD3/Flux)
    # lora_te3_encoder_block_0_layer_0_SelfAttention_q -> t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.q.weight
```

### 10.2 UNet/DiT LoRA 映射

```python
def model_lora_keys_unet(model, key_map={}):
    """建立 UNet LoRA 键名映射"""
    # 标准格式
    # lora_unet_input_blocks_0_0_weight -> diffusion_model.input_blocks.0.0.weight

    # Diffusers 格式
    # down_blocks.0.attentions.0.processor.to_q -> diffusion_model.input_blocks...

    # SD3/Flux (MMDiT)
    # transformer.joint_blocks.0.x_block.attn.qkv -> diffusion_model.joint_blocks...
```

## 十一、生命周期回调 (line 995-1014)

ModelPatcher 提供完整的生命周期回调：

```python
class CallbacksMP:
    ON_CLONE = "on_clone"              # 克隆时
    ON_LOAD = "on_load"                # 加载到 GPU 时
    ON_DETACH = "on_detach"            # 卸载时
    ON_CLEANUP = "on_cleanup"          # 清理时
    ON_PRE_RUN = "on_pre_run"          # 推理前
    ON_PREPARE_STATE = "on_prepare_state"  # 准备状态时
    ON_APPLY_HOOKS = "on_apply_hooks"  # 应用 hooks 时
    ON_INJECT_MODEL = "on_inject_model"    # 注入时
    ON_EJECT_MODEL = "on_eject_model"      # 弹出时
    ON_REGISTER_ALL_HOOK_PATCHES = "on_register_all_hook_patches"

# 使用示例
def add_callback(self, call_type: str, callback: Callable):
    self.callbacks.setdefault(call_type, {}).setdefault(None, []).append(callback)
```

## 十二、工作流程图

```
LoRA 加载流程:
┌─────────────────────────────────────────────────────────────┐
│  1. load_lora_for_models(model, clip, lora, strength)      │
│     ├─ model_lora_keys_unet() 建立键名映射                  │
│     ├─ model_lora_keys_clip() 建立键名映射                  │
│     └─ load_lora() 加载 LoRA 权重                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. model.clone() 克隆 ModelPatcher                         │
│     └─ new_patcher.add_patches(loaded, strength)           │
│        ├─ 遍历 patches，匹配 model.state_dict()             │
│        └─ 存储到 self.patches[key]                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 推理时 patch_model() / load()                           │
│     ├─ 判断显存是否足够                                      │
│     ├─ 足够: patch_weight_to_device() 预计算                │
│     │   ├─ 备份原始权重到 self.backup                        │
│     │   ├─ calculate_weight() 计算新权重                    │
│     │   └─ set_attr_param() 应用到模型                      │
│     └─ 不足: 设置 LowVramPatch 即时计算                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 推理完成 unpatch_model()                                │
│     ├─ 从 self.backup 恢复原始权重                          │
│     └─ 清理 LowVramPatch                                   │
└─────────────────────────────────────────────────────────────┘
```

## 总结

model_patcher.py 实现了 ComfyUI 灵活高效的模型修补系统：

1. **高效克隆**: 多个配置共享底层模型，只存储差异
2. **动态补丁**: 支持 LoRA、LoHa、LoKR 等多种适配器
3. **智能显存**: 根据显存自动选择预计算或即时计算
4. **Hook 系统**: 支持动态权重切换（AnimateDiff 等）
5. **细粒度控制**: 部分加载/卸载实现精细显存管理
6. **扩展性**: 回调和包装器系统支持第三方扩展
