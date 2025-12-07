# ComfyUI 条件系统源码分析

> 源码文件:
> - `comfy/conds.py` (约137行) - 条件包装类
> - `comfy/sampler_helpers.py` (约185行) - 条件辅助函数

## 概述

条件 (Conditioning) 是扩散模型生成过程中的核心概念，用于引导模型生成特定内容。ComfyUI 的条件系统处理：

1. **文本条件** (Cross Attention): CLIP 编码的文本嵌入
2. **向量条件** (y/pooled): 全局向量嵌入
3. **控制条件** (ControlNet/GLIGEN): 图像控制信号
4. **区域条件** (Area/Mask): 局部区域控制
5. **时间步条件** (Timestep): 条件生效的时间范围

## 条件数据结构

ComfyUI 中的条件是一个元组列表：

```python
conditioning = [
    (
        tensor,      # cross_attn 张量 [B, seq_len, dim]
        {
            "pooled_output": tensor,  # 池化输出 [B, dim]
            "control": ControlNet,     # 可选：ControlNet
            "mask": tensor,            # 可选：区域mask
            "area": tuple,             # 可选：区域坐标
            "start_percent": float,    # 可选：开始时间
            "end_percent": float,      # 可选：结束时间
            "hooks": HookGroup,        # 可选：动态hooks
            # ...更多可选字段
        }
    ),
    # ... 可以有多个条件
]
```

## conds.py 核心类

### 1. CONDRegular - 基础条件类

```python
class CONDRegular:
    """标准条件包装器，用于批处理和拼接"""

    def __init__(self, cond):
        self.cond = cond  # 原始张量

    def _copy_with(self, cond):
        """创建新实例"""
        return self.__class__(cond)

    def process_cond(self, batch_size, **kwargs):
        """处理条件以匹配批次大小"""
        return self._copy_with(
            comfy.utils.repeat_to_batch_size(self.cond, batch_size)
        )

    def can_concat(self, other):
        """检查是否可以与另一个条件拼接"""
        if self.cond.shape != other.cond.shape:
            return False
        if self.cond.device != other.cond.device:
            logging.warning("WARNING: conds not on same device, skipping concat.")
            return False
        return True

    def concat(self, others):
        """拼接多个条件"""
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)

    def size(self):
        """返回条件尺寸"""
        return list(self.cond.size())
```

### 2. CONDNoiseShape - 噪声形状条件

```python
class CONDNoiseShape(CONDRegular):
    """支持区域裁剪的条件"""

    def process_cond(self, batch_size, area, **kwargs):
        data = self.cond
        if area is not None:
            # 根据区域裁剪条件
            dims = len(area) // 2
            for i in range(dims):
                # narrow(dim, start, length)
                data = data.narrow(i + 2, area[i + dims], area[i])

        return self._copy_with(
            comfy.utils.repeat_to_batch_size(data, batch_size)
        )
```

**用途**: 当条件需要匹配噪声的空间区域时使用（如 latent mask 或 inpainting）。

### 3. CONDCrossAttn - 交叉注意力条件

```python
class CONDCrossAttn(CONDRegular):
    """处理不同序列长度的cross attention条件"""

    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            # batch 和 embedding dim 必须相同
            if s1[0] != s2[0] or s1[2] != s2[2]:
                return False
            # 检查序列长度的最小公倍数
            mult_min = math.lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if diff > 4:  # 限制padding倍数
                return False
        return True

    def concat(self, others):
        """拼接时自动对齐序列长度"""
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]

        # 找到所有条件的最小公倍数长度
        for x in others:
            crossattn_max_len = math.lcm(crossattn_max_len, x.cond.shape[1])
            conds.append(x.cond)

        # 通过repeat对齐长度
        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                # 重复以匹配长度（不影响结果）
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1)
            out.append(c)

        return torch.cat(out)
```

**关键设计**:
- 不同长度的文本条件可以拼接（如77 token 和 154 token）
- 使用 LCM 找到公共长度
- 通过 repeat 对齐，这在数学上等价于原始条件

### 4. CONDConstant - 常量条件

```python
class CONDConstant(CONDRegular):
    """不随批次变化的常量条件"""

    def process_cond(self, batch_size, **kwargs):
        return self._copy_with(self.cond)  # 不复制到批次

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond  # 返回单个值

    def size(self):
        return [1]
```

**用途**: 某些模型需要标量或单值条件（如图像尺寸信息）。

### 5. CONDList - 列表条件

```python
class CONDList(CONDRegular):
    """处理条件张量列表"""

    def process_cond(self, batch_size, **kwargs):
        out = []
        for c in self.cond:
            out.append(comfy.utils.repeat_to_batch_size(c, batch_size))
        return self._copy_with(out)

    def can_concat(self, other):
        if len(self.cond) != len(other.cond):
            return False
        for i in range(len(self.cond)):
            if self.cond[i].shape != other.cond[i].shape:
                return False
        return True

    def concat(self, others):
        out = []
        for i in range(len(self.cond)):
            o = [self.cond[i]]
            for x in others:
                o.append(x.cond[i])
            out.append(torch.cat(o))
        return out
```

**用途**: 某些模型需要多个张量作为条件（如多层特征）。

## sampler_helpers.py 辅助函数

### 1. convert_cond - 转换条件格式

```python
def convert_cond(cond):
    """将条件元组转换为字典格式"""
    out = []
    for c in cond:
        temp = c[1].copy()          # 复制属性字典
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            temp["cross_attn"] = c[0]  # 添加cross attention
        temp["model_conds"] = model_conds
        temp["uuid"] = uuid.uuid4()    # 唯一标识符
        out.append(temp)
    return out
```

**转换前后**:
```python
# 原始格式
[(tensor, {"pooled_output": ..., "control": ...}), ...]

# 转换后
[{"cross_attn": tensor, "pooled_output": ..., "control": ..., "uuid": ...}, ...]
```

### 2. get_models_from_cond - 提取模型

```python
def get_models_from_cond(cond, model_type):
    """从条件中提取指定类型的模型"""
    models = []
    for c in cond:
        if model_type in c:
            if isinstance(c[model_type], list):
                models += c[model_type]
            else:
                models += [c[model_type]]
    return models

# 使用示例
control_nets = get_models_from_cond(cond, "control")
gligen_models = get_models_from_cond(cond, "gligen")
```

### 3. get_hooks_from_cond - 提取 Hooks

```python
def get_hooks_from_cond(cond, full_hooks: HookGroup):
    """从条件中收集所有hooks"""
    cnets = []
    for c in cond:
        # 从条件直接获取hooks
        if 'hooks' in c:
            for hook in c['hooks'].hooks:
                full_hooks.add(hook)
        # 收集ControlNet
        if 'control' in c:
            cnets.append(c['control'])

    # 从ControlNet链中获取extra_hooks
    def get_extra_hooks_from_cnet(cnet, _list):
        if cnet.extra_hooks is not None:
            _list.append(cnet.extra_hooks)
        if cnet.previous_controlnet is None:
            return _list
        return get_extra_hooks_from_cnet(cnet.previous_controlnet, _list)

    hooks_list = []
    for base_cnet in set(cnets):
        get_extra_hooks_from_cnet(base_cnet, hooks_list)

    extra_hooks = HookGroup.combine_all_hooks(hooks_list)
    if extra_hooks is not None:
        for hook in extra_hooks.hooks:
            full_hooks.add(hook)

    return full_hooks
```

### 4. get_additional_models - 获取额外模型

```python
def get_additional_models(conds, dtype):
    """获取条件中需要加载的额外模型"""
    cnets = []
    gligen = []
    add_models = []

    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")
        add_models += get_models_from_cond(conds[k], "additional_models")

    control_nets = set(cnets)

    inference_memory = 0
    control_models = []
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen = [x[1] for x in gligen]
    models = control_models + gligen + add_models

    return models, inference_memory
```

### 5. prepare_sampling - 准备采样

```python
def prepare_sampling(model: ModelPatcher, noise_shape, conds, model_options=None):
    """准备采样所需的模型和条件"""
    # 获取额外模型
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    models += get_additional_models_from_model_options(model_options)
    models += model.get_nested_additional_models()

    # 估算内存需求
    memory_required, minimum_memory_required = estimate_memory(model, noise_shape, conds)

    # 加载模型到GPU
    comfy.model_management.load_models_gpu(
        [model] + models,
        memory_required=memory_required + inference_memory,
        minimum_memory_required=minimum_memory_required + inference_memory
    )

    real_model = model.model
    return real_model, conds, models
```

### 6. prepare_model_patcher - 注册 Hooks

```python
def prepare_model_patcher(model: ModelPatcher, conds, model_options: dict):
    """从条件中注册hooks到模型"""
    # 收集所有hooks
    hooks = HookGroup()
    for k in conds:
        get_hooks_from_cond(conds[k], hooks)

    # 添加ModelPatcher的wrappers和callbacks
    merge_nested_dicts(
        model_options["transformer_options"].setdefault("wrappers", {}),
        model.wrappers
    )
    merge_nested_dicts(
        model_options["transformer_options"].setdefault("callbacks", {}),
        model.callbacks
    )

    # 注册hooks
    registered = HookGroup()
    target_dict = create_target_dict(EnumWeightTarget.Model)

    # 按类型处理hooks
    for hook in hooks.get_type(EnumHookType.TransformerOptions):
        hook.add_hook_patches(model, model_options, target_dict, registered)

    for hook in hooks.get_type(EnumHookType.AdditionalModels):
        hook.add_hook_patches(model, model_options, target_dict, registered)

    # 注册权重hooks
    model.register_all_hook_patches(hooks, target_dict, model_options, registered)

    if len(registered) > 0:
        model_options["registered_hooks"] = registered

    return model_options.setdefault("to_load_options", {})
```

## 条件类型映射

在 `model_base.py` 的 `extra_conds()` 中定义了条件类型映射：

```python
def extra_conds(self, **kwargs):
    out = {}

    # Cross Attention (文本嵌入)
    cross_attn = kwargs.get("cross_attn", None)
    if cross_attn is not None:
        out['c_crossattn'] = CONDCrossAttn(cross_attn)

    # 池化输出 (向量条件)
    pooled = kwargs.get("pooled_output", None)
    if pooled is not None:
        out['y'] = CONDRegular(pooled)

    # 噪声增强 (某些模型需要)
    noise_aug = kwargs.get("noise_augmentation", None)
    if noise_aug is not None:
        out['c_noise_aug'] = CONDRegular(noise_aug)

    return out
```

## 数据流图

```
用户输入 (Prompt)
      │
      ▼ CLIP Encode
┌─────────────────────────────────────────────┐
│  conditioning = [                           │
│    (cross_attn_tensor,                      │
│     {"pooled_output": ..., ...})            │
│  ]                                          │
└─────────────────────────────────────────────┘
      │
      ▼ convert_cond()
┌─────────────────────────────────────────────┐
│  cond_dict = [                              │
│    {"cross_attn": tensor,                   │
│     "pooled_output": ...,                   │
│     "uuid": uuid4()}                        │
│  ]                                          │
└─────────────────────────────────────────────┘
      │
      ▼ model.extra_conds()
┌─────────────────────────────────────────────┐
│  model_conds = {                            │
│    "c_crossattn": CONDCrossAttn(tensor),   │
│    "y": CONDRegular(pooled),               │
│  }                                          │
└─────────────────────────────────────────────┘
      │
      ▼ process_cond(batch_size)
┌─────────────────────────────────────────────┐
│  扩展到匹配批次大小                          │
│  处理区域裁剪 (如果有area)                   │
└─────────────────────────────────────────────┘
      │
      ▼ 传入模型
┌─────────────────────────────────────────────┐
│  model.forward(                             │
│    x,                                       │
│    timestep,                                │
│    c_crossattn=...,                         │
│    y=...,                                   │
│  )                                          │
└─────────────────────────────────────────────┘
```

## 条件拼接 (Batching)

当同时处理多个条件时，系统会尝试拼接以提高效率：

```python
# 检查是否可以拼接
if cond1.can_concat(cond2):
    # 可以拼接，一次推理多个条件
    batched_cond = cond1.concat([cond2])
    output = model(batched_cond)
else:
    # 无法拼接，分别推理
    output1 = model(cond1)
    output2 = model(cond2)
```

**拼接条件**:
- 相同形状（或可对齐的序列长度）
- 相同设备
- LCM 长度差不超过 4 倍

## 区域条件 (Area)

```python
# 区域格式: (height, width, y_offset, x_offset)
area = (256, 256, 0, 128)  # 从(0,128)开始的256x256区域

# CONDNoiseShape 会裁剪条件以匹配区域
def process_cond(self, batch_size, area, **kwargs):
    data = self.cond
    if area is not None:
        dims = len(area) // 2  # 2 for 2D
        for i in range(dims):
            # narrow 裁剪指定维度
            data = data.narrow(i + 2, area[i + dims], area[i])
    return self._copy_with(repeat_to_batch_size(data, batch_size))
```

## 时间步条件

条件可以指定生效的时间范围：

```python
cond = {
    "cross_attn": tensor,
    "start_percent": 0.0,   # 从开始
    "end_percent": 0.5,     # 到50%
}

# 在采样时检查
if timestep_percent >= start_percent and timestep_percent <= end_percent:
    # 应用此条件
    ...
```

## 总结

条件系统的核心设计：

1. **灵活的包装类**: 不同类型的条件有专门的包装器处理
2. **智能拼接**: 自动对齐和拼接兼容的条件
3. **区域支持**: 可以为图像不同区域应用不同条件
4. **时间调度**: 条件可以在特定的采样阶段生效
5. **Hook集成**: 与Hook系统紧密配合实现动态条件

这套系统使得 ComfyUI 能够灵活处理各种复杂的条件组合场景。
