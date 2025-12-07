# ComfyUI Hook 系统源码分析

> 源码文件: `comfy/hooks.py` (约785行)

## 概述

Hook 系统是 ComfyUI 中一个强大的扩展机制，允许条件 (conds) 在采样过程中动态影响模型行为，而无需在核心代码中为每种特殊情况编写硬编码逻辑。

**设计目标**（源码注释）:
> The purpose of hooks is to allow conds to influence sampling without the need for ComfyUI core code to make explicit special cases like it does for ControlNet and GLIGEN.

主要用途：
- **AnimateDiff**: 在不同帧使用不同的权重
- **IP-Adapter**: 动态图像提示注入
- **LoRA 调度**: 在采样过程中切换或调整 LoRA 强度
- **区域控制**: 对图像不同区域应用不同的条件

## 核心枚举类型

### Hook 类型

```python
class EnumHookType(enum.Enum):
    Weight = "weight"              # 权重修改（LoRA等）
    ObjectPatch = "object_patch"   # 对象补丁（未实现）
    AdditionalModels = "add_models"  # 额外模型加载
    TransformerOptions = "transformer_options"  # Transformer选项
    Injections = "add_injections"  # 注入（未实现）
```

### Hook 模式

```python
class EnumHookMode(enum.Enum):
    MinVram = "minvram"   # 最小显存：不缓存hook权重
    MaxSpeed = "maxspeed" # 最大速度：缓存hook权重以加速切换
```

### Hook 作用域

```python
class EnumHookScope(enum.Enum):
    AllConditioning = "all_conditioning"  # 影响所有条件
    HookedOnly = "hooked_only"            # 只影响附加了hook的条件
```

### 权重目标

```python
class EnumWeightTarget(enum.Enum):
    Model = "model"  # UNet/DiT 模型
    Clip = "clip"    # CLIP 文本编码器
```

## 核心类层次结构

```
Hook (基类)
├── WeightHook           # 权重修改（LoRA等）
├── ObjectPatchHook      # 对象补丁（未实现）
├── AdditionalModelsHook # 额外模型管理
├── TransformerOptionsHook # Transformer选项注入
└── InjectionsHook       # 注入（未实现）

HookGroup                # Hook 集合管理
HookKeyframe             # 单个关键帧
HookKeyframeGroup        # 关键帧组（调度）
```

## 1. Hook 基类

```python
class Hook:
    def __init__(self, hook_type: EnumHookType=None, hook_ref: _HookRef=None,
                 hook_id: str=None, hook_keyframe: HookKeyframeGroup=None,
                 hook_scope=EnumHookScope.AllConditioning):
        self.hook_type = hook_type           # Hook类型
        self.hook_ref = hook_ref             # 引用（用于克隆共享）
        self.hook_id = hook_id               # 可选ID
        self.hook_keyframe = hook_keyframe   # 关键帧调度
        self.hook_scope = hook_scope         # 作用域
        self.custom_should_register = default_should_register  # 自定义注册条件

    @property
    def strength(self):
        """从关键帧获取当前强度"""
        return self.hook_keyframe.strength

    def initialize_timesteps(self, model: BaseModel):
        """初始化时间步，将百分比转换为sigma"""
        self.reset()
        self.hook_keyframe.initialize_timesteps(model)

    def should_register(self, model, model_options, target_dict, registered):
        """决定是否应该注册此hook"""
        return self.custom_should_register(self, model, model_options, target_dict, registered)

    def add_hook_patches(self, model, model_options, target_dict, registered):
        """子类必须实现：添加hook补丁"""
        raise NotImplementedError
```

## 2. WeightHook - 权重Hook

最常用的 Hook 类型，用于动态应用 LoRA 等权重修改。

```python
class WeightHook(Hook):
    def __init__(self, strength_model=1.0, strength_clip=1.0):
        super().__init__(hook_type=EnumHookType.Weight,
                        hook_scope=EnumHookScope.HookedOnly)
        self.weights: dict = None        # 模型权重
        self.weights_clip: dict = None   # CLIP权重
        self.need_weight_init = True     # 是否需要初始化
        self._strength_model = strength_model
        self._strength_clip = strength_clip

    @property
    def strength_model(self):
        """模型强度 = 基础强度 × 关键帧强度"""
        return self._strength_model * self.strength

    @property
    def strength_clip(self):
        """CLIP强度 = 基础强度 × 关键帧强度"""
        return self._strength_clip * self.strength

    def add_hook_patches(self, model, model_options, target_dict, registered):
        """添加LoRA权重到模型"""
        if not self.should_register(...):
            return False

        target = target_dict.get('target', None)
        if target == EnumWeightTarget.Clip:
            strength = self._strength_clip
        else:
            strength = self._strength_model

        if self.need_weight_init:
            # 首次使用，需要转换LoRA key
            key_map = {}
            if target == EnumWeightTarget.Clip:
                key_map = comfy.lora.model_lora_keys_clip(model.model, key_map)
            else:
                key_map = comfy.lora.model_lora_keys_unet(model.model, key_map)
            weights = comfy.lora.load_lora(self.weights, key_map, log_missing=False)
        else:
            # 已初始化，直接使用
            weights = self.weights_clip if target == EnumWeightTarget.Clip else self.weights

        # 添加到模型的hook_patches
        model.add_hook_patches(hook=self, patches=weights, strength_patch=strength)
        registered.add(self)
        return True
```

## 3. TransformerOptionsHook

用于向 transformer_options 注入自定义的 wrappers、callbacks 等。

```python
class TransformerOptionsHook(Hook):
    def __init__(self, transformers_dict: dict=None,
                 hook_scope=EnumHookScope.AllConditioning):
        super().__init__(hook_type=EnumHookType.TransformerOptions)
        self.transformers_dict = transformers_dict  # 要注入的选项
        self._skip_adding = False  # 避免重复加载

    def add_hook_patches(self, model, model_options, target_dict, registered):
        if self.hook_scope == EnumHookScope.AllConditioning:
            # 添加到 model_options，影响所有条件
            add_model_options = {
                "transformer_options": self.transformers_dict,
                "to_load_options": self.transformers_dict
            }
            self._skip_adding = True  # 避免重复
        else:
            # 只添加到 to_load_options，按需加载
            add_model_options = {"to_load_options": self.transformers_dict}

        registered.add(self)
        merge_nested_dicts(model_options, add_model_options, copy_dict1=False)
        return True

    def on_apply_hooks(self, model, transformer_options):
        """在采样时被调用，注入选项"""
        if not self._skip_adding:
            merge_nested_dicts(transformer_options, self.transformers_dict, copy_dict1=False)
```

## 4. AdditionalModelsHook

管理需要额外加载的模型（如 IP-Adapter 的图像编码器）。

```python
class AdditionalModelsHook(Hook):
    def __init__(self, models: list[ModelPatcher]=None, key: str=None):
        super().__init__(hook_type=EnumHookType.AdditionalModels)
        self.models = models  # 额外需要加载的模型列表
        self.key = key        # 标识key

    def add_hook_patches(self, model, model_options, target_dict, registered):
        if not self.should_register(...):
            return False
        registered.add(self)
        return True
```

## 5. HookGroup - Hook 集合管理

```python
class HookGroup:
    """管理一组 Hooks，支持按类型查询"""

    def __init__(self):
        self.hooks: list[Hook] = []
        self._hook_dict: dict[EnumHookType, list[Hook]] = {}

    def add(self, hook: Hook):
        """添加hook（自动去重）"""
        if hook not in self.hooks:
            self.hooks.append(hook)
            self._hook_dict.setdefault(hook.hook_type, []).append(hook)

    def get_type(self, hook_type: EnumHookType):
        """按类型获取hooks"""
        return self._hook_dict.get(hook_type, [])

    def clone(self):
        """深克隆"""
        c = HookGroup()
        for hook in self.hooks:
            c.add(hook.clone())
        return c

    def clone_and_combine(self, other: HookGroup):
        """克隆并合并另一个HookGroup"""
        c = self.clone()
        if other is not None:
            for hook in other.hooks:
                c.add(hook.clone())
        return c

    @staticmethod
    def combine_all_hooks(hooks_list: list[HookGroup], require_count=0) -> HookGroup:
        """合并多个HookGroup"""
        actual = [g for g in hooks_list if g is not None]
        if len(actual) == 0:
            return None
        if len(actual) == 1:
            return actual[0]

        final_hook = None
        for hook in actual:
            if final_hook is None:
                final_hook = hook.clone()
            else:
                final_hook = final_hook.clone_and_combine(hook)
        return final_hook
```

## 6. 关键帧系统 - 动态强度调度

### HookKeyframe

```python
class HookKeyframe:
    def __init__(self, strength: float, start_percent=0.0, guarantee_steps=1):
        self.strength = strength              # 该关键帧的强度
        self.start_percent = start_percent    # 开始百分比 (0.0-1.0)
        self.start_t = 999999999.9            # 转换后的sigma值
        self.guarantee_steps = guarantee_steps # 最少保持步数
```

### HookKeyframeGroup

```python
class HookKeyframeGroup:
    def __init__(self):
        self.keyframes: list[HookKeyframe] = []
        self._current_keyframe: HookKeyframe = None
        self._current_used_steps = 0
        self._current_index = 0

    @property
    def strength(self):
        """获取当前关键帧的强度"""
        if self._current_keyframe is not None:
            return self._current_keyframe.strength
        return 1.0  # 默认强度

    def add(self, keyframe: HookKeyframe):
        """添加关键帧并按 start_percent 排序"""
        self.keyframes.append(keyframe)
        self.keyframes = sorted(self.keyframes, key=lambda x: x.start_percent)
        self._set_first_as_current()

    def initialize_timesteps(self, model: BaseModel):
        """将百分比转换为sigma时间步"""
        for keyframe in self.keyframes:
            keyframe.start_t = model.model_sampling.percent_to_sigma(keyframe.start_percent)

    def prepare_current_keyframe(self, curr_t: float, transformer_options) -> bool:
        """根据当前时间步更新活动关键帧"""
        if self.is_empty():
            return False

        max_sigma = torch.max(transformer_options["sample_sigmas"])

        # 如果已满足保证步数，检查是否需要切换
        if self._current_used_steps >= self._current_keyframe.get_effective_guarantee_steps(max_sigma):
            # 查找下一个匹配的关键帧
            for i in range(self._current_index + 1, len(self.keyframes)):
                eval_c = self.keyframes[i]
                # sigma越大表示越早的步骤
                if eval_c.start_t >= curr_t:
                    self._current_index = i
                    self._current_keyframe = eval_c
                    self._current_used_steps = 0
                    if self._current_keyframe.get_effective_guarantee_steps(max_sigma) > 0:
                        break
                else:
                    break

        self._current_used_steps += 1
        return True  # 返回是否有变化
```

## 7. 插值方法

```python
class InterpolationMethod:
    LINEAR = "linear"         # 线性
    EASE_IN = "ease_in"       # 缓入
    EASE_OUT = "ease_out"     # 缓出
    EASE_IN_OUT = "ease_in_out"  # 缓入缓出

    @classmethod
    def get_weights(cls, num_from: float, num_to: float, length: int, method: str):
        diff = num_to - num_from
        if method == cls.LINEAR:
            weights = torch.linspace(num_from, num_to, length)
        elif method == cls.EASE_IN:
            index = torch.linspace(0, 1, length)
            weights = diff * np.power(index, 2) + num_from
        elif method == cls.EASE_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * (1 - np.power(1 - index, 2)) + num_from
        elif method == cls.EASE_IN_OUT:
            index = torch.linspace(0, 1, length)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + num_from
        return weights
```

## 8. 便捷函数

### 创建 LoRA Hook

```python
def create_hook_lora(lora: dict[str, torch.Tensor], strength_model: float, strength_clip: float):
    """从LoRA权重创建HookGroup"""
    hook_group = HookGroup()
    hook = WeightHook(strength_model=strength_model, strength_clip=strength_clip)
    hook_group.add(hook)
    hook.weights = lora
    return hook_group
```

### 将模型作为 LoRA 使用

```python
def create_hook_model_as_lora(weights_model, weights_clip, strength_model, strength_clip):
    """将完整模型权重作为LoRA使用"""
    hook_group = HookGroup()
    hook = WeightHook(strength_model=strength_model, strength_clip=strength_clip)
    hook_group.add(hook)

    # 将权重包装为 "model_as_lora" 类型
    if weights_model is not None:
        patches_model = {}
        for key in weights_model:
            patches_model[key] = ("model_as_lora", (weights_model[key],))
        hook.weights = patches_model

    hook.need_weight_init = False  # 已经是正确格式
    return hook_group
```

### 设置条件的 Hooks

```python
def set_hooks_for_conditioning(cond, hooks: HookGroup, append_hooks=True, cache=None):
    """为条件附加hooks"""
    if hooks is None:
        return cond
    return conditioning_set_values_with_hooks(cond, {'hooks': hooks}, append_hooks=append_hooks, cache=cache)
```

### 组合条件和属性

```python
def set_conds_props(conds: list, strength: float, set_cond_area: str,
                   mask=None, hooks=None, timesteps_range=None, append_hooks=True):
    """一次性设置条件的多个属性"""
    final_conds = []
    cache = {}
    for c in conds:
        # 1. 应用 hooks
        c = set_hooks_for_conditioning(c, hooks, append_hooks=append_hooks, cache=cache)
        # 2. 应用 mask
        c = set_mask_for_conditioning(cond=c, mask=mask, strength=strength, set_cond_area=set_cond_area)
        # 3. 应用时间步范围
        c = set_timesteps_for_conditioning(cond=c, timestep_range=timesteps_range)
        final_conds.append(c)
    return final_conds
```

## 9. 数据流图

```
创建 Hook
    │
    ▼
┌─────────────────────────────────────────────┐
│  WeightHook / TransformerOptionsHook / ...  │
│  设置 weights, strength, keyframes          │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│              HookGroup                      │
│  管理多个 hooks，支持按类型查询              │
└─────────────────────────────────────────────┘
    │
    ▼ set_hooks_for_conditioning()
┌─────────────────────────────────────────────┐
│            Conditioning                     │
│  cond[1]['hooks'] = HookGroup               │
└─────────────────────────────────────────────┘
    │
    ▼ 采样时
┌─────────────────────────────────────────────┐
│         hook.add_hook_patches()             │
│  将 hook 权重添加到 ModelPatcher            │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│     ModelPatcher.hook_patches[hook]         │
│  存储每个 hook 的权重补丁                    │
└─────────────────────────────────────────────┘
    │
    ▼ 每个采样步骤
┌─────────────────────────────────────────────┐
│  keyframe.prepare_current_keyframe(t)       │
│  根据当前时间步更新强度                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│        应用权重到模型计算                    │
│  strength = base_strength × keyframe.strength│
└─────────────────────────────────────────────┘
```

## 10. 与 ModelPatcher 的交互

Hook 系统与 ModelPatcher 紧密配合：

```python
# 在 ModelPatcher 中
class ModelPatcher:
    def __init__(self):
        self.hook_patches: dict[Hook, dict] = {}  # hook -> 权重补丁

    def add_hook_patches(self, hook: Hook, patches: dict, strength_patch: float):
        """添加 hook 的权重补丁"""
        # 类似于 add_patches，但按 hook 分组存储
        self.hook_patches[hook] = patches

    def apply_hook_patches(self, hooks: HookGroup):
        """在采样时应用活动 hooks 的权重"""
        for hook in hooks.get_type(EnumHookType.Weight):
            if hook in self.hook_patches:
                strength = hook.strength_model  # 动态强度
                patches = self.hook_patches[hook]
                # 应用到模型权重
                self._apply_patches(patches, strength)
```

## 11. 使用示例

### AnimateDiff 风格的帧调度

```python
# 创建关键帧组
kf_group = HookKeyframeGroup()
kf_group.add(HookKeyframe(strength=1.0, start_percent=0.0))   # 开始时全强度
kf_group.add(HookKeyframe(strength=0.5, start_percent=0.5))   # 中间减半
kf_group.add(HookKeyframe(strength=0.0, start_percent=0.8))   # 结束前淡出

# 创建 LoRA hook
hook = WeightHook(strength_model=1.0, strength_clip=1.0)
hook.weights = lora_weights
hook.hook_keyframe = kf_group

# 添加到 HookGroup
hook_group = HookGroup()
hook_group.add(hook)

# 附加到条件
cond = set_hooks_for_conditioning(cond, hook_group)
```

### 区域特定的 LoRA

```python
# 创建只影响特定区域的 hook
hook = WeightHook(strength_model=0.8)
hook.weights = style_lora
hook.hook_scope = EnumHookScope.HookedOnly  # 只影响附加了这个hook的条件

hook_group = HookGroup()
hook_group.add(hook)

# 创建带 mask 的条件
masked_cond = set_conds_props(
    conds=[cond],
    strength=1.0,
    set_cond_area="default",
    mask=region_mask,
    hooks=hook_group
)
```

## 总结

Hook 系统的核心特点：

1. **解耦设计**: 将 LoRA/IP-Adapter 等特殊处理从核心采样代码中分离
2. **动态调度**: 通过关键帧系统实现强度随时间变化
3. **作用域控制**: 可以选择影响所有条件还是仅特定条件
4. **类型分类**: 支持权重、Transformer选项、额外模型等多种Hook类型
5. **高效缓存**: MaxSpeed 模式下缓存 hook 权重以加速切换
6. **组合能力**: HookGroup 支持多个 hooks 的管理和合并

这套系统使得复杂的条件控制（如 AnimateDiff 的帧间 LoRA 切换）可以优雅地实现，而不需要修改核心采样逻辑。
