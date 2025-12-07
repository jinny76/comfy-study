# ComfyUI ControlNet 源码分析

> 源码文件: `comfy/controlnet.py` (约884行)

## 概述

ControlNet 是一种条件控制机制，允许用户通过额外的输入图像（如边缘检测、深度图、姿态等）来精确控制图像生成。ComfyUI 的 controlnet.py 实现了对各种 ControlNet 变体的统一管理。

## 核心类层次结构

```
ControlBase (基类)
├── ControlNet         # 标准 ControlNet
│   ├── ControlLora    # 使用 LoRA 权重的 ControlNet
│   └── ControlNetSD35 # SD3.5 专用 ControlNet
└── T2IAdapter         # T2I-Adapter (另一种控制方式)
```

## 1. ControlBase 基类

### 核心属性

```python
class ControlBase:
    def __init__(self):
        self.cond_hint_original = None  # 原始控制图像
        self.cond_hint = None           # 处理后的控制图像
        self.strength = 1.0             # 控制强度 (0-1)
        self.timestep_percent_range = (0.0, 1.0)  # 生效的时间步范围
        self.latent_format = None       # Latent格式
        self.vae = None                 # VAE (某些控制需要)
        self.global_average_pooling = False  # 全局平均池化
        self.timestep_range = None      # 计算后的时间步范围
        self.compression_ratio = 8      # 压缩比
        self.upscale_algorithm = 'nearest-exact'  # 缩放算法
        self.extra_args = {}            # 额外参数
        self.previous_controlnet = None # 链式ControlNet
        self.extra_conds = []           # 额外条件
        self.strength_type = StrengthType.CONSTANT  # 强度类型
```

### 强度类型

```python
class StrengthType(Enum):
    CONSTANT = 1    # 恒定强度
    LINEAR_UP = 2   # 线性增加
```

### 关键方法

```python
def set_cond_hint(self, cond_hint, strength=1.0, timestep_percent_range=(0.0, 1.0), vae=None, extra_concat=[]):
    """设置控制提示图像"""
    self.cond_hint_original = cond_hint  # 保存原始图像
    self.strength = strength
    self.timestep_percent_range = timestep_percent_range
    if vae is not None:
        self.vae = vae
    self.extra_concat_orig = extra_concat
    return self

def pre_run(self, model, percent_to_timestep_function):
    """运行前准备，计算时间步范围"""
    self.timestep_range = (
        percent_to_timestep_function(self.timestep_percent_range[0]),
        percent_to_timestep_function(self.timestep_percent_range[1])
    )
    # 递归调用前一个ControlNet
    if self.previous_controlnet is not None:
        self.previous_controlnet.pre_run(model, percent_to_timestep_function)
```

### 控制信号合并

```python
def control_merge(self, control, control_prev, dtype):
    """合并当前控制和前一个ControlNet的控制"""
    out = {'input': [], 'middle': [], 'output': []}

    for key in control:
        control_output = control[key]
        applied_to = set()
        for i in range(len(control_output)):
            x = control_output[i]
            if x is not None:
                # 应用强度
                if self.strength_type == StrengthType.CONSTANT:
                    x *= self.strength
                elif self.strength_type == StrengthType.LINEAR_UP:
                    x *= (self.strength ** float(len(googontrol_output) - i))
                # ...
                out[key].append(x)

    # 与前一个控制合并
    if control_prev is not None:
        for key in control_prev:
            # 将两个控制信号相加
            prev = control_prev[key]
            for i in range(len(prev)):
                if out[key][i] is None:
                    out[key][i] = prev[i]
                elif prev[i] is not None:
                    out[key][i] += prev[i]
    return out
```

## 2. ControlNet 标准实现

### 类定义

```python
class ControlNet(ControlBase):
    def __init__(self, control_model=None, global_average_pooling=False,
                 compression_ratio=8, latent_format=None, load_device=None,
                 manual_cast_dtype=None, extra_conds=["y"], strength_type=StrengthType.CONSTANT):
        super().__init__()
        self.control_model = control_model       # ControlNet模型
        self.load_device = load_device           # 加载设备
        self.control_model_wrapped = ModelPatcher(...)  # 包装为Patcher
```

### get_control 核心方法

```python
def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
    """获取控制信号"""

    # 1. 链式处理前一个ControlNet
    control_prev = None
    if self.previous_controlnet is not None:
        control_prev = self.previous_controlnet.get_control(...)

    # 2. 检查时间步是否在范围内
    if self.timestep_range is not None:
        if t[0] > self.timestep_range[0] or t[0] < self.timestep_range[1]:
            return control_prev  # 不在范围内，跳过此控制

    # 3. 处理控制提示图像
    if self.cond_hint is None or 尺寸不匹配:
        # 缩放到合适尺寸
        self.cond_hint = common_upscale(
            self.cond_hint_original,
            width, height,
            self.upscale_algorithm,
            "center"
        ).to(dtype).to(device)

    # 4. 处理批次广播
    if x_noisy.shape[0] != self.cond_hint.shape[0]:
        self.cond_hint = broadcast_image_to(self.cond_hint, x_noisy.shape[0], batched_number)

    # 5. 准备额外条件
    if self.extra_concat_orig:
        # 处理ControlNet++ Union等需要的额外输入
        image = torch.cat(extra_concat, dim=1)

    # 6. 运行ControlNet模型
    context = cond.get('crossattn_controlnet', cond['c_crossattn'])
    y = cond.get('y', None)  # 向量条件

    control = self.control_model(
        x=x_noisy.to(dtype),
        hint=self.cond_hint,
        timesteps=timestep.float(),
        context=context.to(dtype),
        y=y
    )

    # 7. 合并控制信号
    return self.control_merge(control, control_prev, output_dtype)
```

### 控制信号结构

ControlNet 输出的控制信号结构：

```python
control = {
    'input': [tensor1, tensor2, ...],   # 输入层控制
    'middle': [tensor],                  # 中间层控制
    'output': [tensor1, tensor2, ...]   # 输出层控制
}
```

这些信号会在 UNet 的对应层注入。

## 3. ControlLora 实现

ControlLora 使用 LoRA 权重实现 ControlNet，大大减少模型体积。

### 自定义操作

```python
class ControlLoraOps:
    """自定义Linear/Conv2d，支持LoRA权重"""

    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, bias=True, ...):
            self.weight = None           # 主权重 (可能为None)
            self.bias = None
            self.up = None               # LoRA up矩阵
            self.down = None             # LoRA down矩阵

        def forward(self, input):
            weight, bias = comfy.ops.cast_bias_weight(self, input)
            if self.up is not None:
                # 应用LoRA: weight + up @ down
                return torch.nn.functional.linear(
                    input,
                    weight + (self.up.reshape(...) @ self.down.reshape(...)).reshape(weight.shape),
                    bias
                )
            else:
                return torch.nn.functional.linear(input, weight, bias)
```

### ControlLora 类

```python
class ControlLora(ControlNet):
    def __init__(self, control_weights, global_average_pooling=False, ...):
        ControlBase.__init__(self)
        self.control_weights = control_weights  # LoRA权重字典

    def pre_run(self, model, percent_to_timestep_function):
        """运行前加载LoRA权重到模型"""
        # 从基础模型复制结构
        controlnet_config = model.model_config.unet_config.copy()
        controlnet_config["operations"] = ControlLoraOps  # 使用自定义操作

        # 创建ControlNet模型
        self.control_model = control_net_class(...)

        # 应用LoRA权重
        for k in self.control_weights:
            layer = model_sd[k[:-len(".weight")]]
            layer.up = self.control_weights[k + ".up"]
            layer.down = self.control_weights[k + ".down"]
```

## 4. T2I-Adapter

T2I-Adapter 是另一种轻量级控制方式，模型更小但效果略弱。

```python
class T2IAdapter(ControlBase):
    def __init__(self, t2i_model, channels_in, compression_ratio, upscale_algorithm, device=None):
        self.t2i_model = t2i_model      # T2I适配器模型
        self.channels_in = channels_in   # 输入通道数
        self.control_input = None        # 缓存的控制输入
        self.compression_ratio = compression_ratio

    def get_control(self, x_noisy, t, cond, batched_number, transformer_options):
        # 只需运行一次，结果可缓存
        if self.control_input is None:
            self.t2i_model.to(device)
            self.control_input = self.t2i_model(self.cond_hint)
            self.t2i_model.cpu()  # 用完移回CPU

        # 克隆控制输入
        control_input = {k: [a.clone() for a in v] for k, v in self.control_input.items()}
        return self.control_merge(control_input, control_prev, dtype)
```

## 5. 模型加载系统

### 自动检测和加载

```python
def load_controlnet_state_dict(state_dict, model=None, model_options={}):
    """自动检测并加载ControlNet"""

    # 检测模型类型的key特征
    if "controlnet_cond_embedding.conv_in.weight" in state_dict:
        # 标准 ControlNet 格式
        return load_controlnet(state_dict, ...)

    if "controlnet_blocks.0.weight" in state_dict:
        # Flux ControlNet (XLabs)
        return load_controlnet_flux_xlabs_mistoline(state_dict, ...)

    if "controlnet_x_embedder.weight" in state_dict:
        # Flux ControlNet (InstantX)
        return load_controlnet_flux_instantx(state_dict, ...)

    if "input_hint_block.0.weight" in state_dict:
        # SD 1.5/2.x ControlNet
        return load_controlnet_old(state_dict, ...)

    if "lora_controlnet" in state_dict:
        # ControlLora
        return load_controlnet_lora(state_dict, ...)

    # ... 更多类型检测
```

### 支持的 ControlNet 类型

| 类型 | 检测Key | 加载函数 |
|------|---------|----------|
| 标准 ControlNet | `controlnet_cond_embedding.conv_in.weight` | `load_controlnet` |
| SD 1.5/2.x | `input_hint_block.0.weight` | `load_controlnet_old` |
| SD3 MMDiT | `controlnet_blocks.0.weight` (+ pos_embed) | `load_controlnet_mmdit` |
| SD3.5 | (特殊检测) | `load_controlnet_sd35` |
| HunyuanDiT | `input_block.0.weight` | `load_controlnet_hunyuandit` |
| Flux XLabs | `controlnet_blocks.0.weight` | `load_controlnet_flux_xlabs_mistoline` |
| Flux InstantX | `controlnet_x_embedder.weight` | `load_controlnet_flux_instantx` |
| ControlLora | `lora_controlnet` | `ControlLora` |
| T2I-Adapter | (多种特征) | `load_t2i_adapter` |

## 6. 辅助函数

### 图像广播

```python
def broadcast_image_to(tensor, target_batch_size, batched_number):
    """将图像广播到目标批次大小"""
    current_batch_size = tensor.shape[0]
    per_batch = target_batch_size // batched_number

    # 重复到匹配批次
    tensor = tensor.repeat(per_batch, 1, 1, 1)[:per_batch]
    return tensor.repeat(batched_number, 1, 1, 1)
```

### 模型选项处理

```python
def load_controlnet(ckpt_path, model=None, model_options={}):
    """从文件加载ControlNet"""
    model_options = model_options.copy()

    # 自动检测shuffle模型
    filename = os.path.splitext(ckpt_path)[0]
    if filename.endswith("_shuffle"):
        model_options["global_average_pooling"] = True

    # 加载并返回
    return load_controlnet_state_dict(load_torch_file(ckpt_path), ...)
```

## 7. 数据流图

```
用户输入图像 (Canny/Depth/Pose/...)
        │
        ▼
┌─────────────────────────────────────────────┐
│           set_cond_hint()                   │
│  保存原始图像, 设置strength, timestep_range  │
└─────────────────────────────────────────────┘
        │
        ▼ (采样时调用)
┌─────────────────────────────────────────────┐
│             get_control()                   │
│  1. 检查时间步是否在范围内                    │
│  2. 缩放/处理控制图像                        │
│  3. 运行ControlNet模型                       │
│  4. 应用强度和合并                           │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│  control = {                                │
│    'input': [t1, t2, ...],                  │
│    'middle': [t],                           │
│    'output': [t1, t2, ...]                  │
│  }                                          │
└─────────────────────────────────────────────┘
        │
        ▼ (注入到UNet)
┌─────────────────────────────────────────────┐
│  UNet Layer + Control Signal                │
│  h = h + control[layer_type][i]             │
└─────────────────────────────────────────────┘
```

## 8. 链式 ControlNet

多个 ControlNet 可以链式组合：

```python
# 设置链式结构
controlnet2.previous_controlnet = controlnet1

# 在 get_control 中会递归调用
def get_control(self, ...):
    # 先获取前一个ControlNet的控制
    control_prev = None
    if self.previous_controlnet is not None:
        control_prev = self.previous_controlnet.get_control(...)

    # 计算当前控制
    control = self.control_model(...)

    # 合并两者
    return self.control_merge(control, control_prev, ...)
```

合并时，两个控制信号会**相加**：

```python
if out[key][i] is None:
    out[key][i] = prev[i]
elif prev[i] is not None:
    out[key][i] += prev[i]  # 信号相加
```

## 9. 与 UNet 的集成

ControlNet 的控制信号在采样过程中注入 UNet：

```python
# 在 samplers.py 的 calc_cond_batch 中
control = cond.get('control', None)
if control is not None:
    # 获取控制信号
    ctrl = control.get_control(x, timestep, cond, ...)
    # 传递给模型
    transformer_options['control'] = ctrl

# 在 UNet forward 中
if 'control' in transformer_options:
    control = transformer_options['control']
    for i, layer in enumerate(self.input_blocks):
        h = layer(h, ...)
        if control['input'][i] is not None:
            h = h + control['input'][i]  # 注入控制
```

## 10. 内存优化

### 设备管理

```python
def get_control(self, ...):
    # ControlNet 模型按需加载到GPU
    control_model_to_device()

    # 运行后可以卸载
    # (由 model_management 管理)
```

### 缓存控制输入

T2I-Adapter 会缓存控制输入：

```python
if self.control_input is None:
    self.t2i_model.to(device)
    self.control_input = self.t2i_model(self.cond_hint)
    self.t2i_model.cpu()  # 立即移回CPU

# 后续采样步直接使用缓存
control_input = clone(self.control_input)
```

## 总结

ControlNet 模块的设计要点：

1. **统一的基类**: `ControlBase` 提供通用的控制接口
2. **灵活的强度控制**: 支持恒定和线性增加两种强度模式
3. **时间步范围**: 可指定 ControlNet 生效的去噪阶段
4. **链式组合**: 多个 ControlNet 可以串联使用
5. **自动检测**: 根据权重key自动识别模型类型
6. **多模型支持**: 支持 SD1.5/2.x、SDXL、SD3、Flux 等
7. **轻量变体**: ControlLora 和 T2I-Adapter 提供更小的模型选项
