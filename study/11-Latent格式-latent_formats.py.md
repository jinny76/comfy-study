# ComfyUI Latent 格式系统源码分析

> 源码文件: `comfy/latent_formats.py` (约760行)

## 概述

Latent 格式定义了不同扩散模型的潜在空间特性。每个模型家族（SD1.5、SDXL、SD3、Flux、视频模型等）使用不同的 VAE 编码器，产生不同维度、通道数和数值范围的 latent 表示。

这个模块的核心作用：
1. **归一化处理**: 将 latent 缩放到模型期望的数值范围
2. **预览生成**: 提供 latent → RGB 的近似转换矩阵
3. **维度信息**: 定义 latent 的通道数和维度（2D图像/3D视频/1D音频）

## 基类 LatentFormat

```python
class LatentFormat:
    scale_factor = 1.0              # 缩放因子
    latent_channels = 4             # latent 通道数
    latent_dimensions = 2           # 维度：2=图像, 3=视频, 1=音频
    latent_rgb_factors = None       # latent→RGB 转换矩阵
    latent_rgb_factors_bias = None  # RGB 偏置
    taesd_decoder_name = None       # TAESD 解码器名称

    def process_in(self, latent):
        """输入处理：编码后的 latent → 模型输入"""
        return latent * self.scale_factor

    def process_out(self, latent):
        """输出处理：模型输出 → 解码前的 latent"""
        return latent / self.scale_factor
```

## 支持的模型格式一览

### 图像模型

| 模型 | 通道数 | 缩放因子 | 特殊处理 |
|------|--------|----------|----------|
| SD 1.5 | 4 | 0.18215 | 标准 |
| SDXL | 4 | 0.13025 | 带偏置 |
| SDXL Playground 2.5 | 4 | 0.5 | 均值/标准差归一化 |
| SD3 | 16 | 1.5305 | shift_factor |
| Flux | 16 | 0.3611 | shift_factor |
| Flux2 | 128 | 1.0 | 特殊 reshape |
| HunyuanImage21 | 64 | 0.75289 | - |

### 视频模型

| 模型 | 通道数 | 维度 | 特殊处理 |
|------|--------|------|----------|
| Mochi | 12 | 3D | 均值/标准差归一化 |
| LTXV | 128 | 3D | - |
| HunyuanVideo | 16 | 3D | 标准缩放 |
| HunyuanVideo1.5 | 32 | 3D | - |
| Cosmos | 16 | 3D | - |
| Wan2.1 | 16 | 3D | 均值/标准差归一化 |
| Wan2.2 | 48 | 3D | 均值/标准差归一化 |

### 音频/3D模型

| 模型 | 通道数 | 维度 |
|------|--------|------|
| StableAudio1 | 64 | 1D |
| ACEAudio | 8 | 2D |
| Hunyuan3D v2 | 64 | 1D |

## 详细分析

### 1. SD 1.5 格式

```python
class SD15(LatentFormat):
    def __init__(self, scale_factor=0.18215):
        self.scale_factor = scale_factor
        self.latent_rgb_factors = [
            #   R        G        B
            [ 0.3512,  0.2297,  0.3227],  # 通道0
            [ 0.3250,  0.4974,  0.2350],  # 通道1
            [-0.2829,  0.1762,  0.2721],  # 通道2
            [-0.2120, -0.2616, -0.7177]   # 通道3
        ]
        self.taesd_decoder_name = "taesd_decoder"
```

**关键点**:
- `scale_factor = 0.18215`: VAE 编码后的 latent 需要乘以此值
- 4 通道 latent，每个通道对 RGB 的贡献不同
- `latent_rgb_factors` 是一个 4×3 矩阵，用于快速预览

### 2. SDXL 格式

```python
class SDXL(LatentFormat):
    scale_factor = 0.13025

    def __init__(self):
        self.latent_rgb_factors = [
            [ 0.3651,  0.4232,  0.4341],
            [-0.2533, -0.0042,  0.1068],
            [ 0.1076,  0.1111, -0.0362],
            [-0.3165, -0.2492, -0.2188]
        ]
        self.latent_rgb_factors_bias = [0.1084, -0.0175, -0.0011]
        self.taesd_decoder_name = "taesdxl_decoder"
```

**与 SD1.5 的区别**:
- 更小的 `scale_factor`（0.13025 vs 0.18215）
- 添加了 `latent_rgb_factors_bias` 用于更准确的预览
- 不同的 TAESD 解码器

### 3. SD3 / Flux 格式 (16通道)

```python
class SD3(LatentFormat):
    latent_channels = 16  # 16通道！

    def __init__(self):
        self.scale_factor = 1.5305
        self.shift_factor = 0.0609  # 新增偏移因子
        self.latent_rgb_factors = [
            # 16行×3列的矩阵
            [-0.0922, -0.0175,  0.0749],
            # ... 共16行
        ]

    def process_in(self, latent):
        return (latent - self.shift_factor) * self.scale_factor

    def process_out(self, latent):
        return (latent / self.scale_factor) + self.shift_factor
```

**新架构特点**:
- 16 通道 latent（4倍于 SD1.5）
- 引入 `shift_factor` 进行居中处理
- 不再是简单的乘除，而是先偏移再缩放

### 4. Flux 格式

```python
class Flux(SD3):  # 继承自 SD3
    latent_channels = 16

    def __init__(self):
        self.scale_factor = 0.3611   # 不同的缩放因子
        self.shift_factor = 0.1159   # 不同的偏移
        self.taesd_decoder_name = "taef1_decoder"
```

### 5. Flux2 格式 (128通道)

```python
class Flux2(LatentFormat):
    latent_channels = 128  # 超高通道数！

    def __init__(self):
        self.latent_rgb_factors = [
            # 32行×3列 (不是128行，因为有特殊 reshape)
            [0.0058, 0.0113, 0.0073],
            # ...
        ]
        # 特殊的 reshape 函数
        self.latent_rgb_factors_reshape = lambda t: t.reshape(
            t.shape[0], 32, 2, 2, t.shape[-2], t.shape[-1]
        ).permute(0, 1, 4, 2, 5, 3).reshape(
            t.shape[0], 32, t.shape[-2] * 2, t.shape[-1] * 2
        )

    def process_in(self, latent):
        return latent  # 无缩放

    def process_out(self, latent):
        return latent
```

**Flux2 特点**:
- 128 通道的超高维 latent
- 需要特殊的 reshape 操作来生成预览
- 无缩放处理

### 6. 视频模型：均值/标准差归一化

```python
class Mochi(LatentFormat):
    latent_channels = 12
    latent_dimensions = 3  # 3D: [batch, channels, frames, height, width]

    def __init__(self):
        self.scale_factor = 1.0
        # 每个通道的均值
        self.latents_mean = torch.tensor([
            -0.0673, -0.0380, -0.0748, -0.0557, 0.0128, -0.0470,
            0.0439, -0.0935, -0.0992, -0.0087, -0.0119, -0.0322
        ]).view(1, 12, 1, 1, 1)
        # 每个通道的标准差
        self.latents_std = torch.tensor([
            0.9264, 0.9249, 0.9393, 0.9593, 0.8245, 0.9173,
            0.9294, 1.3721, 0.8814, 0.9168, 0.9185, 0.9275
        ]).view(1, 12, 1, 1, 1)

    def process_in(self, latent):
        """标准化: (x - mean) / std"""
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        """反标准化: x * std + mean"""
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean
```

### 7. HunyuanImage21Refiner (特殊帧处理)

```python
class HunyuanImage21Refiner(LatentFormat):
    latent_channels = 64
    latent_dimensions = 3
    scale_factor = 1.03682

    def process_in(self, latent):
        out = latent * self.scale_factor
        # 复制第一帧并添加到开头
        out = torch.cat((out[:, :, :1], out), dim=2)
        # 重新排列维度
        out = out.permute(0, 2, 1, 3, 4)
        b, f_times_2, c, h, w = out.shape
        # 将相邻帧合并到通道维度
        out = out.reshape(b, f_times_2 // 2, 2 * c, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        return out

    def process_out(self, latent):
        # 逆操作...
```

## Latent → RGB 预览

`latent_rgb_factors` 用于快速预览 latent 内容，无需完整解码：

```python
def latent_preview(latent, latent_format):
    """将 latent 近似转换为 RGB 预览"""
    factors = torch.tensor(latent_format.latent_rgb_factors)
    bias = latent_format.latent_rgb_factors_bias or [0, 0, 0]

    # latent: [B, C, H, W]
    # factors: [C, 3]
    # 矩阵乘法: [B, C, H, W] × [C, 3] → [B, 3, H, W]
    rgb = torch.einsum('bchw,crgb->b rgb hw', latent, factors)
    rgb = rgb + torch.tensor(bias).view(1, 3, 1, 1)

    return rgb
```

## 维度说明

```python
latent_dimensions = 1  # 1D: 音频 [B, C, T]
latent_dimensions = 2  # 2D: 图像 [B, C, H, W]
latent_dimensions = 3  # 3D: 视频 [B, C, F, H, W]
```

## 数据流图

```
原始图像/视频
      │
      ▼ VAE Encoder
┌─────────────────────────────────────────────┐
│  Latent (原始)                              │
│  例如: torch.Size([1, 4, 64, 64])           │
└─────────────────────────────────────────────┘
      │
      ▼ process_in()
┌─────────────────────────────────────────────┐
│  Latent (归一化)                            │
│  SD1.5: latent * 0.18215                   │
│  SD3:   (latent - 0.0609) * 1.5305         │
│  Mochi: (latent - mean) / std               │
└─────────────────────────────────────────────┘
      │
      ▼ 扩散模型采样
┌─────────────────────────────────────────────┐
│  Latent (去噪后)                            │
└─────────────────────────────────────────────┘
      │
      ▼ process_out()
┌─────────────────────────────────────────────┐
│  Latent (反归一化)                          │
│  SD1.5: latent / 0.18215                   │
│  SD3:   latent / 1.5305 + 0.0609           │
└─────────────────────────────────────────────┘
      │
      ▼ VAE Decoder
┌─────────────────────────────────────────────┐
│  输出图像/视频                               │
└─────────────────────────────────────────────┘
```

## TAESD 解码器

TAESD (Tiny AutoEncoder for Stable Diffusion) 是轻量级解码器，用于实时预览：

```python
taesd_decoder_name = "taesd_decoder"      # SD 1.5
taesd_decoder_name = "taesdxl_decoder"    # SDXL
taesd_decoder_name = "taesd3_decoder"     # SD3
taesd_decoder_name = "taef1_decoder"      # Flux
taesd_decoder_name = "taehv"              # HunyuanVideo
taesd_decoder_name = "lighttaew2_1"       # Wan 2.1
```

## 模型格式对比表

| 模型 | 通道 | scale | shift | 归一化方式 | 维度 |
|------|------|-------|-------|------------|------|
| SD 1.5 | 4 | 0.182 | - | 简单缩放 | 2D |
| SDXL | 4 | 0.130 | - | 简单缩放 | 2D |
| SD3 | 16 | 1.531 | 0.061 | 偏移+缩放 | 2D |
| Flux | 16 | 0.361 | 0.116 | 偏移+缩放 | 2D |
| Flux2 | 128 | 1.0 | - | 无缩放 | 2D |
| Mochi | 12 | 1.0 | - | 均值/标准差 | 3D |
| LTXV | 128 | 1.0 | - | 无缩放 | 3D |
| HunyuanVideo | 16 | 0.477 | - | 简单缩放 | 3D |
| Wan 2.1 | 16 | 1.0 | - | 均值/标准差 | 3D |
| Wan 2.2 | 48 | 1.0 | - | 均值/标准差 | 3D |

## 使用示例

```python
# 加载模型时获取 latent_format
latent_format = model.get_model_object("latent_format")

# 编码图像
latent = vae.encode(image)
latent = latent_format.process_in(latent)

# 采样
for step in sampling_steps:
    latent = denoise_step(latent)

# 解码
latent = latent_format.process_out(latent)
image = vae.decode(latent)
```

## 总结

Latent 格式系统的核心功能：

1. **数值归一化**: 确保不同 VAE 编码的 latent 在模型期望的数值范围内
2. **预览支持**: 通过 `latent_rgb_factors` 实现快速预览
3. **多模态支持**: 支持 2D 图像、3D 视频、1D 音频
4. **向后兼容**: 从 4 通道 SD1.5 到 128 通道 Flux2 都有支持
5. **模块化设计**: 每个模型类型只需继承并覆盖必要的参数

这套系统使得 ComfyUI 能够透明地支持各种不同架构的扩散模型，用户无需关心底层的 latent 处理细节。
