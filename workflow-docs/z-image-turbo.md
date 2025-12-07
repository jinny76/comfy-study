# Z-Image-Turbo 文生图工作流

> 文件: `image_z_image_turbo.json`

## 概述

Z-Image-Turbo 是一个高效的文生图模型，基于单流 Diffusion Transformer 架构，支持中英文提示词，生成速度快。

## 工作流结构图

```
┌─────────────────┐
│   UNETLoader    │──┐
│ (加载主模型)     │  │
└─────────────────┘  │
                     ▼
┌─────────────────┐  ┌────────────────────┐
│   CLIPLoader    │  │ ModelSamplingAura  │
│ (加载文本编码器) │  │    Flow (shift)    │
└────────┬────────┘  └─────────┬──────────┘
         │                     │
         ▼                     │
┌─────────────────┐            │
│ CLIPTextEncode  │            │
│   (提示词编码)   │            │
└────────┬────────┘            │
         │                     │
         ├──────────┐          │
         │          ▼          │
         │  ┌───────────────┐  │
         │  │ConditionZero │  │
         │  │   (负面条件)   │  │
         │  └───────┬───────┘  │
         │          │          │
         ▼          ▼          ▼
       ┌─────────────────────────┐
       │       KSampler          │◄── EmptySD3LatentImage
       │      (采样生成)          │     (空白画布)
       └───────────┬─────────────┘
                   │
                   ▼
       ┌─────────────────┐
       │    VAEDecode    │◄── VAELoader
       │   (解码为图片)   │     (加载VAE)
       └────────┬────────┘
                │
                ▼
       ┌─────────────────┐
       │    SaveImage    │
       │    (保存图片)    │
       └─────────────────┘
```

## 节点详解

### 1. UNETLoader (加载主模型)

加载 Z-Image-Turbo 的主扩散模型。

| 参数 | 值 | 说明 |
|------|-----|------|
| unet_name | `z_image_turbo_bf16.safetensors` | 主模型文件 (11.5GB) |
| weight_dtype | `default` | 权重数据类型，default 自动选择 |

**输出:**
- MODEL - 扩散模型，用于后续采样

---

### 2. CLIPLoader (加载文本编码器)

加载 Qwen 3.4B 文本编码器，用于将提示词转换为模型可理解的向量。

| 参数 | 值 | 说明 |
|------|-----|------|
| clip_name | `qwen_3_4b.safetensors` | 文本编码器文件 (7.5GB) |
| type | `lumina2` | **关键！** 必须设为 lumina2 |
| device | `default` | 运行设备 |

**输出:**
- CLIP - 文本编码器实例

---

### 3. VAELoader (加载VAE)

加载变分自编码器，用于将潜空间数据解码为图像。

| 参数 | 值 | 说明 |
|------|-----|------|
| vae_name | `ae.safetensors` | Flux VAE 文件 (320MB) |

**输出:**
- VAE - VAE 编码/解码器

---

### 4. CLIPTextEncode (提示词编码)

将文字提示词编码为条件向量。

| 参数 | 值 | 说明 |
|------|-----|------|
| text | `1girl` | 提示词内容，支持中英文 |

**输入:**
- clip - 来自 CLIPLoader

**输出:**
- CONDITIONING - 正向条件，告诉模型要生成什么

**提示词技巧:**
- 支持中英文混合
- 简洁明了效果更好
- 可用 `(关键词:1.2)` 调整权重

---

### 5. ConditioningZeroOut (负面条件清零)

将正向条件转换为空的负面条件。Z-Image-Turbo 使用这种方式而不是传统负面提示词。

**输入:**
- conditioning - 来自 CLIPTextEncode

**输出:**
- CONDITIONING - 清零后的负面条件

---

### 6. EmptySD3LatentImage (空白画布)

创建初始的空白潜空间图像。

| 参数 | 值 | 说明 |
|------|-----|------|
| width | `1024` | 图片宽度 (像素) |
| height | `1024` | 图片高度 (像素) |
| batch_size | `1` | 批量大小，一次生成几张 |

**输出:**
- LATENT - 空白潜空间数据

**推荐分辨率:**
- 1024 x 1024 (正方形)
- 1024 x 768 (横向)
- 768 x 1024 (纵向)

---

### 7. ModelSamplingAuraFlow (模型采样设置)

调整模型的采样行为，shift 参数影响生成质量。

| 参数 | 值 | 说明 |
|------|-----|------|
| shift | `3` | 采样偏移值，影响生成风格 |

**输入:**
- model - 来自 UNETLoader

**输出:**
- MODEL - 配置后的模型

**shift 值影响:**
- 1-2: 更写实
- 3: 平衡 (推荐)
- 4-6: 更艺术化

---

### 8. KSampler (核心采样器)

执行扩散采样，这是图像生成的核心步骤。

| 参数 | 值 | 说明 |
|------|-----|------|
| seed | `723736760801122` | 随机种子，相同种子生成相同图片 |
| steps | `9` | 采样步数，Z-Image-Turbo 只需 8-12 步 |
| cfg | `1` | 提示词遵循度，Z-Image 推荐 1-2 |
| sampler_name | `res_multistep` | 采样算法，推荐 res_multistep |
| scheduler | `simple` | 调度器，推荐 simple |
| denoise | `1` | 去噪强度，文生图固定为 1 |

**输入:**
- model - 来自 ModelSamplingAuraFlow
- positive - 来自 CLIPTextEncode (正向条件)
- negative - 来自 ConditioningZeroOut (负面条件)
- latent_image - 来自 EmptySD3LatentImage

**输出:**
- LATENT - 采样后的潜空间数据

**参数调优:**
- 增加 steps (12-20) 可提升细节
- cfg 保持 1-2，过高会过饱和
- seed 设为 -1 或点击骰子图标随机

---

### 9. VAEDecode (VAE解码)

将潜空间数据解码为可见图像。

**输入:**
- samples - 来自 KSampler
- vae - 来自 VAELoader

**输出:**
- IMAGE - RGB 图像数据

---

### 10. SaveImage (保存图片)

将生成的图像保存到磁盘。

| 参数 | 值 | 说明 |
|------|-----|------|
| filename_prefix | `z-image` | 文件名前缀 |

**输入:**
- images - 来自 VAEDecode

**保存位置:** `ComfyUI/output/z-image_00001.png`

---

## 所需模型

| 模型 | 目录 | 大小 | SHA256 |
|------|------|------|--------|
| z_image_turbo_bf16.safetensors | diffusion_models | 11.5GB | `2407613050b809ff...` |
| qwen_3_4b.safetensors | text_encoders | 7.5GB | `6c671498573ac2f7...` |
| ae.safetensors | vae | 320MB | `afc8e28272cd15db...` |

## 显存需求

- 最低: 8GB (需要 offload)
- 推荐: 16GB+
- 如果显存不足，使用 FP8 量化版模型

## 常见问题

**Q: 出现格子/花屏?**
- 检查 qwen_3_4b.safetensors 哈希值是否正确
- CLIPLoader 的 type 必须设为 `lumina2`

**Q: 生成速度慢?**
- Z-Image-Turbo 设计为 8-12 步，不需要 20+ 步

**Q: 提示词不生效?**
- 尝试简化提示词
- 检查 cfg 值，建议 1-2
