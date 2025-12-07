# Hunyuan3D v2.1 图生3D工作流

> 文件: `3d_hunyuan3d-v2.1.json`

## 概述

Hunyuan3D v2.1 是腾讯混元推出的图像转 3D 模型，可以将 2D 图片转换为 3D 模型 (GLB 格式)。

## 工作流结构图

```
┌─────────────────────┐
│ ImageOnlyCheckpoint │
│   Loader (加载模型)  │
└──────────┬──────────┘
           │
     ┌─────┼─────┐
     │     │     │
     ▼     ▼     ▼
  MODEL  CLIP   VAE
     │   VISION  │
     │     │     │
     ▼     │     │
┌─────────┐│     │
│ModelSamp││     │
│lingAura ││     │
│ Flow    ││     │
└────┬────┘│     │
     │     │     │
     │     ▼     │
     │ ┌───────────────┐
     │ │  LoadImage    │
     │ │  (上传图片)    │
     │ └───────┬───────┘
     │         │
     │         ▼
     │ ┌───────────────┐
     │ │CLIPVisionEnco │
     │ │de (图像编码)   │
     │ └───────┬───────┘
     │         │
     │         ▼
     │ ┌───────────────┐
     │ │Hunyuan3Dv2    │
     │ │Conditioning   │
     │ └───────┬───────┘
     │         │
     ▼         ▼
┌───────────────────────────┐
│        KSampler           │◄── EmptyLatentHunyuan3Dv2
│       (采样生成)           │     (空白3D潜空间)
└─────────────┬─────────────┘
              │
              ▼
┌─────────────────────────┐
│   VAEDecodeHunyuan3D    │◄── VAE
│     (解码为体素)         │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│      VoxelToMesh        │
│    (体素转网格)          │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│        SaveGLB          │
│   (保存3D模型文件)       │
└─────────────────────────┘
```

## 节点详解

### 1. ImageOnlyCheckpointLoader (加载3D模型)

加载 Hunyuan3D v2.1 的完整模型包，包含 MODEL、CLIP_VISION 和 VAE。

| 参数 | 值 | 说明 |
|------|-----|------|
| ckpt_name | `hunyuan_3d_v2.1.safetensors` | 3D 模型文件 |

**输出:**
- MODEL - 3D 扩散模型
- CLIP_VISION - 视觉编码器，用于理解输入图像
- VAE - 3D VAE，用于解码体素数据

---

### 2. LoadImage (加载图片)

上传要转换为 3D 的图片。

| 参数 | 值 | 说明 |
|------|-----|------|
| image | `hanzong.png` | 输入图片文件名 |

**输出:**
- IMAGE - 图像数据
- MASK - 蒙版 (未使用)

**图片要求:**
- 最好是单个物体
- 白色或简单背景效果更好
- 正面角度为佳

---

### 3. CLIPVisionEncode (图像视觉编码)

将输入图片编码为视觉特征向量。

| 参数 | 值 | 说明 |
|------|-----|------|
| crop | `center` | 裁剪方式: center 居中裁剪 |

**输入:**
- clip_vision - 来自 ImageOnlyCheckpointLoader
- image - 来自 LoadImage

**输出:**
- CLIP_VISION_OUTPUT - 图像特征向量

---

### 4. Hunyuan3Dv2Conditioning (3D 条件生成)

将图像特征转换为 3D 生成的正/负条件。

**输入:**
- clip_vision_output - 来自 CLIPVisionEncode

**输出:**
- positive - 正向条件 (要生成什么)
- negative - 负向条件 (避免什么)

---

### 5. ModelSamplingAuraFlow (模型采样设置)

调整 3D 模型的采样行为。

| 参数 | 值 | 说明 |
|------|-----|------|
| shift | `1` | 采样偏移值，3D 生成推荐 1 |

**输入:**
- model - 来自 ImageOnlyCheckpointLoader

**输出:**
- MODEL - 配置后的模型

---

### 6. EmptyLatentHunyuan3Dv2 (空白3D潜空间)

创建初始的空白 3D 潜空间数据。

| 参数 | 值 | 说明 |
|------|-----|------|
| resolution | `4096` | 潜空间分辨率，影响最终精度 |
| batch_size | `1` | 批量大小 |

**输出:**
- LATENT - 空白 3D 潜空间数据

**resolution 说明:**
- 2048: 较粗糙，速度快
- 4096: 平衡 (推荐)
- 8192: 高精度，显存需求大

---

### 7. KSampler (核心采样器)

执行 3D 扩散采样。

| 参数 | 值 | 说明 |
|------|-----|------|
| seed | `272337695461673` | 随机种子 |
| steps | `30` | 采样步数，3D 需要更多步 |
| cfg | `5` | 提示词遵循度 |
| sampler_name | `euler` | 采样算法 |
| scheduler | `normal` | 调度器 |
| denoise | `1` | 去噪强度 |

**输入:**
- model - 来自 ModelSamplingAuraFlow
- positive - 来自 Hunyuan3Dv2Conditioning
- negative - 来自 Hunyuan3Dv2Conditioning
- latent_image - 来自 EmptyLatentHunyuan3Dv2

**输出:**
- LATENT - 采样后的 3D 潜空间数据

**参数调优:**
- steps 30-50 效果较好
- cfg 4-6 通常最佳

---

### 8. VAEDecodeHunyuan3D (3D VAE 解码)

将潜空间数据解码为体素 (Voxel) 数据。

| 参数 | 值 | 说明 |
|------|-----|------|
| num_chunks | `8000` | 解码块数，影响速度和显存 |
| octree_resolution | `256` | 八叉树分辨率，影响细节程度 |

**输入:**
- samples - 来自 KSampler
- vae - 来自 ImageOnlyCheckpointLoader

**输出:**
- VOXEL - 体素数据

**octree_resolution 说明:**
- 128: 低精度，速度快
- 256: 中等精度 (推荐)
- 512: 高精度，显存需求大

---

### 9. VoxelToMesh (体素转网格)

将体素数据转换为三角网格。

| 参数 | 值 | 说明 |
|------|-----|------|
| algorithm | `surface net` | 网格生成算法 |
| threshold | `0.6` | 表面阈值，影响网格边界 |

**输入:**
- voxel - 来自 VAEDecodeHunyuan3D

**输出:**
- MESH - 三角网格数据

**algorithm 选项:**
- `surface net` - 平滑，推荐
- `marching cubes` - 经典算法

**threshold 说明:**
- 0.4-0.5: 较厚实
- 0.6: 平衡 (推荐)
- 0.7-0.8: 较薄，更多细节

---

### 10. SaveGLB (保存3D模型)

将网格保存为 GLB 格式的 3D 模型文件。

| 参数 | 值 | 说明 |
|------|-----|------|
| filename_prefix | `mesh/ComfyUI` | 文件名前缀 |

**输入:**
- mesh - 来自 VoxelToMesh

**保存位置:** `ComfyUI/output/mesh/ComfyUI_00001.glb`

**GLB 格式:**
- 二进制 glTF 格式
- 可在大多数 3D 软件中打开
- 支持 Blender、Unity、Unreal 等

---

## 所需模型

| 模型 | 目录 | 下载链接 |
|------|------|----------|
| hunyuan_3d_v2.1.safetensors | checkpoints | [HuggingFace](https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged/resolve/main/hunyuan_3d_v2.1.safetensors) |

## 显存需求

- 最低: 12GB
- 推荐: 16GB+
- 高精度 (octree_resolution=512): 24GB+

## 使用步骤

1. 下载并放置模型到 `models/checkpoints/`
2. 加载工作流
3. 在 LoadImage 节点上传图片
4. 点击 Queue Prompt 运行
5. 生成的 GLB 文件在 `output/mesh/` 目录

## 输入图片建议

- **单个物体**: 一张图片中只有一个主体
- **清晰背景**: 白色或纯色背景效果最好
- **正面角度**: 物体正面朝向相机
- **光照均匀**: 避免强烈阴影
- **分辨率**: 512x512 或 1024x1024

## 常见问题

**Q: 生成的模型有空洞?**
- 降低 threshold 值 (如 0.5)
- 增加 octree_resolution

**Q: 生成速度很慢?**
- 降低 resolution 和 octree_resolution
- 减少 steps (如 20)

**Q: 显存不足?**
- 降低 num_chunks (如 4000)
- 降低 octree_resolution (如 128)

**Q: 模型表面不光滑?**
- 使用 `surface net` 算法
- 在 Blender 中进行后处理平滑
