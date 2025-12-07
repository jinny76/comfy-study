# ComfyUI 学习笔记 17：模型检测

## 概述

模型检测是 ComfyUI 加载模型的第一步，通过分析模型文件的 state_dict 结构来自动识别模型类型。

| 文件 | 职责 |
|------|------|
| `model_detection.py` | 检测 UNet 配置、匹配模型类型 |
| `supported_models.py` | 60+ 种模型的定义和配置 |
| `supported_models_base.py` | 模型基类定义 |

---

## 检测流程

```
加载 state_dict
      ↓
detect_unet_config()  ← 分析权重键名和形状
      ↓
model_config_from_unet_config()  ← 匹配 supported_models
      ↓
返回模型配置对象
```

### 核心函数

```python
# model_detection.py:766
def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False):
    # 1. 检测 UNet 配置
    unet_config = detect_unet_config(state_dict, unet_key_prefix)

    # 2. 匹配支持的模型
    model_config = model_config_from_unet_config(unet_config, state_dict)

    # 3. 检测量化配置
    quant_config = comfy.utils.detect_layer_quantization(state_dict, unet_key_prefix)
    if quant_config:
        model_config.quant_config = quant_config

    return model_config
```

---

## UNet 配置检测

`detect_unet_config()` 通过检查特定键名来识别模型架构：

### DiT 系列检测

```python
# model_detection.py:40
def detect_unet_config(state_dict, key_prefix):
    state_dict_keys = list(state_dict.keys())

    # MMDiT (SD3)
    if '{}joint_blocks.0.context_block.attn.qkv.weight'.format(key_prefix) in state_dict_keys:
        unet_config = {}
        unet_config["in_channels"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[1]
        unet_config["patch_size"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[2]
        unet_config["depth"] = state_dict['{}x_embedder.proj.weight'.format(key_prefix)].shape[0] // 64
        return unet_config

    # Flux
    if '{}double_blocks.0.img_attn.norm.key_norm.scale'.format(key_prefix) in state_dict_keys:
        dit_config = {}
        dit_config["image_model"] = "flux"
        dit_config["depth"] = count_blocks(state_dict_keys, '{}double_blocks.'.format(key_prefix) + '{}.')
        dit_config["depth_single_blocks"] = count_blocks(state_dict_keys, '{}single_blocks.'.format(key_prefix) + '{}.')
        return dit_config

    # Stable Cascade
    if '{}clf.1.weight'.format(key_prefix) in state_dict_keys:
        unet_config = {}
        if '{}clip_txt_mapper.weight'.format(key_prefix) in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'c'
        elif '{}clip_mapper.weight'.format(key_prefix) in state_dict_keys:
            unet_config['stable_cascade_stage'] = 'b'
        return unet_config
```

### 模型特征键对照表

| 模型 | 特征键 |
|------|--------|
| SD3/MMDiT | `joint_blocks.0.context_block.attn.qkv.weight` |
| Flux | `double_blocks.0.img_attn.norm.key_norm.scale` |
| Flux 2 | `double_stream_modulation_img.lin.weight` |
| Stable Cascade | `clf.1.weight` |
| Stable Audio | `transformer.rotary_pos_emb.inv_freq` |
| AuraFlow | `double_layers.0.attn.w1q.weight` |
| Hunyuan DiT | `mlp_t5.0.weight` |
| Hunyuan Video | `txt_in.individual_token_refiner.blocks.0.norm1.weight` |
| PixArt | `t_block.1.weight` |
| Mochi | `t5_yproj.weight` |
| LTXV | `adaln_single.emb.timestep_embedder.linear_1.bias` |
| Cosmos | `blocks.block0.blocks.0.block.attn.to_q.0.weight` |
| Lumina 2 | `cap_embedder.1.weight` |
| Wan 2.1 | `head.modulation` |
| Hunyuan 3D | `latent_in.weight` |
| HiDream | `caption_projection.0.linear.weight` |
| ACE-Step | `genre_embedder.weight` |

### UNet 检测（SD 1.x/2.x/XL）

```python
# model_detection.py:632
if '{}input_blocks.0.0.weight'.format(key_prefix) in state_dict_keys:
    unet_config = {
        "use_checkpoint": False,
        "image_size": 32,
        "use_spatial_transformer": True,
        "legacy": False
    }

    # 检测 ADM 条件通道
    y_input = '{}label_emb.0.0.weight'.format(key_prefix)
    if y_input in state_dict_keys:
        unet_config["num_classes"] = "sequential"
        unet_config["adm_in_channels"] = state_dict[y_input].shape[1]

    # 模型通道数
    model_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[0]
    in_channels = state_dict['{}input_blocks.0.0.weight'.format(key_prefix)].shape[1]

    # 遍历 input_blocks 计算 transformer_depth
    input_block_count = count_blocks(state_dict_keys, '{}input_blocks'.format(key_prefix) + '.{}.')
    for count in range(input_block_count):
        prefix = '{}input_blocks.{}.'.format(key_prefix, count)
        # 检测每层的 transformer 深度
        out = calculate_transformer_depth(prefix, state_dict_keys, state_dict)
        if out is not None:
            transformer_depth.append(out[0])
            context_dim = out[1]  # 从 attn2.to_k.weight 获取
```

---

## 支持的模型

`supported_models.py` 定义了 60+ 种模型配置：

### 模型类继承结构

```
supported_models_base.BASE
├── SD15
│   ├── SD15_instructpix2pix
│   └── SD20
│       ├── SD21UnclipL
│       ├── SD21UnclipH
│       ├── SD_X4Upscaler
│       └── LotusD
├── SDXL
│   ├── SDXL_instructpix2pix
│   ├── SSD1B
│   ├── Segmind_Vega
│   ├── KOALA_700M
│   └── KOALA_1B
├── SDXLRefiner
├── Stable_Cascade_C
│   └── Stable_Cascade_B
├── SD3
├── Flux
│   ├── FluxInpaint
│   ├── FluxSchnell
│   └── Flux2
├── HunyuanVideo
│   ├── HunyuanVideoI2V
│   └── HunyuanVideo15
├── WAN21_T2V
│   ├── WAN21_I2V
│   ├── WAN21_Vace
│   └── WAN21_Camera
├── Lumina2
│   └── ZImage
└── ... (更多模型)
```

### 模型配置示例

```python
# supported_models.py:32
class SD15(supported_models_base.BASE):
    unet_config = {
        "context_dim": 768,           # CLIP 维度
        "model_channels": 320,        # 基础通道数
        "use_linear_in_transformer": False,
        "adm_in_channels": None,      # 无 ADM 条件
        "use_temporal_attention": False,
    }

    unet_extra_config = {
        "num_heads": 8,
        "num_head_channels": -1,
    }

    latent_format = latent_formats.SD15
    memory_usage_factor = 1.0

    def clip_target(self, state_dict={}):
        return ClipTarget(sd1_clip.SD1Tokenizer, sd1_clip.SD1ClipModel)
```

```python
# supported_models.py:185
class SDXL(supported_models_base.BASE):
    unet_config = {
        "model_channels": 320,
        "use_linear_in_transformer": True,
        "transformer_depth": [0, 0, 2, 2, 10, 10],  # 每层 transformer 块数
        "context_dim": 2048,          # 双 CLIP 拼接
        "adm_in_channels": 2816,      # 时间 + 尺寸条件
        "use_temporal_attention": False,
    }

    latent_format = latent_formats.SDXL
    memory_usage_factor = 0.8

    def model_type(self, state_dict, prefix=""):
        if 'edm_mean' in state_dict:
            return model_base.ModelType.EDM  # Playground V2.5
        elif "v_pred" in state_dict:
            return model_base.ModelType.V_PREDICTION
        else:
            return model_base.ModelType.EPS
```

```python
# supported_models.py:694
class Flux(supported_models_base.BASE):
    unet_config = {
        "image_model": "flux",
        "guidance_embed": True,
    }

    latent_format = latent_formats.Flux
    memory_usage_factor = 3.1
    supported_inference_dtypes = [torch.bfloat16, torch.float16, torch.float32]

    def clip_target(self, state_dict={}):
        t5_detect = sd3_clip.t5_xxl_detect(state_dict, "{}t5xxl.transformer.".format(pref))
        return ClipTarget(FluxTokenizer, flux_clip(**t5_detect))
```

---

## 模型匹配

### 匹配逻辑

```python
# model_detection.py:758
def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in comfy.supported_models.models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None
```

### 匹配规则（BASE 类）

```python
# supported_models_base.py
class BASE:
    @classmethod
    def matches(cls, unet_config, state_dict=None):
        # 检查必需键
        for k in cls.unet_config:
            if k not in unet_config or cls.unet_config[k] != unet_config[k]:
                return False

        # 检查必需的权重键
        if state_dict is not None:
            for k in cls.required_keys:
                if k not in state_dict:
                    return False

        return True
```

### 模型列表（顺序重要）

```python
# supported_models.py:1532
models = [
    LotusD, Stable_Zero123, SD15_instructpix2pix, SD15, SD20,
    SD21UnclipL, SD21UnclipH, SDXL_instructpix2pix, SDXLRefiner, SDXL,
    SSD1B, KOALA_700M, KOALA_1B, Segmind_Vega, SD_X4Upscaler,
    Stable_Cascade_C, Stable_Cascade_B, SV3D_u, SV3D_p,
    SD3, StableAudio, AuraFlow, PixArtAlpha, PixArtSigma,
    HunyuanDiT, HunyuanDiT1, FluxInpaint, Flux, FluxSchnell,
    GenmoMochi, LTXV, HunyuanVideo, HunyuanVideoI2V,
    CosmosT2V, CosmosI2V, ZImage, Lumina2,
    WAN21_T2V, WAN21_I2V, WAN21_Vace, WAN21_Camera,
    Hunyuan3Dv2, HiDream, Chroma, ACEStep, Omnigen2, ...
]
```

**注意**：更具体的模型（如 `SD15_instructpix2pix`）必须在通用模型（如 `SD15`）之前，否则会被错误匹配。

---

## Diffusers 格式转换

支持从 Diffusers 格式检测和转换模型：

```python
# model_detection.py:839
def unet_config_from_diffusers_unet(state_dict, dtype=None):
    if "conv_in.weight" not in state_dict:
        return None

    # 计算 transformer_depth
    down_blocks = count_blocks(state_dict, "down_blocks.{}")
    for i in range(down_blocks):
        attn_blocks = count_blocks(state_dict, "down_blocks.{}.attentions.".format(i) + '{}')
        for ab in range(attn_blocks):
            transformer_count = count_blocks(state_dict,
                "down_blocks.{}.attentions.{}.transformer_blocks.".format(i, ab) + '{}')
            transformer_depth.append(transformer_count)

    # 匹配预定义配置
    supported_models = [SDXL, SDXL_refiner, SD21, SD15, ...]
    for unet_config in supported_models:
        if all(match[k] == unet_config[k] for k in match):
            return convert_config(unet_config)

    return None
```

### MMDiT 转换

```python
# model_detection.py:998
def convert_diffusers_mmdit(state_dict, output_prefix=""):
    if 'joint_transformer_blocks.0.attn.add_k_proj.weight' in state_dict:
        # AuraFlow
        sd_map = auraflow_to_diffusers(...)
    elif 'x_embedder.weight' in state_dict:
        # Flux
        sd_map = flux_to_diffusers(...)
    elif 'transformer_blocks.0.attn.add_q_proj.weight' in state_dict:
        # SD3
        sd_map = mmdit_to_diffusers(...)

    # 应用权重映射
    for k in sd_map:
        weight = state_dict.get(k)
        if weight is not None:
            out_sd[sd_map[k]] = weight
```

---

## 辅助函数

### 计数块

```python
# model_detection.py:9
def count_blocks(state_dict_keys, prefix_string):
    """统计连续编号的块数量"""
    count = 0
    while True:
        found = False
        for k in state_dict_keys:
            if k.startswith(prefix_string.format(count)):
                found = True
                break
        if not found:
            break
        count += 1
    return count

# 用法示例
count_blocks(keys, 'input_blocks.{}.')  # → 12
count_blocks(keys, 'double_blocks.{}.')  # → 19 (Flux)
```

### 计算 Transformer 深度

```python
# model_detection.py:22
def calculate_transformer_depth(prefix, state_dict_keys, state_dict):
    """计算指定前缀下的 transformer 块深度"""
    transformer_prefix = prefix + "1.transformer_blocks."
    transformer_keys = list(filter(lambda a: a.startswith(transformer_prefix), state_dict_keys))

    if len(transformer_keys) > 0:
        depth = count_blocks(state_dict_keys, transformer_prefix + '{}')
        context_dim = state_dict['{}0.attn2.to_k.weight'.format(transformer_prefix)].shape[1]
        use_linear = len(state_dict['{}1.proj_in.weight'.format(prefix)].shape) == 2
        return depth, context_dim, use_linear, time_stack, time_stack_cross

    return None
```

### UNet 前缀检测

```python
# model_detection.py:782
def unet_prefix_from_state_dict(state_dict):
    candidates = [
        "model.diffusion_model.",  # LDM/SGM 格式
        "model.model.",            # 音频模型
        "net.",                    # Cosmos
    ]

    counts = {k: 0 for k in candidates}
    for k in state_dict:
        for c in candidates:
            if k.startswith(c):
                counts[c] += 1

    top = max(counts, key=counts.get)
    if counts[top] > 5:
        return top
    else:
        return "model."  # AuraFlow 等
```

---

## 模型类型检测

### 预测类型

```python
# model_base.py
class ModelType(Enum):
    EPS = 0              # ε 预测（SD 1.x 默认）
    V_PREDICTION = 1     # v 预测（SD 2.x）
    EDM = 2              # EDM 调度
    FLOW = 3             # Flow Matching（Flux Schnell）
    V_PREDICTION_EDM = 4 # v 预测 + EDM
```

### SD 2.x v-prediction 检测

```python
# supported_models.py:96
class SD20(BASE):
    def model_type(self, state_dict, prefix=""):
        if self.unet_config["in_channels"] == 4:
            # 通过权重统计方差判断
            k = "{}output_blocks.11.1.transformer_blocks.0.norm1.bias".format(prefix)
            out = state_dict.get(k, None)
            if out is not None and torch.std(out, unbiased=False) > 0.09:
                return model_base.ModelType.V_PREDICTION
        return model_base.ModelType.EPS
```

### SDXL 变体检测

```python
# supported_models.py:199
class SDXL(BASE):
    def model_type(self, state_dict, prefix=""):
        if 'edm_mean' in state_dict:
            # Playground V2.5
            self.sampling_settings["sigma_data"] = 0.5
            return model_base.ModelType.EDM
        elif "edm_vpred.sigma_max" in state_dict:
            return model_base.ModelType.V_PREDICTION_EDM
        elif "v_pred" in state_dict:
            if "ztsnr" in state_dict:
                self.sampling_settings["zsnr"] = True
            return model_base.ModelType.V_PREDICTION
        else:
            return model_base.ModelType.EPS
```

---

## 完整模型分类

### 图像生成

| 模型 | 类型 | CLIP | Latent |
|------|------|------|--------|
| SD 1.5 | UNet | CLIP-L (768) | SD15 |
| SD 2.x | UNet | CLIP-H (1024) | SD15 |
| SDXL | UNet | CLIP-L + CLIP-G | SDXL |
| SD3 | MMDiT | CLIP-L + CLIP-G + T5 | SD3 |
| Flux | DiT | CLIP-L + T5 | Flux |
| Flux 2 | DiT | Mistral 24B | Flux2 |
| PixArt | DiT | T5 | SD15/SDXL |
| AuraFlow | DiT | T5 | SDXL |
| Lumina 2 | DiT | Gemma 2B | Flux |
| Z-Image | DiT | Qwen 4B | Flux |

### 视频生成

| 模型 | 特点 |
|------|------|
| SVD | 基于 SD 的视频 |
| Mochi | 48层 DiT |
| LTXV | Lightricks 视频 |
| Hunyuan Video | 双流 DiT |
| Cosmos | 时空位置编码 |
| Wan 2.1 | 阿里视频模型 |

### 其他

| 模型 | 用途 |
|------|------|
| Stable Audio | 音频生成 |
| ACE-Step | 音乐生成 |
| Hunyuan 3D | 3D 生成 |
| Stable Zero123 | 3D 重建 |
| SV3D | 3D 视频 |

---

## 总结

| 阶段 | 函数 | 说明 |
|------|------|------|
| 1. 检测 | `detect_unet_config()` | 通过键名和权重形状推断配置 |
| 2. 匹配 | `model_config_from_unet_config()` | 遍历 60+ 模型类寻找匹配 |
| 3. 实例化 | `model_config(unet_config)` | 创建模型配置对象 |
| 4. 加载 | `get_model()` | 创建实际模型实例 |

ComfyUI 通过分析 state_dict 的结构（键名前缀、权重形状、特定层存在性）自动识别模型类型，无需用户手动指定。
