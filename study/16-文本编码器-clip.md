# ComfyUI 学习笔记 16：文本编码器（CLIP 系统）

## 概述

文本编码器负责将用户输入的 Prompt 转换为模型可理解的向量表示。ComfyUI 支持多种文本编码器：

| 编码器 | 模型 | 维度 | 特点 |
|--------|------|------|------|
| CLIP-L | SD 1.x | 768 | OpenAI 原始 CLIP |
| CLIP-G | SDXL | 1280 | 更大的 CLIP 变体 |
| T5-XXL | SD3/Flux | 4096 | Google T5 编码器 |
| Llama/Mistral | Flux 2 | 5120 | LLM 作为编码器 |
| Qwen | Z-Image | - | 中文大模型 |

相关文件：
- `comfy/clip_model.py` - CLIP 模型实现
- `comfy/sd1_clip.py` - SD1 CLIP 封装、Tokenizer
- `comfy/sdxl_clip.py` - SDXL 双 CLIP
- `comfy/clip_vision.py` - CLIP Vision（图像编码）
- `comfy/text_encoders/*.py` - 其他编码器（T5、Llama 等）

---

## CLIP 模型架构

### 核心组件

```
CLIPTextModel
├── CLIPEmbeddings (词嵌入 + 位置嵌入)
│   ├── token_embedding: Embedding(49408, 768)
│   └── position_embedding: Embedding(77, 768)
├── CLIPEncoder (Transformer 编码器)
│   └── layers: [CLIPLayer] × 12
│       ├── layer_norm1
│       ├── self_attn: CLIPAttention
│       ├── layer_norm2
│       └── mlp: CLIPMLP
├── final_layer_norm
└── text_projection: Linear(768, 768)
```

### clip_model.py 实现

```python
# clip_model.py:5
class CLIPAttention(torch.nn.Module):
    def __init__(self, embed_dim, heads, dtype, device, operations):
        self.heads = heads
        self.q_proj = operations.Linear(embed_dim, embed_dim)
        self.k_proj = operations.Linear(embed_dim, embed_dim)
        self.v_proj = operations.Linear(embed_dim, embed_dim)
        self.out_proj = operations.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None, optimized_attention=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        out = optimized_attention(q, k, v, self.heads, mask)
        return self.out_proj(out)
```

### 激活函数

```python
# clip_model.py:24
ACTIVATIONS = {
    "quick_gelu": lambda a: a * torch.sigmoid(1.702 * a),  # SD 1.x
    "gelu": torch.nn.functional.gelu,                       # SDXL
    "gelu_pytorch_tanh": lambda a: torch.nn.functional.gelu(a, approximate="tanh"),
}
```

---

## SD1 CLIP 封装

### SDClipModel

```python
# sd1_clip.py:81
class SDClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    LAYERS = ["last", "pooled", "hidden", "all"]

    def __init__(self, device="cpu", max_length=77, layer="last",
                 layer_idx=None, textmodel_json_config=None, dtype=None,
                 special_tokens={"start": 49406, "end": 49407, "pad": 49407},
                 ...):
        # 加载配置
        with open(textmodel_json_config) as f:
            config = json.load(f)

        # 创建 Transformer
        self.transformer = model_class(config, dtype, device, operations)
        self.max_length = max_length
        self.special_tokens = special_tokens
```

### 模型配置（sd1_clip_config.json）

```json
{
  "_name_or_path": "openai/clip-vit-large-patch14",
  "hidden_size": 768,
  "intermediate_size": 3072,
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "max_position_embeddings": 77,
  "vocab_size": 49408,
  "hidden_act": "quick_gelu",
  "eos_token_id": 49407
}
```

### 输出层选择

```python
# sd1_clip.py:152
def set_clip_options(self, options):
    layer_idx = options.get("layer", self.layer_idx)
    if layer_idx is None or abs(layer_idx) > self.num_layers:
        self.layer = "last"      # 最后一层
    else:
        self.layer = "hidden"    # 指定隐藏层
        self.layer_idx = layer_idx

# SDXL 使用倒数第二层
# layer="hidden", layer_idx=-2
```

---

## Tokenizer 系统

### SDTokenizer

```python
# sd1_clip.py:468
class SDTokenizer:
    def __init__(self, tokenizer_path=None, max_length=77,
                 embedding_directory=None, embedding_size=768,
                 embedding_key='clip_l', tokenizer_class=CLIPTokenizer,
                 has_start_token=True, has_end_token=True, ...):

        self.tokenizer = tokenizer_class.from_pretrained(tokenizer_path)
        self.max_length = max_length

        # 特殊 token
        empty = self.tokenizer('')["input_ids"]
        self.start_token = empty[0]  # 49406
        self.end_token = empty[1]    # 49407
        self.pad_token = self.end_token
```

### Prompt 权重解析

支持 `(word:weight)` 语法：

```python
# sd1_clip.py:330
def token_weights(string, current_weight):
    """解析权重语法 (word:1.2)"""
    a = parse_parentheses(string)
    out = []
    for x in a:
        weight = current_weight
        if x.endswith(')') and x.startswith('('):
            x = x[1:-1]
            xx = x.rfind(":")
            weight *= 1.1  # 默认增加 10%
            if xx > 0:
                try:
                    weight = float(x[xx+1:])
                    x = x[:xx]
                except:
                    pass
            out += token_weights(x, weight)
        else:
            out += [(x, current_weight)]
    return out
```

### 权重应用

```python
# sd1_clip.py:27
class ClipTokenWeightEncoder:
    def encode_token_weights(self, token_weight_pairs):
        # 编码所有 token
        out, pooled = self.encode(to_encode)

        # 应用权重
        for k in range(sections):
            z = out[k:k+1]
            if has_weights:
                z_empty = out[-1]  # 空 token 的编码
                for i in range(len(z)):
                    for j in range(len(z[i])):
                        weight = token_weight_pairs[k][j][1]
                        if weight != 1.0:
                            # 权重插值：(向量 - 空向量) * 权重 + 空向量
                            z[i][j] = (z[i][j] - z_empty[j]) * weight + z_empty[j]
```

---

## Embedding 加载

### 加载 Textual Inversion

```python
# sd1_clip.py:397
def load_embed(embedding_name, embedding_directory, embedding_size, embed_key=None):
    # 搜索嵌入文件
    for embed_dir in embedding_directory:
        embed_path = os.path.join(embed_dir, embedding_name)
        extensions = ['.safetensors', '.pt', '.bin']
        for x in extensions:
            if os.path.isfile(embed_path + x):
                valid_file = embed_path + x
                break

    # 加载嵌入
    if embed_path.endswith(".safetensors"):
        embed = safetensors.torch.load_file(embed_path)
    else:
        embed = torch.load(embed_path)

    # 解析嵌入格式
    if 'string_to_param' in embed:
        embed_out = next(iter(embed['string_to_param'].values()))
    return embed_out
```

### 使用嵌入

在 prompt 中用 `embedding:name` 引用：

```python
# sd1_clip.py:567
if word.startswith(self.embedding_identifier):  # "embedding:"
    embedding_name = word[len(self.embedding_identifier):]
    embed, leftover = self._try_get_embedding(embedding_name)
    if embed is not None:
        # 将嵌入向量插入 token 序列
        tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
```

---

## SDXL 双 CLIP

SDXL 使用两个 CLIP 编码器并联：

```python
# sdxl_clip.py:41
class SDXLClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        # CLIP-L (768 维，倒数第二层)
        self.clip_l = sd1_clip.SDClipModel(
            layer="hidden", layer_idx=-2,
            layer_norm_hidden_state=False
        )
        # CLIP-G (1280 维，倒数第二层)
        self.clip_g = SDXLClipG(device=device, dtype=dtype)

    def encode_token_weights(self, token_weight_pairs):
        # 分别编码
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs["g"])
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs["l"])

        # 拼接：[clip_l, clip_g] → 2048 维
        cut_to = min(l_out.shape[1], g_out.shape[1])
        return torch.cat([l_out[:,:cut_to], g_out[:,:cut_to]], dim=-1), g_pooled
```

### CLIP-G 配置

```json
{
  "hidden_size": 1280,
  "intermediate_size": 5120,
  "num_attention_heads": 20,
  "num_hidden_layers": 32,
  "hidden_act": "gelu"
}
```

---

## T5 编码器

用于 SD3、Flux 等新模型。

### T5 架构

```python
# text_encoders/t5.py:225
class T5(torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        self.num_layers = config_dict["num_layers"]  # 24 层
        model_dim = config_dict["d_model"]           # 4096

        self.encoder = T5Stack(...)
        self.shared = operations.Embedding(vocab_size, model_dim)

    def forward(self, input_ids, attention_mask, ...):
        x = self.shared(input_ids)  # 词嵌入
        return self.encoder(x, attention_mask=attention_mask)
```

### T5 LayerNorm（RMSNorm）

```python
# text_encoders/t5.py:6
class T5LayerNorm(torch.nn.Module):
    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x
```

### 相对位置编码

T5 使用相对位置 Bias 而非绝对位置编码：

```python
# text_encoders/t5.py:88
@staticmethod
def _relative_position_bucket(relative_position, bidirectional=True,
                              num_buckets=32, max_distance=128):
    """将相对位置映射到桶索引"""
    # 使用对数分桶，允许处理更长序列
    relative_buckets = 0
    if bidirectional:
        num_buckets //= 2
        relative_buckets += (relative_position > 0).long() * num_buckets
        relative_position = torch.abs(relative_position)
    ...
```

---

## Flux 编码器

Flux 使用 CLIP-L + T5-XXL 双编码器：

```python
# text_encoders/flux.py:36
class FluxClipModel(torch.nn.Module):
    def __init__(self, dtype_t5=None, device="cpu", dtype=None):
        # CLIP-L (768 维)
        self.clip_l = sd1_clip.SDClipModel(return_projected_pooled=False)
        # T5-XXL (4096 维)
        self.t5xxl = T5XXLModel(dtype=dtype_t5)

    def encode_token_weights(self, token_weight_pairs):
        t5_out, _ = self.t5xxl.encode_token_weights(token_weight_pairs["t5xxl"])
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs["l"])
        # 返回 T5 输出和 CLIP pooled
        return t5_out, l_pooled
```

### Flux 2（Mistral 编码器）

```python
# text_encoders/flux.py:126
class Flux2Tokenizer(sd1_clip.SD1Tokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(name="mistral3_24b", tokenizer=Mistral3Tokenizer)
        # 使用指令模板
        self.llama_template = '[SYSTEM_PROMPT]...[/SYSTEM_PROMPT][INST]{}[/INST]'

    def tokenize_with_weights(self, text, ...):
        llama_text = self.llama_template.format(text)
        return super().tokenize_with_weights(llama_text, disable_weights=True)
```

---

## CLIP Vision

用于图像编码（IP-Adapter、ControlNet 等）。

### 图像预处理

```python
# clip_vision.py:20
def clip_preprocess(image, size=224,
                   mean=[0.48145466, 0.4578275, 0.40821073],
                   std=[0.26862954, 0.26130258, 0.27577711], crop=True):
    image = image.movedim(-1, 1)  # NHWC → NCHW

    # 缩放和裁剪到目标尺寸
    scale = size / min(image.shape[2], image.shape[3])
    image = F.interpolate(image, size=scale_size, mode="bicubic")
    h = (image.shape[2] - size) // 2
    w = (image.shape[3] - size) // 2
    image = image[:, :, h:h+size, w:w+size]

    # 归一化
    return (image - mean) / std
```

### Vision 模型

```python
# clip_vision.py:45
class ClipVisionModel:
    def __init__(self, json_config):
        config = json.load(open(json_config))
        self.image_size = config.get("image_size", 224)
        self.model = CLIPVisionModelProjection(config, dtype, device, ops)

    def encode_image(self, image, crop=True):
        # 预处理
        pixel_values = clip_preprocess(image, size=self.image_size)
        # 编码
        out = self.model(pixel_values=pixel_values, intermediate_output=-2)

        outputs = Output()
        outputs["last_hidden_state"] = out[0]
        outputs["image_embeds"] = out[2]
        outputs["penultimate_hidden_states"] = out[1]
        return outputs
```

### 支持的 Vision 模型

```python
# clip_vision.py:39
IMAGE_ENCODERS = {
    "clip_vision_model": CLIPVisionModelProjection,  # OpenAI CLIP
    "siglip_vision_model": CLIPVisionModelProjection, # Google SigLIP
    "dinov2": Dinov2Model,  # Meta DINOv2
}
```

### 模型检测

```python
# clip_vision.py:118
def load_clipvision_from_sd(sd, prefix="", convert_keys=False):
    # 根据层数判断模型类型
    if "vision_model.encoder.layers.47.layer_norm1.weight" in sd:
        json_config = "clip_vision_config_g.json"     # ViT-G
    elif "vision_model.encoder.layers.30.layer_norm1.weight" in sd:
        json_config = "clip_vision_config_h.json"     # ViT-H
    elif "vision_model.encoder.layers.22.layer_norm1.weight" in sd:
        # 进一步判断 ViT-L / SigLIP
        embed_shape = sd["position_embedding.weight"].shape[0]
        if embed_shape == 729:
            json_config = "clip_vision_siglip_384.json"
        ...
```

---

## 编码器类型对比

| 特性 | CLIP-L | CLIP-G | T5-XXL | Mistral |
|------|--------|--------|--------|---------|
| 维度 | 768 | 1280 | 4096 | 5120 |
| 层数 | 12 | 32 | 24 | 40 |
| 位置编码 | 绝对 | 绝对 | 相对 | RoPE |
| 最大长度 | 77 | 77 | 256+ | 无限 |
| 激活 | quick_gelu | gelu | gelu_tanh | silu |
| 用途 | SD1/SDXL | SDXL | SD3/Flux | Flux 2 |

---

## 使用示例

### 基本编码流程

```python
# 1. Tokenize
tokens = tokenizer.tokenize_with_weights("a beautiful cat")
# → {"l": [[(49406, 1.0), (320, 1.0), (2103, 1.0), (2368, 1.0), (49407, 1.0), ...]]}

# 2. Encode
cond, pooled = clip_model.encode_token_weights(tokens)
# cond: [1, 77, 768]  - 条件向量
# pooled: [1, 768]    - 池化向量

# 3. 传给 UNet
model_options["transformer_options"]["cond"] = cond
```

### 带权重的 Prompt

```python
# 权重语法
"a (beautiful:1.2) cat"        # beautiful 权重 1.2
"a ((important)) thing"        # important 权重 1.1 * 1.1 = 1.21
"(red:0.5) (blue:1.5) sky"     # 分别指定权重
```

### 使用 Embedding

```python
# Prompt 中引用
"a photo of embedding:my_style"

# Tokenizer 会自动加载 embeddings/my_style.safetensors
# 并将其向量插入 token 序列
```

---

## 总结

| 组件 | 职责 |
|------|------|
| `clip_model.py` | CLIP Transformer 实现 |
| `sd1_clip.py` | Tokenizer + 权重解析 + Embedding |
| `sdxl_clip.py` | SDXL 双 CLIP 封装 |
| `clip_vision.py` | 图像编码器 |
| `text_encoders/*.py` | T5、Llama 等新编码器 |

核心流程：
1. **Tokenize** - 将文本转为 token ID + 权重
2. **Embed** - token ID → 向量
3. **Encode** - Transformer 编码
4. **Weight** - 应用用户指定权重
5. **Output** - 返回条件向量供 UNet 使用
