# ComfyUI 学习笔记 06：采样器系统 (samplers.py)

## 概述

`samplers.py` 是 ComfyUI 的核心采样系统，实现了扩散模型的去噪过程。它定义了各种采样算法和调度器。

文件位置：`ComfyUI/comfy/samplers.py` (约 1164 行)

---

## 核心概念

### 扩散模型采样原理

```
噪声图像 ──(逐步去噪)──> 清晰图像

步骤：
1. 从纯噪声开始（或从含噪图像开始，用于 img2img）
2. 使用模型预测噪声
3. 根据调度器计算的 sigma 值逐步去噪
4. 重复直到达到目标清晰度
```

### 关键术语

| 术语 | 说明 |
|------|------|
| `sigma` | 噪声强度，从高到低递减 |
| `sigmas` | 噪声强度序列（调度器生成） |
| `timestep` | 时间步，与 sigma 相关 |
| `CFG` | Classifier-Free Guidance，条件引导强度 |
| `denoise` | 去噪强度（0-1），1.0 为完全从噪声开始 |

---

## 采样器列表

### 内置采样器 (KSAMPLER_NAMES)

```python
KSAMPLER_NAMES = [
    # 基础欧拉方法
    "euler",              # 经典欧拉采样器
    "euler_cfg_pp",       # 带 CFG++ 的欧拉
    "euler_ancestral",    # 欧拉祖先采样（有随机性）
    "euler_ancestral_cfg_pp",

    # Heun 方法
    "heun",               # Heun 采样器（二阶）
    "heunpp2",            # Heun++2

    # DPM 系列
    "dpm_2",              # DPM-2
    "dpm_2_ancestral",    # DPM-2 祖先采样
    "dpm_fast",           # 快速 DPM
    "dpm_adaptive",       # 自适应 DPM

    # DPM++ 系列（推荐）
    "dpmpp_2s_ancestral", # DPM++ 2S 祖先采样
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_sde",          # DPM++ SDE
    "dpmpp_sde_gpu",      # GPU 优化版
    "dpmpp_2m",           # DPM++ 2M（常用）
    "dpmpp_2m_cfg_pp",
    "dpmpp_2m_sde",       # DPM++ 2M SDE
    "dpmpp_2m_sde_gpu",
    "dpmpp_2m_sde_heun",
    "dpmpp_3m_sde",       # DPM++ 3M SDE
    "dpmpp_3m_sde_gpu",

    # 其他方法
    "lms",                # Linear Multi-Step
    "ddpm",               # DDPM 采样
    "lcm",                # Latent Consistency Model
    "ipndm", "ipndm_v",   # iPNDM
    "deis",               # DEIS
    "res_multistep",      # RES Multistep
    "gradient_estimation",
    "er_sde",
    "seeds_2", "seeds_3",
    "sa_solver", "sa_solver_pece",
]

# 完整采样器列表
SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]
```

### 采样器特性对比

| 采样器 | 速度 | 质量 | 确定性 | 推荐场景 |
|--------|------|------|--------|----------|
| `euler` | 快 | 中 | 是 | 快速预览 |
| `euler_ancestral` | 快 | 中 | 否 | 创意变化 |
| `dpmpp_2m` | 中 | 高 | 是 | 通用首选 |
| `dpmpp_2m_sde` | 中 | 高 | 否 | 高质量 |
| `dpmpp_3m_sde` | 慢 | 很高 | 否 | 最高质量 |
| `ddim` | 快 | 中 | 是 | 低步数 |
| `uni_pc` | 快 | 高 | 是 | 低步数高质量 |

---

## 调度器系统

### 内置调度器 (SCHEDULER_NAMES)

```python
SCHEDULER_HANDLERS = {
    "simple":           # 简单线性调度
    "sgm_uniform":      # SGM 均匀调度
    "karras":           # Karras 调度（推荐）
    "exponential":      # 指数调度
    "ddim_uniform":     # DDIM 均匀调度
    "beta":             # Beta 分布调度
    "normal":           # 正态分布调度
    "linear_quadratic": # 线性-二次调度
    "kl_optimal":       # KL 最优调度
}
```

### 调度器函数示例

```python
# Simple 调度器 - 均匀分布 sigma
def simple_scheduler(model_sampling, steps):
    s = model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

# Karras 调度器 - 噪声递减更平滑
# 使用 k_diffusion_sampling.get_sigmas_karras

# Beta 调度器 - 基于 Beta 分布
def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):
    total_timesteps = (len(model_sampling.sigmas) - 1)
    ts = 1 - numpy.linspace(0, 1, steps, endpoint=False)
    ts = numpy.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total_timesteps)
    # ...
```

### 调度器工作原理

```
steps = 20, denoise = 1.0

sigma 序列示例 (karras):
[14.61, 10.49, 7.46, 5.26, 3.67, 2.54, 1.74, 1.18,
 0.79, 0.52, 0.34, 0.21, 0.13, 0.08, 0.04, 0.02,
 0.01, 0.00, 0.00, 0.00, 0.00]

采样过程:
sigma[0]=14.61 ──> sigma[1]=10.49 ──> ... ──> sigma[-1]=0.0
  (高噪声)                                      (无噪声)
```

---

## 核心类

### 1. CFGGuider - 条件引导器

```python
class CFGGuider:
    """CFG (Classifier-Free Guidance) 引导器"""

    def __init__(self, model_patcher: ModelPatcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        """设置正向和负向条件"""
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        """设置 CFG 强度"""
        self.cfg = cfg

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """预测噪声（核心函数）"""
        return sampling_function(
            self.inner_model, x, timestep,
            self.conds.get("negative", None),
            self.conds.get("positive", None),
            self.cfg, model_options=model_options, seed=seed
        )

    def sample(self, noise, latent_image, sampler, sigmas, ...):
        """执行采样过程"""
        # 1. 预处理条件
        self.conds = process_conds(...)

        # 2. 执行采样
        output = self.outer_sample(noise, latent_image, sampler, sigmas, ...)

        return output
```

### 2. KSAMPLER - 采样器封装

```python
class KSAMPLER(Sampler):
    """K-diffusion 采样器封装"""

    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback,
               noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        # 设置 inpaint 模型
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        model_k.noise = noise

        # 噪声缩放
        noise = model_wrap.inner_model.model_sampling.noise_scaling(
            sigmas[0], noise, latent_image,
            self.max_denoise(model_wrap, sigmas)
        )

        # 执行采样
        samples = self.sampler_function(
            model_k, noise, sigmas,
            extra_args=extra_args,
            callback=k_callback,
            disable=disable_pbar,
            **self.extra_options
        )

        return samples
```

### 3. KSampler - 高级采样器接口

```python
class KSampler:
    """供节点使用的高级采样器接口"""

    SCHEDULERS = SCHEDULER_NAMES  # 可用调度器
    SAMPLERS = SAMPLER_NAMES      # 可用采样器

    def __init__(self, model, steps, device,
                 sampler=None, scheduler=None, denoise=None):
        self.model = model
        self.device = device
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)

    def calculate_sigmas(self, steps):
        """计算 sigma 序列"""
        sigmas = calculate_sigmas(
            self.model.get_model_object("model_sampling"),
            self.scheduler,
            steps
        )
        return sigmas

    def set_steps(self, steps, denoise=None):
        """设置步数和去噪强度"""
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            # 完全去噪
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            # 部分去噪（img2img）
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg,
               latent_image=None, denoise_mask=None, ...):
        """执行采样"""
        sampler = sampler_object(self.sampler)
        return sample(self.model, noise, positive, negative, cfg,
                      self.device, sampler, self.sigmas, ...)
```

---

## CFG 引导机制

### CFG 公式

```python
def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, ...):
    """CFG 引导计算"""

    if "sampler_cfg_function" in model_options:
        # 自定义 CFG 函数
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        # 标准 CFG 公式
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    # 后处理函数
    for fn in model_options.get("sampler_post_cfg_function", []):
        cfg_result = fn(args)

    return cfg_result
```

### CFG 工作原理

```
无条件预测 (uncond_pred): 模型在没有提示词时的输出
有条件预测 (cond_pred):   模型在有提示词时的输出
CFG 强度 (cond_scale):    引导强度，通常 5-15

公式:
output = uncond_pred + (cond_pred - uncond_pred) * cfg_scale

效果:
- cfg=1.0: 无引导，直接使用条件预测
- cfg=7.0: 适中引导（默认值）
- cfg=15+:  强引导，更符合提示词但可能过饱和
```

---

## 采样流程详解

### 主采样函数

```python
def sampling_function(model, x, timestep, uncond, cond, cond_scale, ...):
    """主采样函数，被所有采样器共享"""

    # 1. 优化：CFG=1 时跳过无条件计算
    if math.isclose(cond_scale, 1.0):
        uncond_ = None
    else:
        uncond_ = uncond

    # 2. 批量计算条件
    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)

    # 3. 应用 CFG
    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, ...)
```

### 条件批处理

```python
def calc_cond_batch(model, conds, x_in, timestep, model_options):
    """批量计算多个条件"""

    out_conds = []
    out_counts = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                # 获取区域和权重
                p = get_area_and_mult(x, x_in, timestep)
                if p is not None:
                    # 按 hooks 分组
                    hooked_to_run.setdefault(p.hooks, list())
                    hooked_to_run[p.hooks] += [(p, i)]

    # 批量执行模型
    for hooks, to_run in hooked_to_run.items():
        # 动态批处理
        output = model.apply_model(input_x, timestep_, **c)

        # 累加输出
        for o in range(batch_chunks):
            out_conds[cond_index] += output[o] * mult[o]
            out_counts[cond_index] += mult[o]

    # 归一化
    for i in range(len(out_conds)):
        out_conds[i] /= out_counts[i]

    return out_conds
```

---

## Inpaint 采样

### KSamplerX0Inpaint

```python
class KSamplerX0Inpaint:
    """Inpaint 采样器包装"""

    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas

    def __call__(self, x, sigma, denoise_mask, ...):
        if denoise_mask is not None:
            # 计算潜在遮罩
            latent_mask = 1. - denoise_mask

            # 混合原图和噪声
            x = x * denoise_mask + \
                self.inner_model.scale_latent_inpaint(...) * latent_mask

        # 执行去噪
        out = self.inner_model(x, sigma, ...)

        if denoise_mask is not None:
            # 混合去噪结果和原图
            out = out * denoise_mask + self.latent_image * latent_mask

        return out
```

---

## 区域条件

### 区域处理

```python
def get_area_and_mult(conds, x_in, timestep_in):
    """获取条件的区域和权重"""

    # 时间步范围检查
    if 'timestep_start' in conds:
        if timestep_in[0] > conds['timestep_start']:
            return None
    if 'timestep_end' in conds:
        if timestep_in[0] < conds['timestep_end']:
            return None

    # 区域裁剪
    if 'area' in conds:
        area = conds['area']
        for i in range(len(dims)):
            input_x = input_x.narrow(i + 2, area[len(dims) + i], area[i])

    # 遮罩处理
    if 'mask' in conds:
        mask = conds['mask']
        mult = mask * strength
    else:
        mult = torch.ones_like(input_x) * strength

    return cond_obj(input_x, mult, conditioning, area, control, ...)
```

---

## 采样器选择指南

### 按用途选择

| 场景 | 推荐采样器 | 推荐调度器 | 步数 |
|------|-----------|-----------|------|
| 快速预览 | euler | normal | 8-15 |
| 通用生成 | dpmpp_2m | karras | 20-30 |
| 高质量 | dpmpp_3m_sde | karras | 25-40 |
| 低步数 | uni_pc | karras | 8-12 |
| 一致性 | euler, ddim | normal | 20+ |
| 多样性 | euler_ancestral | karras | 20-30 |
| Img2Img | dpmpp_2m_sde | karras | 20-30 |
| Inpaint | dpmpp_2m | karras | 20-30 |

### 参数建议

```python
# 通用设置
{
    "sampler": "dpmpp_2m",
    "scheduler": "karras",
    "steps": 20,
    "cfg": 7.0,
    "denoise": 1.0  # txt2img 用 1.0, img2img 用 0.5-0.8
}

# 高质量设置
{
    "sampler": "dpmpp_3m_sde",
    "scheduler": "karras",
    "steps": 30,
    "cfg": 7.5,
}

# 快速设置
{
    "sampler": "uni_pc",
    "scheduler": "karras",
    "steps": 10,
    "cfg": 7.0,
}
```

---

## 调用流程

```
KSampler.sample()
    │
    ├── calculate_sigmas()     # 计算噪声序列
    │       └── SCHEDULER_HANDLERS[scheduler]()
    │
    └── sample()               # 入口函数
            │
            ├── CFGGuider()
            │       ├── set_conds(positive, negative)
            │       └── set_cfg(cfg)
            │
            └── cfg_guider.sample()
                    │
                    ├── process_conds()     # 预处理条件
                    │
                    ├── outer_sample()
                    │       │
                    │       ├── prepare_sampling()
                    │       │
                    │       └── inner_sample()
                    │               │
                    │               └── sampler.sample()
                    │                       │
                    │                       ├── KSamplerX0Inpaint
                    │                       │
                    │                       └── sampler_function()
                    │                               │
                    │                               └── sampling_function()
                    │                                       │
                    │                                       ├── calc_cond_batch()
                    │                                       │
                    │                                       └── cfg_function()
                    │
                    └── cleanup_models()
```

---

## 关键函数一览

| 函数 | 作用 |
|------|------|
| `sample()` | 采样入口函数 |
| `sampling_function()` | 主采样函数 |
| `calc_cond_batch()` | 批量计算条件 |
| `cfg_function()` | CFG 引导计算 |
| `calculate_sigmas()` | 计算 sigma 序列 |
| `sampler_object()` | 创建采样器对象 |
| `process_conds()` | 预处理条件 |
| `get_area_and_mult()` | 获取区域和权重 |

---

## 下一步学习

- `comfy/k_diffusion/sampling.py`：底层采样算法实现
- `comfy/sd.py`：了解模型加载和预处理
- 实践：尝试不同采样器和调度器组合
