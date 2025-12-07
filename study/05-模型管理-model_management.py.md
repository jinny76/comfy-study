# ComfyUI 学习笔记 05：模型管理 (comfy/model_management.py)

## 概述

`model_management.py` 是 ComfyUI 的显存和模型管理核心，负责：
- 检测硬件环境（GPU类型、显存大小）
- 管理模型的加载/卸载
- 显存分配和释放策略
- 数据类型选择（fp16/bf16/fp32）
- 中断处理机制

文件位置：`ComfyUI/comfy/model_management.py` (约 1555 行)

---

## 硬件状态枚举

### VRAMState - 显存状态

```python
class VRAMState(Enum):
    DISABLED = 0    # 无显存（纯CPU模式）
    NO_VRAM = 1     # 极低显存（<2GB），启用所有节省显存的选项
    LOW_VRAM = 2    # 低显存（2-4GB）
    NORMAL_VRAM = 3 # 正常显存（4-8GB）
    HIGH_VRAM = 4   # 高显存（>8GB）
    SHARED = 5      # 共享显存（如核显，CPU和GPU共用内存）
```

### CPUState - 计算设备状态

```python
class CPUState(Enum):
    GPU = 0   # 使用GPU（NVIDIA/AMD/Intel）
    CPU = 1   # 仅CPU
    MPS = 2   # Apple Silicon (M1/M2/M3)
```

---

## 硬件检测

### 支持的硬件平台

```python
# NVIDIA CUDA
torch.cuda.is_available()

# AMD ROCm
torch.version.hip

# Intel XPU (Arc显卡)
torch.xpu.is_available()

# Apple MPS
torch.backends.mps.is_available()

# 华为昇腾 NPU
torch.npu.is_available()

# 寒武纪 MLU
torch.mlu.is_available()
```

### 获取计算设备

```python
def get_torch_device():
    if directml_enabled:
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        elif is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        else:
            return torch.device(torch.cuda.current_device())
```

---

## 内存管理

### 获取内存信息

```python
def get_total_memory(dev=None):
    """获取设备总显存"""
    if dev.type == 'cpu' or dev.type == 'mps':
        return psutil.virtual_memory().total
    else:
        _, mem_total = torch.cuda.mem_get_info(dev)
        return mem_total

def get_free_memory(dev=None):
    """获取设备可用显存"""
    if dev.type == 'cpu':
        return psutil.virtual_memory().available
    else:
        stats = torch.cuda.memory_stats(dev)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
        mem_free_torch = mem_reserved - mem_active
        return mem_free_cuda + mem_free_torch
```

### 显存保留策略

```python
# Windows 需要更多保留显存（共享显存问题）
EXTRA_RESERVED_VRAM = 400 * 1024 * 1024  # 400MB
if WINDOWS:
    EXTRA_RESERVED_VRAM = 600 * 1024 * 1024  # 600MB
    if total_vram > (15 * 1024):  # 16GB+ 显卡
        EXTRA_RESERVED_VRAM += 100 * 1024 * 1024  # 额外100MB

def minimum_inference_memory():
    """推理所需的最小显存"""
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()  # ~1GB
```

---

## 模型加载管理

### LoadedModel 类

```python
class LoadedModel:
    def __init__(self, model):
        self.model = model
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True

    def model_memory(self):
        """获取模型占用的显存"""
        return self.model.model_size()

    def model_load(self, lowvram_model_memory, force_patch_weights=False):
        """加载模型到GPU"""
        self.model.model_patches_to(self.model.model_dtype())
        self.model_use_more_vram(lowvram_model_memory)
        return self.model.model

    def model_unload(self, memory_to_free=None):
        """从GPU卸载模型"""
        if memory_to_free is not None:
            # 部分卸载
            freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
            if freed >= memory_to_free:
                return False
        # 完全卸载
        self.model.detach()
        return True
```

### 全局模型列表

```python
current_loaded_models = []  # 当前已加载的模型列表
```

### 加载模型到GPU

```python
def load_models_gpu(models, memory_required=0, force_patch_weights=False):
    """将模型加载到GPU"""
    cleanup_models_gc()

    # 1. 计算需要的总显存
    total_memory_required = {}
    for loaded_model in models_to_load:
        device = loaded_model.device
        total_memory_required[device] = (
            total_memory_required.get(device, 0) +
            loaded_model.model_memory_required(device)
        )

    # 2. 释放显存给新模型
    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

    # 3. 加载模型
    for loaded_model in models_to_load:
        lowvram_model_memory = calculate_lowvram_memory(...)
        loaded_model.model_load(lowvram_model_memory)
        current_loaded_models.insert(0, loaded_model)
```

### 释放显存

```python
def free_memory(memory_required, device, keep_loaded=[]):
    """释放指定大小的显存"""
    cleanup_models_gc()

    # 1. 找出可以卸载的模型
    can_unload = []
    for shift_model in current_loaded_models:
        if shift_model.device == device:
            if shift_model not in keep_loaded:
                can_unload.append(shift_model)

    # 2. 按优先级卸载（优先卸载不常用的）
    for model in sorted(can_unload, key=priority_func):
        free_mem = get_free_memory(device)
        if free_mem > memory_required:
            break
        logging.debug(f"Unloading {model.model.model.__class__.__name__}")
        model.model_unload()

    # 3. 清理缓存
    soft_empty_cache()
```

---

## 数据类型选择

### UNet 数据类型

```python
def unet_dtype(device=None, model_params=0, supported_dtypes=[...]):
    """选择UNet模型的数据类型"""
    # 命令行参数优先
    if args.fp32_unet:
        return torch.float32
    if args.fp16_unet:
        return torch.float16
    if args.bf16_unet:
        return torch.bfloat16
    if args.fp8_e4m3fn_unet:
        return torch.float8_e4m3fn

    # 自动选择
    if should_use_fp16(device=device, model_params=model_params):
        return torch.float16
    if should_use_bf16(device, model_params=model_params):
        return torch.bfloat16

    return torch.float32
```

### 是否使用 FP16

```python
def should_use_fp16(device=None, model_params=0, prioritize_performance=True):
    """判断是否应该使用FP16"""
    if is_device_cpu(device):
        return False

    if args.force_fp16:
        return True

    if FORCE_FP32:
        return False

    # MPS 和 DirectML 使用 FP16
    if is_device_mps(device) or is_directml_enabled():
        return True

    # 检查显存是否足够用 FP32
    if prioritize_performance:
        free_model_memory = maximum_vram_for_weights(device)
        if model_params * 4 > free_model_memory:  # FP32 每参数4字节
            return True

    return True  # 默认使用 FP16
```

---

## 设备选择策略

### UNet 设备

```python
def unet_offload_device():
    """UNet卸载到的设备"""
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()  # 高显存保留在GPU
    else:
        return torch.device("cpu")  # 其他情况卸载到CPU

def unet_inital_load_device(parameters, dtype):
    """UNet初始加载设备"""
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()

    if DISABLE_SMART_MEMORY or vram_state == VRAMState.NO_VRAM:
        return torch.device("cpu")

    # 智能选择：比较GPU和CPU可用内存
    model_size = dtype_size(dtype) * parameters
    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev
```

### 文本编码器设备

```python
def text_encoder_device():
    """文本编码器使用的设备"""
    if args.gpu_only:
        return get_torch_device()
    elif vram_state in [VRAMState.HIGH_VRAM, VRAMState.NORMAL_VRAM]:
        if should_use_fp16(prioritize_performance=False):
            return get_torch_device()
    return torch.device("cpu")
```

### VAE 设备

```python
def vae_device():
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()
```

---

## 注意力机制选择

```python
# xformers（高效注意力）
XFORMERS_IS_AVAILABLE = True  # 如果安装了xformers

# PyTorch 原生注意力
ENABLE_PYTORCH_ATTENTION = False
if is_nvidia() and torch_version >= (2, 0):
    ENABLE_PYTORCH_ATTENTION = True

# 启用 PyTorch 注意力优化
if ENABLE_PYTORCH_ATTENTION:
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

def xformers_enabled():
    """是否启用xformers"""
    if cpu_state != CPUState.GPU:
        return False
    if is_intel_xpu() or is_ascend_npu():
        return False
    return XFORMERS_IS_AVAILABLE

def pytorch_attention_enabled():
    """是否启用PyTorch原生注意力"""
    return ENABLE_PYTORCH_ATTENTION
```

---

## 缓存清理

```python
def soft_empty_cache(force=False):
    """软清理GPU缓存"""
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def unload_all_models():
    """卸载所有模型"""
    free_memory(1e30, get_torch_device())

def cleanup_models_gc():
    """清理已死亡的模型引用"""
    for cur in current_loaded_models:
        if cur.is_dead():
            logging.info("Potential memory leak detected...")
            gc.collect()
            soft_empty_cache()
```

---

## 中断处理

```python
class InterruptProcessingException(Exception):
    """处理中断异常"""
    pass

interrupt_processing = False
interrupt_processing_mutex = threading.RLock()

def interrupt_current_processing(value=True):
    """设置中断标志"""
    global interrupt_processing
    with interrupt_processing_mutex:
        interrupt_processing = value

def processing_interrupted():
    """检查是否已中断"""
    global interrupt_processing
    with interrupt_processing_mutex:
        return interrupt_processing

def throw_exception_if_processing_interrupted():
    """如果已中断，抛出异常"""
    global interrupt_processing
    with interrupt_processing_mutex:
        if interrupt_processing:
            interrupt_processing = False
            raise InterruptProcessingException()
```

**使用场景**：在 `nodes.py` 中每个节点执行前调用：
```python
def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()
```

---

## 异步卸载（CUDA Streams）

```python
NUM_STREAMS = 2  # Nvidia默认启用2个stream

def get_offload_stream(device):
    """获取用于异步卸载的stream"""
    if NUM_STREAMS == 0:
        return None

    if device in STREAMS:
        stream = STREAMS[device][stream_counter]
        stream.wait_stream(current_stream(device))
        return stream
    else:
        # 创建新的stream
        streams = [torch.cuda.Stream(device=device) for _ in range(NUM_STREAMS)]
        STREAMS[device] = streams
        return streams[0]
```

---

## Pinned Memory（锁页内存）

```python
# 锁页内存可以加速CPU到GPU的数据传输
MAX_PINNED_MEMORY = get_total_memory(cpu) * 0.45  # Windows限制50%

def pin_memory(tensor):
    """将张量注册为锁页内存"""
    if MAX_PINNED_MEMORY <= 0:
        return False
    if not tensor.is_contiguous():
        return False

    size = tensor.numel() * tensor.element_size()
    if (TOTAL_PINNED_MEMORY + size) > MAX_PINNED_MEMORY:
        return False

    ptr = tensor.data_ptr()
    torch.cuda.cudart().cudaHostRegister(ptr, size, 1)
    return True
```

---

## 命令行参数影响

| 参数 | 作用 |
|------|------|
| `--cpu` | 强制使用CPU |
| `--lowvram` | 低显存模式 |
| `--novram` | 极低显存模式 |
| `--highvram` / `--gpu-only` | 高显存模式，模型不卸载 |
| `--fp16-unet` | UNet使用FP16 |
| `--fp32-unet` | UNet使用FP32 |
| `--bf16-unet` | UNet使用BF16 |
| `--fp8-e4m3fn-unet` | UNet使用FP8 |
| `--use-pytorch-cross-attention` | 使用PyTorch注意力 |
| `--disable-xformers` | 禁用xformers |
| `--reserve-vram N` | 保留N GB显存给其他应用 |

---

## 显存管理流程图

```
模型加载请求
      │
      ▼
load_models_gpu()
      │
      ├── 1. 计算所需显存
      │
      ├── 2. get_free_memory() 检查可用显存
      │
      ├── 3. free_memory() 释放显存
      │       │
      │       ├── 找出可卸载模型
      │       ├── 按优先级卸载
      │       └── soft_empty_cache()
      │
      └── 4. model_load() 加载模型
              │
              ├── 选择数据类型 (unet_dtype)
              ├── 选择设备 (unet_inital_load_device)
              └── 加入 current_loaded_models
```

---

## 关键设计总结

1. **智能显存管理**：根据显存大小自动选择加载策略
2. **动态模型卸载**：显存不足时自动卸载不常用模型
3. **多平台支持**：NVIDIA/AMD/Intel/Apple/华为/寒武纪
4. **数据类型优化**：自动选择最佳精度（FP32/FP16/BF16/FP8）
5. **异步传输**：使用CUDA Streams加速数据传输
6. **锁页内存**：加速CPU-GPU数据交换
7. **中断机制**：支持随时中断长时间任务

---

## 学习完成

恭喜！你已完成 ComfyUI 核心模块的学习：

1. ✅ `main.py` - 启动流程
2. ✅ `server.py` - Web服务器
3. ✅ `nodes.py` - 节点系统
4. ✅ `execution.py` - 执行引擎
5. ✅ `model_management.py` - 模型管理

**下一步建议**：
- 学习 `comfy/sd.py` - Stable Diffusion 模型加载
- 学习 `comfy/samplers.py` - 采样算法
- 实践：开发一个简单的自定义节点
