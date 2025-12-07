# ComfyUI Desktop 代理和镜像设置

## 配置文件位置

```
Z:\user\default\comfy.settings.json
```

（Z: 是你的 ComfyUI Desktop 数据目录）

---

## 镜像配置项

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `Comfy-Desktop.UV.PypiInstallMirror` | Python 包镜像 | `https://mirrors.aliyun.com/pypi/simple/` |
| `Comfy-Desktop.UV.PythonInstallMirror` | Python 安装镜像 | `https://mirrors.huaweicloud.com/python/` |
| `Comfy-Desktop.UV.TorchInstallMirror` | PyTorch 镜像 | `https://download.pytorch.org/whl/cu129` |
| `Comfy-Desktop.UV.HuggingFaceMirror` | HuggingFace 模型镜像 | `https://hf-mirror.com` |

---

## 完整配置示例

```json
{
    "Comfy-Desktop.AutoUpdate": true,
    "Comfy-Desktop.SendStatistics": true,
    "Comfy.ColorPalette": "dark",
    "Comfy.UseNewMenu": "Top",
    "Comfy.Workflow.WorkflowTabsPosition": "Topbar",
    "Comfy.Workflow.ShowMissingModelsWarning": true,
    "Comfy.Server.LaunchArgs": {},
    "Comfy-Desktop.UV.PythonInstallMirror": "https://mirrors.huaweicloud.com/python/",
    "Comfy-Desktop.UV.PypiInstallMirror": "https://mirrors.aliyun.com/pypi/simple/",
    "Comfy-Desktop.UV.TorchInstallMirror": "https://download.pytorch.org/whl/cu129",
    "Comfy-Desktop.UV.HuggingFaceMirror": "https://hf-mirror.com",
    "Comfy.InstalledVersion": "1.32.10",
    "Comfy.TutorialCompleted": true,
    "Comfy.Release.Version": "0.3.76",
    "Comfy.Release.Status": "what's new seen",
    "Comfy.Release.Timestamp": 1765095498291
}
```

---

## 国内镜像源汇总

### PyPI 镜像（Python 包）

| 名称 | 地址 |
|------|------|
| 阿里云 | `https://mirrors.aliyun.com/pypi/simple/` |
| 清华 | `https://pypi.tuna.tsinghua.edu.cn/simple/` |
| 中科大 | `https://pypi.mirrors.ustc.edu.cn/simple/` |
| 豆瓣 | `https://pypi.doubanio.com/simple/` |

### HuggingFace 镜像（模型下载）

| 名称 | 地址 |
|------|------|
| hf-mirror | `https://hf-mirror.com` |

---

## 修改步骤

1. **关闭 ComfyUI Desktop**

2. **备份配置文件**
   ```
   复制 comfy.settings.json 为 comfy.settings.json.backup
   ```

3. **编辑配置文件**
   - 用文本编辑器打开 `Z:\user\default\comfy.settings.json`
   - 添加或修改镜像配置项

4. **重启 ComfyUI Desktop**

---

## 如果还是很慢

### 方法1：设置系统环境变量

1. 右键 **此电脑** → **属性** → **高级系统设置** → **环境变量**
2. 新建系统变量：

| 变量名 | 值 |
|--------|-----|
| `HF_ENDPOINT` | `https://hf-mirror.com` |
| `HTTP_PROXY` | `http://127.0.0.1:7890` (如果有代理) |
| `HTTPS_PROXY` | `http://127.0.0.1:7890` (如果有代理) |

### 方法2：手动下载模型

从以下网站下载模型后放到 `Z:\models\` 对应目录：

- **Civitai**: https://civitai.com
- **HuggingFace 镜像**: https://hf-mirror.com
- **LibLib**: https://www.liblib.art

模型目录结构：
```
Z:\models\
├── checkpoints\    # 主模型 (.safetensors, .ckpt)
├── loras\          # LoRA 模型
├── vae\            # VAE 模型
├── clip\           # CLIP 模型
├── controlnet\     # ControlNet 模型
└── embeddings\     # Embedding/Textual Inversion
```

---

## 备份文件位置

```
Z:\user\default\comfy.settings.json.backup
```

如果配置出错，可以用备份文件恢复。
