# ComfyUI 学习笔记 15：前端架构

## 概述

ComfyUI 前端已从内嵌源码分离为**独立的 npm 包**，采用现代化的 Vue 3 技术栈重构。

| 组件 | 说明 |
|------|------|
| `comfyui-frontend-package` | 主前端 UI 包 |
| `comfyui-workflow-templates` | 工作流模板包 |
| `comfyui-embedded-docs` | 内嵌文档包 |

源码仓库：[Comfy-Org/ComfyUI_frontend](https://github.com/Comfy-Org/ComfyUI_frontend)

---

## 技术栈

```
Vue 3 + TypeScript + Vite
├── 状态管理：Pinia
├── 路由：Vue Router
├── 国际化：vue-i18n
├── UI 组件：PrimeVue + Reka UI
├── 节点画布：LiteGraph.js（封装）
├── 3D 渲染：Three.js
├── 图表：Chart.js
├── 终端：xterm.js
└── 测试：Vitest + Playwright
```

---

## 后端前端管理

后端通过 `app/frontend_management.py` 管理前端资源：

```python
# frontend_management.py:201
class FrontendManager:
    CUSTOM_FRONTENDS_ROOT = "web_custom_versions"

    @classmethod
    def default_frontend_path(cls) -> str:
        """获取默认前端路径（从 pip 包）"""
        import comfyui_frontend_package
        return str(importlib.resources.files(comfyui_frontend_package) / "static")

    @classmethod
    def init_frontend(cls, version_string: str) -> str:
        """初始化前端，支持自定义版本"""
        if version_string == DEFAULT_VERSION_STRING:
            check_frontend_version()
            return cls.default_frontend_path()
        # 支持格式：owner/repo@version
        # 例如：Comfy-Org/ComfyUI_frontend@latest
```

### 版本管理

```python
# 从 requirements.txt 读取所需版本
def get_required_frontend_version():
    with open(requirements_path, "r") as f:
        for line in f:
            if line.startswith("comfyui-frontend-package=="):
                return line.split("==")[-1]

# 检查版本是否匹配
def check_frontend_version():
    installed = get_installed_frontend_version()
    required = get_required_frontend_version()
    if parse_version(installed) < parse_version(required):
        # 显示警告
```

### 自定义前端

支持从 GitHub 下载第三方前端：

```python
@dataclass
class FrontEndProvider:
    owner: str  # GitHub 用户名
    repo: str   # 仓库名

    @property
    def release_url(self) -> str:
        return f"https://api.github.com/repos/{self.owner}/{self.repo}/releases"

    def get_release(self, version: str) -> Release:
        if version == "latest":
            return self.latest_release
        elif version == "prerelease":
            return self.latest_prerelease
        # 或指定版本号
```

启动参数：
```bash
python main.py --front-end-version Comfy-Org/ComfyUI_frontend@latest
python main.py --front-end-version Comfy-Org/ComfyUI_frontend@v1.2.3
python main.py --front-end-root /path/to/custom/frontend
```

---

## 静态路由

`server.py` 中配置的静态资源路由：

```python
# server.py:852
def add_routes(self):
    # 自定义节点的 Web 扩展
    for name, dir in nodes.EXTENSION_WEB_DIRS.items():
        self.app.add_routes([web.static('/extensions/' + name, dir)])

    # 工作流模板
    self.app.add_routes([
        web.static('/templates', workflow_templates_path)
    ])

    # 内嵌文档
    self.app.add_routes([
        web.static('/docs', embedded_docs_path)
    ])

    # 主前端（最后注册，作为 fallback）
    self.app.add_routes([
        web.static('/', self.web_root),
    ])
```

### API 前缀

所有路由自动添加 `/api` 前缀，便于前端开发服务器代理：

```python
# server.py:864
api_routes = web.RouteTableDef()
for route in self.routes:
    if isinstance(route, web.RouteDef):
        api_routes.route(route.method, "/api" + route.path)(route.handler)
```

访问方式：
- `/ws` 或 `/api/ws` - WebSocket
- `/prompt` 或 `/api/prompt` - 提交工作流
- `/queue` 或 `/api/queue` - 队列操作

---

## 前端目录结构

```
src/
├── App.vue              # 根组件
├── main.ts              # 入口
├── router.ts            # 路由配置
├── i18n.ts              # 国际化
│
├── core/                # 核心模块
│   ├── graph/           # 图形系统
│   │   ├── subgraph/    # 子图（组）
│   │   └── widgets/     # 节点小部件
│   └── schemas/         # 数据结构定义
│
├── services/            # 服务层（19个服务）
│   ├── litegraphService.ts    # LiteGraph 封装（31KB，核心）
│   ├── nodeSearchService.ts   # 节点搜索
│   ├── extensionService.ts    # 扩展管理
│   ├── keybindingService.ts   # 快捷键
│   ├── dialogService.ts       # 对话框
│   ├── audioService.ts        # 音频
│   ├── load3dService.ts       # 3D 加载
│   ├── subgraphService.ts     # 子图
│   └── gateway/               # API 网关
│
├── stores/              # Pinia 状态仓库（38个）
│   ├── nodeDefStore.ts        # 节点定义
│   ├── executionStore.ts      # 执行状态
│   ├── queueStore.ts          # 队列
│   ├── modelStore.ts          # 模型
│   ├── workspaceStore.ts      # 工作区
│   ├── keybindingStore.ts     # 快捷键
│   └── ...
│
├── components/          # Vue 组件
├── views/               # 页面视图
├── composables/         # 组合式函数
├── renderer/            # 渲染器
│   ├── core/
│   ├── extensions/
│   └── utils/
│
├── extensions/          # 前端扩展点
├── locales/             # 多语言文件
└── platform/            # 平台适配（Web/Electron）
```

---

## 服务层架构

### 设计原则

1. **领域驱动** - 每个服务专注特定功能
2. **尽量无状态** - 减少内部状态
3. **可复用** - 跨组件共享
4. **可测试** - 易于单元测试
5. **隔离** - 明确边界和依赖

### 分层结构

```
UI Components（展示）
      ↓
Composables（组合逻辑）
      ↓
Services（业务逻辑）
      ↓
Stores / External APIs（状态/外部）
```

### 服务模式

| 模式 | 数量 | 特点 | 示例 |
|------|------|------|------|
| 类（Class） | 4 | 复杂数据结构、昂贵初始化 | NodeSearchService |
| 组合式（Composable） | 18+ | Vue 响应式集成 | useLitegraphService |
| 引导（Bootstrap） | 1 | 一次性初始化 | - |
| 共享状态（Shared） | 1 | 模块级单例 | - |

---

## LiteGraph 服务

`litegraphService.ts` 是连接 Vue 和 LiteGraph.js 的桥梁：

```typescript
// 组合式服务模式
export function useLitegraphService() {
    // 内部辅助方法
    function addInputSocket() { /* ... */ }
    function addInputWidget() { /* ... */ }
    function setupStrokeStyles() { /* ... */ }

    // 节点注册
    function registerNodeDef(nodeData: ComfyNodeDef) {
        // 创建动态类继承 LGraphNode
        class ComfyNode extends LGraphNode {
            configure(info) { /* 反序列化 */ }
            onDrawBackground() { /* 预览渲染 */ }
        }
        LiteGraph.registerNodeType(nodeData.name, ComfyNode)
    }

    // 导出公共 API
    return {
        registerNodeDef,
        registerSubgraphNodeDef,
        addNodeOnGraph,
        addNodeInput,
        getCanvasCenter,
        goToNode,
        resetView,
        fitView,
        updatePreviews
    }
}
```

### 核心功能

1. **节点定义注册** - 将后端节点定义转换为 LiteGraph 节点类
2. **输入/输出管理** - 处理节点的连接点和小部件
3. **视觉样式** - 运行状态、错误、拖放指示器
4. **上下文菜单** - 图像操作、旁路、剪贴板
5. **画布操作** - 缩放、居中、定位节点

---

## WebSocket 通信

前端通过 WebSocket 与后端实时通信：

```javascript
// 连接
const ws = new WebSocket('ws://localhost:8188/ws?clientId=xxx')

// 接收消息类型
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data)
    switch(msg.type) {
        case 'status':      // 队列状态
        case 'executing':   // 当前执行节点
        case 'progress':    // 进度更新
        case 'executed':    // 节点完成
        case 'execution_error':  // 执行错误
    }
}

// 二进制消息（预览图像）
// 前 4 字节：事件类型
// 剩余：图像数据
```

### Feature Flags 协商

```javascript
// 首条消息：特性协商
ws.send(JSON.stringify({
    type: 'feature_flags',
    data: { /* 客户端支持的特性 */ }
}))

// 服务器响应
// { type: 'feature_flags', data: { /* 服务器特性 */ } }
```

---

## 扩展系统

### 前端扩展

自定义节点可以添加前端扩展：

```javascript
// custom_nodes/my_node/web/extension.js
import { app } from "../../scripts/app.js"

app.registerExtension({
    name: "MyExtension",

    async setup() {
        // 初始化
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 修改节点定义
    },

    nodeCreated(node) {
        // 节点创建后
    }
})
```

### 扩展加载

```python
# server.py:325
@routes.get("/extensions")
async def get_extensions(request):
    # 扫描 web/extensions/**/*.js
    files = glob.glob(os.path.join(self.web_root, 'extensions/**/*.js'))

    # 加上自定义节点的扩展
    for name, dir in nodes.EXTENSION_WEB_DIRS.items():
        files.extend(glob.glob(os.path.join(dir, '**/*.js')))

    return web.json_response(extensions)
```

---

## 状态管理（Pinia）

### 节点定义 Store

```typescript
// stores/nodeDefStore.ts
export const useNodeDefStore = defineStore('nodeDef', () => {
    const nodeDefs = ref<Record<string, ComfyNodeDef>>({})

    async function loadNodeDefs() {
        const response = await api.getNodeDefs()
        nodeDefs.value = response
    }

    function getNodeDef(name: string) {
        return nodeDefs.value[name]
    }

    return { nodeDefs, loadNodeDefs, getNodeDef }
})
```

### 执行 Store

```typescript
// stores/executionStore.ts
export const useExecutionStore = defineStore('execution', () => {
    const isExecuting = ref(false)
    const currentNode = ref<string | null>(null)
    const progress = ref(0)

    function onExecuting(nodeId: string) {
        isExecuting.value = true
        currentNode.value = nodeId
    }

    function onProgress(value: number, max: number) {
        progress.value = value / max
    }

    return { isExecuting, currentNode, progress, onExecuting, onProgress }
})
```

---

## 开发指南

### 本地开发

```bash
# 克隆前端仓库
git clone https://github.com/Comfy-Org/ComfyUI_frontend.git
cd ComfyUI_frontend

# 安装依赖
pnpm install

# 启动开发服务器
pnpm dev
```

### 代理配置

开发服务器代理 API 到后端：

```typescript
// vite.config.mts
export default defineConfig({
    server: {
        proxy: {
            '/api': 'http://localhost:8188',
            '/ws': {
                target: 'ws://localhost:8188',
                ws: true
            }
        }
    }
})
```

### 代码规范

- 使用 `es-toolkit` 代替 lodash
- 遵循 Vue 3 Composition API
- 所有用户文本使用 `vue-i18n`
- 避免 `@ts-expect-error`
- HTML 使用 DOMPurify 清理

---

## 与后端交互流程

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (Vue)                            │
├─────────────────────────────────────────────────────────────┤
│  nodeDefStore ←── GET /object_info ───→ 节点定义            │
│  workspaceStore ←── GET /api/workflow ───→ 工作流           │
│  queueStore ←── POST /prompt ───→ 提交执行                  │
│  executionStore ←── WebSocket ───→ 实时状态                 │
│  modelStore ←── GET /models/{type} ───→ 模型列表            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     后端 (aiohttp)                           │
├─────────────────────────────────────────────────────────────┤
│  PromptServer                                                │
│  ├── routes (REST API)                                       │
│  ├── sockets (WebSocket)                                     │
│  └── prompt_queue → PromptExecutor                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 总结

| 方面 | 说明 |
|------|------|
| 架构 | Vue 3 + Pinia + LiteGraph.js |
| 打包 | 独立 pip 包，支持自定义版本 |
| 通信 | REST API + WebSocket 双通道 |
| 扩展 | 支持自定义节点添加前端扩展 |
| 开发 | Vite 热重载，代理后端 API |

前端架构清晰，服务层抽象良好，状态管理规范。LiteGraph.js 负责节点画布渲染，Vue 负责 UI 和状态管理，两者通过服务层桥接。
