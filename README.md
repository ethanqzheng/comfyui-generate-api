# comfyui-generate-api

一个基于ComfyUI的高性能文生图API服务，支持动态workflow加载和RESTful接口调用。

### 在 ComfyUI 中作为 custom node 使用
- 将整个目录放置在 `ComfyUI/custom_nodes/`目录下
- 默认使用`comfyui-generate-api`插件内置的 `workflows` 目录 `ComfyUI/custom_nodes/comfyui-generate-api/workflows/`

```bash
# 进入ComfyUI自定义节点
cd ComfyUI/custom_nodes

# 克隆项目
git clone https://github.com/ethanqzheng/comfyui-generate-api.git
cd comfyui-generate-api

# 安装依赖
pip install -r requirements.txt
```


## API接口

- `POST /api/generate`：提交 workflow，并返回 `prompt_id`
- `GET /api/generate/{prompt_id}`：查询任务状态及输出
- `POST /api/upload_files`：将图片/视频上传到 ComfyUI input 目录
- `GET /api/workflows`：列出所有可用 workflow（名称/类型等关键信息）

## 项目结构

```
comfyui-generate-api/
├── __init__.py           # Custom node 入口
├── server_setup.py       # 路由注册
├── api/                  # API 相关
│   ├── helpers.py        # 输出 URL 提取等工具函数
│   └── types/            # 数据模型定义
│       ├── generate.py   # 生成请求/响应模型
│       └── workflow.py   # Workflow 模型
├── core/                 # 核心功能
│   ├── config.py         # 配置管理
│   ├── executor.py       # ComfyUI 执行器
│   ├── workflow_loader.py    # Workflow 加载器
│   └── workflow_analyzer.py  # Workflow 分析器
├── utils/                # 工具函数
│   ├── progress.py        # 进度日志
│   └── progress_tracker.py   # 进度跟踪
├── workflows/            # Workflow JSON文件目录
│   ├── text2img_basic.json
│   ├── img2img_basic.json
│   └── ...
├── README.md
└── requirements.txt
```
