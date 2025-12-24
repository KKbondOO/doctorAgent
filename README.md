# 🔥 MedGemma 医学对话助手

基于 **MedGemma** 和 **LangGraph** 构建的智能医学对话助手，支持医学问答、医学图片分析和 RAG 增强检索。

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ 功能特性

- 🤖 **智能医学问答**: 基于 MedGemma 模型提供专业的医学咨询服务
- 🖼️ **医学图片分析**: 支持 X光、CT、MRI 等医学影像的智能分析
- 🔍 **RAG 增强检索**: 基于 Milvus 向量数据库的医学知识库检索
- 💾 **对话持久化**: 使用 PostgreSQL 存储对话历史，支持对话管理
- 📁 **图片存储**: 集成 MinIO 对象存储，安全管理医学图片
- 🎯 **智能路由**: 自动识别问题类型，选择最优处理流程
- 📊 **LangSmith 集成**: 支持对话追踪和性能监控

## 🏗️ 系统架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Gradio UI     │────▶│   LangGraph      │────▶│   MedGemma      │
│  (用户界面)      │     │   (对话流程)      │     │   (医学模型)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
           ┌───────────────────┼───────────────────┐
           ▼                   ▼                   ▼
   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
   │   PostgreSQL  │   │    Milvus     │   │    MinIO      │
   │  (对话存储)    │   │  (向量检索)    │   │  (图片存储)    │
   └───────────────┘   └───────────────┘   └───────────────┘
```

## 📁 项目结构

```
doctorAgent/
├── project/                  # 核心代码
│   ├── medicalAssistant.py   # Gradio 用户界面和主入口
│   ├── bot_graph.py          # LangGraph 对话流程定义
│   └── db.py                 # PostgreSQL 数据库操作
├── milvus/                   # 向量数据库相关
│   └── milvus_import.py      # 医学问答数据导入脚本
├── README.md                 # 项目说明文档
└── LICENSE                   # MIT 开源许可证
```

## 🚀 快速开始

### 环境要求

- Python 3.10+
- NVIDIA GPU (推荐 RTX 4060 或更高)
- Docker & NVIDIA Container Toolkit
- PostgreSQL 数据库
- Milvus 向量数据库
- MinIO 对象存储

### 安装依赖

```bash
pip install langchain langchain-openai langgraph psycopg-pool gradio minio pymilvus ollama numpy pandas tqdm
```

### 🐳 模型部署 (Docker + vLLM)

本项目使用 **vLLM** 在本地 GPU 上部署模型，以下是 Docker 部署命令：

#### 1. 部署 MedGemma 医学模型

```bash
docker run --runtime nvidia --gpus all \
    --name Medgemma-4b-it \
    --network minio1 \
    -v C:\Users\27160\.cache\huggingface:/root/.cache/huggingface \
    --env HUGGING_FACE_HUB_TOKEN=your_hf_token \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model unsloth/medgemma-4b-it-bnb-4bit \
    --tokenizer google/medgemma-4b-it \
    --gpu-memory-utilization=0.85 \
    --max-model-len 4096
```

#### 2. 部署 Qwen3 路由模型

```bash
docker run --runtime nvidia --gpus all \
    --name Qwen3-0.6B \
    -v D:\vllm\qwen3:/root/.cache/huggingface \
    -p 8900:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen3-0.6B \
    --gpu-memory-utilization=0.6 \
    --max-model-len 2048
```

#### 3. 部署 Embedding 模型 (Ollama)

```bash
# 安装 Ollama 后拉取 embedding 模型
ollama pull qwen3-embedding:0.6b
```

### 🔑 环境变量配置

为了安全起见，LangChain API Key 已改为从环境变量中读取。在运行程序前，请先设置环境变量：

**Windows (PowerShell):**
```powershell
$env:LANGCHAIN_API_KEY="你的_LANGCHAIN_API_KEY"
```

**Windows (CMD):**
```cmd
set LANGCHAIN_API_KEY=你的_LANGCHAIN_API_KEY
```

**Linux/macOS:**
```bash
export LANGCHAIN_API_KEY="你的_LANGCHAIN_API_KEY"
```

### 配置数据库

1. **PostgreSQL**: 修改 `project/db.py` 中的 `DB_URI` 连接字符串
2. **Milvus**: 修改 `project/bot_graph.py` 中的 `MILVUS_HOST` 和 `MILVUS_PORT`
3. **MinIO**: 修改 `project/medicalAssistant.py` 中的 MinIO 配置

### 导入医学知识库

```bash
cd milvus
python milvus_import.py
```

### 启动应用

```bash
cd project
python medicalAssistant.py
```

### 访问应用

打开浏览器访问 `http://localhost:7860`

## 🔄 对话流程

```
用户输入 ──▶ 智能路由 ──┬──▶ 普通对话 ──▶ 直接回复
                       │
                       └──▶ RAG模式 ──┬──▶ 文本问答 ──▶ 科室识别 ──▶ 向量检索 ──▶ 增强回复
                                      │
                                      └──▶ 图片分析 ──▶ 图片解读 ──▶ 科室识别 ──▶ 向量检索 ──▶ 综合建议
```

## 📋 支持的科室

| 科室类别 | 包含科室 |
|---------|---------|
| 内科系统 | 内分泌科、呼吸科、消化科、神经科、肾内科、血液科、风湿免疫科 |
| 外科系统 | 心外科、普通外科、泌尿科、神经脑外科、肝胆科、胸外科、血管科 |
| 专科 | 乳腺科、产科、心血管科、感染科、肛肠、肝病科、眼科 |

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 前端界面 | Gradio | 用户交互界面 |
| 对话引擎 | LangGraph | 对话流程编排 |
| 医学模型 | MedGemma-4B (vLLM) | 医学问答和图片分析 |
| 路由模型 | Qwen3-0.6B (vLLM) | 意图识别和标题生成 |
| Embedding | qwen3-embedding (Ollama) | 文本向量化 |
| 向量数据库 | Milvus | 医学知识检索 |
| 关系数据库 | PostgreSQL | 对话历史存储 |
| 对象存储 | MinIO | 医学图片存储 |
| 可观测性 | LangSmith | 对话追踪和监控 |
| GPU 推理 | vLLM + Docker | 本地 GPU 模型部署 |

## 💻 硬件配置参考

| 组件 | 推荐配置 |
|------|---------|
| GPU | NVIDIA RTX 4060 8GB 或更高 |
| 内存 | 16GB+ |
| 存储 | SSD 100GB+ |

## ⚠️ 免责声明

本项目仅供学习和研究使用，**不能替代专业医疗诊断**。如有健康问题，请咨询专业医疗机构。

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📧 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**⭐ 如果这个项目对你有帮助，请给个 Star 支持一下！**
