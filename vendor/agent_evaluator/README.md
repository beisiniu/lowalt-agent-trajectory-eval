# Agent Evaluator

基于 [Ragas](https://docs.ragas.io/) 构建的智能体测评框架，支持对 Dify、Bisheng、n8n、Coze、LangChain、LangGraph 等平台构建的智能体、工作流、RAG 应用进行统一评估。

## 特性

- 🎯 **配置驱动**：通过 YML 文件即可完成评估配置，无需编写代码
- 🔌 **多平台支持**：统一接口适配多个主流智能体平台
- 📊 **丰富的评估指标**：基于 Ragas 提供的 LLM-based 评估指标
- 📈 **多样化报告**：支持控制台、JSON、HTML、CSV 等多种输出格式

## 项目结构

```
agent_evaluator/
├── src/agent_evaluator/       # 核心代码
│   ├── core/                  # 核心数据结构
│   ├── adapters/              # 平台适配器
│   ├── evaluator/             # 评估引擎
│   ├── loaders/               # 配置和数据加载
│   └── reporters/             # 结果输出
├── examples/                  # 示例配置和数据
├── tests/                     # 测试代码
└── schemas/                   # 配置文件 JSON Schema
```

## 快速开始

### 安装

```bash
pip install -e .
```

或使用开发模式：

```bash
pip install -e ".[dev]"
```

### 使用示例

1. 准备配置文件（`config.yml`）：

```yaml
platform: dify
api_config:
  api_key: "your-api-key"
  base_url: "https://api.dify.ai/v1"

dataset: ./examples/data/rag_qa_dataset.json

metrics:
  - faithfulness
  - answer_relevancy

evaluator_llm:
  provider: openai
  model: gpt-4

output:
  format: [console, html]
  save_path: ./results/
```

2. 运行评估：

```bash
agent-eval run config.yml
```

## 开发

### 项目依赖管理

本项目同时提供 `pyproject.toml` 和 `requirements.txt`：
- `pyproject.toml`：现代 Python 项目的标准配置文件（PEP 518/621）
- `requirements.txt`：传统依赖列表，便于快速安装

### 添加新平台适配器

1. 在 `src/agent_evaluator/adapters/` 创建新文件
2. 继承 `PlatformAdapter` 基类
3. 实现 `invoke()` 方法

## License

MIT