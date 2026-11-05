# Dify评估框架 - 快速开始指南

## 已完成的功能

✅ **核心组件**
- 配置数据结构（Pydantic模型）
- 配置和数据集加载器
- Dify适配器（支持流式和非流式）
- 评估执行器（集成Ragas）
- 报告生成器（Console、JSON、HTML、CSV）

✅ **性能指标收集**
- 总耗时
- 首Token时间（TTFT）
- Token统计
- 流式延迟

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或
pip install -e .
```

### 2. 配置评估

编辑 `examples/configs/dify_eval.yml`：

```yaml
platform: dify
api_config:
  api_key: "your-dify-api-key"
  base_url: "http://your-dify-api-url/v1"
  timeout: 30

dataset: ./examples/data/dify_test_dataset.json

metrics:
  - faithfulness
  - answer_relevancy

evaluator_llm:
  provider: openai
  model: gpt-4o-mini
  api_key: "your-openai-api-key"

output:
  format: [console, json, html]
  save_path: ./results/

stream: true
```

### 3. 准备数据集

编辑 `examples/data/dify_test_dataset.json`：

```json
[
  {
    "user_input": "你的问题",
    "reference": "参考答案（可选）",
    "reference_contexts": ["上下文1", "上下文2"]
  }
]
```

### 4. 运行评估

```bash
# 使用CLI
python -m agent_evaluator.cli examples/configs/dify_eval.yml

# 或使用测试脚本
python test_full_evaluation.py
```

### 5. 查看结果

报告会保存在 `./results/` 目录：
- `report.json` - JSON格式
- `report.html` - HTML格式（浏览器打开）
- 控制台输出 - 实时显示

## 支持的指标

- `faithfulness` - 忠实度
- `answer_relevancy` / `response_relevancy` - 回答相关性
- `context_precision` - 上下文精确度
- `context_recall` - 上下文召回率
- `answer_correctness` - 答案正确性
- `answer_accuracy` - 答案准确性

## 性能指标说明

评估报告包含以下性能指标：

- **总耗时**: 从请求开始到完成的总时间
- **TTFT (Time To First Token)**: 首Token响应时间
- **Token统计**: 输入/输出Token数量
- **流式延迟**: 每个chunk的平均延迟

## 注意事项

1. **OpenAI API Key**: 评估器需要OpenAI API Key来运行Ragas指标
2. **Dify API**: 确保Dify API可访问且API Key有效
3. **流式模式**: 设置 `stream: true` 可以收集更详细的性能指标
4. **数据集格式**: 支持Ragas标准格式和自定义格式

## 故障排除

如果遇到问题：

1. 检查API密钥是否正确
2. 确认网络连接正常
3. 查看错误日志了解详细信息
4. 确保所有依赖已正确安装

