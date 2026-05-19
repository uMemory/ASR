# LLM 后处理模块说明文档

## 概述

本项目在 ASR（语音识别）+ Speaker Diarization（说话人分离）管线之后，引入 LLM（大语言模型）进行三层后处理增强，解决纯声学模型无法处理的语义层问题。

## 架构

```
Stage 1-4 (GPU 管线)
    │
    ├── Diarization:  谁在什么时候说  → {start, end, speaker}
    ├── ASR:          说了什么        → {start, end, text}
    └── Alignment:    时间戳对齐      → {start, end, speaker, text}
            │
            ▼
Stage 5: LLM 后处理 (CPU / API)
    │
    ├── 5a. 上下文纠错  → 修正 ASR 错误
    ├── 5b. 一致性检查  → 修复说话人标注
    └── 5c. 意图标注    → 对话行为分类
            │
            ▼
Stage 6: 检索索引构建 (Week 5)
```

## 三层后处理详解

### 5a. 上下文纠错 (`src/llm/corrector.py`)

**目的**：利用对话上下文修正 ASR 的典型错误（同音字混淆、语义不通、上下文矛盾）。

**策略**：
- 将转写结果按 15 段/批发送给 LLM
- LLM 阅读完整对话上下文后，逐句判断是否有错误
- 只修正 `text` 字段，不改时间戳和说话人
- 如果某句无明显错误，保持原样

**Prompt 设计要点**：
- System prompt 明确角色："语音识别后处理助手"
- 规定修正原则（不改变说话风格、不增删句子）
- 要求 JSON 输出，含 `index`、`text`、`note` 字段

**适用场景**：
- 中文同音字混淆（"计划/计画" → "计划"）
- 上下文矛盾（前面说"同意"，后面说"我也同意"但 ASR 识别为"我也统一"）
- 专有名词纠正

### 5b. 说话人一致性检查 (`src/llm/consistency.py`)

**目的**：检测并修复 Speaker Diarization 算法的两类典型错误：
1. **过分割**（同一人被分配多个 Speaker ID）→ 提出合并建议
2. **欠分割**（不同人被合并到同一 ID）→ 提出拆分建议

**策略**：
- LLM 分析每位说话人的所有发言，比较说话风格、用词习惯
- 输出 JSON 格式的合并/拆分建议，含理由和置信度
- 程序自动执行高置信度合并（不自动拆分，避免引入新错误）

**实际效果示例**（AISHELL-4 测试）：
```
输入: SPEAKER_UNKNOWN 和 SPEAKER_05 分别标注了连续报数的片段
LLM: "两者都在报数字编号，格式完全一致，应为同一人在列举"
结果: SPEAKER_UNKNOWN → SPEAKER_05 (合并)
```

### 5c. 意图标注 (`src/llm/intent_tagger.py`)

**目的**：为每个发言标注对话行为意图，支持后续多维检索中的"意图"维度过滤。

## 意图标签体系（12 类）

### 分类维度

| 维度 | 包含标签 | 说明 |
|------|---------|------|
| **内容型** | 陈述、提问、提议 | 话语的核心内容功能 |
| **立场型** | 同意、反对 | 对他人观点的态度 |
| **元对话** | 总结、澄清、确认 | 对话管理和信息校验 |
| **话轮管理** | 打断 | 对话流程控制 |
| **社交型** | 命令、寒暄、回应 | 人际关系和场景礼仪 |

### 中文标签详解

| 序号 | 标签 | 含义 | 典型示例 |
|------|------|------|---------|
| 1 | **陈述** | 陈述事实、观点、信息（最常见的意图） | "今天把各位都叫过来啊"、"这个方案成本太高了" |
| 2 | **提问** | 提出问题，向他人询问 | "你觉得怎么样？"、"为什么这么做？" |
| 3 | **同意** | 表示赞同、认可 | "我同意"、"这个思路没问题"、"确实如此" |
| 4 | **反对** | 表示不同意、否定、质疑 | "我不同意"、"这不对吧？"、"但是预算只有这么多" |
| 5 | **提议** | 主动提出建议、方案 | "咱们讨论一下那个小区吧"、"我觉得可以先试试" |
| 6 | **总结** | 归纳讨论、做出结论 | "总结一下，我们今天确定了三件事"、"那就这么定了" |
| 7 | **命令** | 发出指令、要求、安排 | "你去联系一下供应商"、"明天之前交报告" |
| 8 | **澄清** | 追问细节、要求进一步解释 | "你刚才说的那个数字是多少？"、"能具体说说吗？" |
| 9 | **确认** | 确认收到的信息 | "对吗？"、"明白了"、"收到了"、"好的我知道了" |
| 10 | **打断** | 打断别人、插话 | "等一下——"、"不是，你听我说" |
| 11 | **寒暄** | 开场白、问候、闲聊 | "大家好"、"今天天气不错"、"辛苦了" |
| 12 | **回应** | 简短回应词，无实质内容 | "嗯"、"好"、"对"、"哦"、"是" |

### 英文标签

| Label | Description |
|-------|-------------|
| statement | Stating a fact, opinion, or information |
| question | Asking a question |
| agree | Expressing agreement |
| disagree | Expressing disagreement or challenge |
| proposal | Suggesting an idea or plan |
| summary | Summarizing or concluding |
| command | Issuing an instruction or demand |
| clarification | Requesting further explanation |
| confirmation | Confirming received information |
| interruption | Cutting someone off or interjecting |
| smalltalk | Greetings, pleasantries, off-topic chat |
| acknowledgment | Brief backchannel ("mm", "okay") |

### 设计原则

1. **多标签支持**：一句话可以有多个意图。`"intent"` 字段为列表，如 `["陈述"]`、`["同意", "反对"]`
2. **互斥性**：同一句中通常不重复标注相似维度的标签（如不同时标"陈述"和"回应"），但跨维度可共存
3. **语境优先**：不只看表面词汇，结合对话上下文判断
4. **实用导向**：标签设计服务于检索需求——支持"找出所有反对意见"、"列出所有提议"等查询
5. **可扩展**：通过 `configs/languages.yaml` 的 `intent_labels` 字段可自定义标签集

## 检索应用

意图标签是 5 维复合检索（语义/关键词/说话人/时间/意图）中的关键维度：

```
用户输入                               系统拆解
"Speaker B 的反对意见"           → 说话人=B + 意图=反对
"包含'预算'的反对发言"           → 关键词=预算 + 意图=反对
"找出所有提问"                   → 意图=提问
"后半段的总结和提议"             → 时间=后50% + 意图=总结,提议
```

## LLM 后端配置

### 支持的 LLM

| 后端 | 适配器文件 | API 协议 | 默认模型 |
|------|-----------|---------|---------|
| DeepSeek | `deepseek.py` | OpenAI 兼容 | deepseek-chat |
| Claude | `claude.py` | Anthropic SDK | claude-sonnet-4-20250514 |
| MiniMax | `factory.py` (复用 DeepSeek) | OpenAI 兼容 | minimax-text-01 |

### 切换后端

编辑 `.env` 文件中的 `ACTIVE_LLM` 变量：

```bash
# 使用 DeepSeek（默认）
ACTIVE_LLM=deepseek

# 使用 Claude
ACTIVE_LLM=claude

# 使用 MiniMax
ACTIVE_LLM=minimax
```

### 开关各阶段

在 `configs/models.yaml` 中独立控制：

```yaml
llm:
  enabled: true           # 总开关
  correction: true        # 上下文纠错
  consistency: true       # 说话人一致性
  intent_tagging: true    # 意图标注
```

或在命令行中通过 `demo_llm.py` 参数控制：

```bash
python scripts/demo_llm.py --no-correction   # 跳过纠错
python scripts/demo_llm.py --no-consistency  # 跳过一致性
python scripts/demo_llm.py --no-intent       # 跳过意图标注
python scripts/demo_llm.py --no-llm          # 完全跳过 LLM（基线）
```

## API 用量估算

以 30 秒会议片段为例（~8 个发言段）：

| 阶段 | API 调用 | Token 估算 |
|------|---------|-----------|
| 纠错 | 1 次 | ~500 input + ~200 output |
| 一致性 | 1 次 | ~600 input + ~300 output |
| 意图标注 | 1 次 | ~500 input + ~200 output |
| **合计** | **3 次** | **~2300 tokens** |

以 DeepSeek 当前价格（¥1/百万 tokens），30 秒音频的 LLM 成本约 **¥0.002**。

## 扩展指南

### 自定义意图标签

1. 修改 `configs/languages.yaml` 中的 `intent_labels` 列表
2. 在 `src/llm/intent_tagger.py` 中更新 `_ZH_LABEL_DEFS` / `_EN_LABEL_DEFS` 字典
3. Prompt 模板会自动注入新的标签定义

### 添加新的 LLM 后端

1. 在 `src/llm/` 下创建 `your_backend.py`
2. 继承 `LLMAdapter` 基类，实现 `chat()` 和 `model_name`
3. 在 `factory.py` 的 `load_llm_adapter()` 中添加分支
