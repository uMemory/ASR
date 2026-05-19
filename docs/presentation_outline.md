# 多说话人语音转写与智能检索系统 —— 答辩PPT大纲

## Slide 1: 标题页
**标题**: 多说话人语音转写与智能检索系统
**副标题**: 基于 SOTA 预训练模型 + LLM 增强 + 多维复合检索
**作者/日期**: 2026年5月

---

## Slide 2: 问题背景
- 传统 ASR 只能回答"说了什么"，无法回答"谁在什么时候说了什么"
- 真实场景需求：会议转录、影视字幕、庭审记录
- 核心痛点：多说话人 + 背景噪声 + 时间戳对齐 + 智能检索

---

## Slide 3: 系统管线总览
```
[音频输入] → Demucs BGM分离 → Pyannote 3.1 说话人分离
           → WhisperX ASR → VAD AND 时间戳对齐
           → LLM 三层后处理 → BGE-M3 多维检索
           → Gradio GUI 展示
```
- 本地 RTX 4060 8GB / 云端 T4 16GB 双模式
- 串行执行 + 主动释放避免 OOM

---

## Slide 4: 前端处理 + Diarization
- **Demucs htdemucs_ft**: 混合时域/频域音源分离，提取人声
- **Pyannote 3.1**: Segmentation + WeSpeaker Embedding + Agglomerative Clustering
- 输出: `[{start, end, speaker_id}]`
- 显存占用: ~2GB

---

## Slide 5: ASR + 时间戳对齐（创新点1）
- **WhisperX**: Whisper-medium + wav2vec2 强制对齐 → 词级时间戳
- **VAD AND 策略**: Pyannote VAD ∩ WhisperX Silero VAD → 消除跨模块时间戳漂移
- 输出: `[{start, end, speaker, text}]`

---

## Slide 6: LLM 三层后处理（创新点2）
| 层 | 功能 | 示例 |
|----|------|------|
| 纠错 | 上下文纠错 | 同音字\→修正，语义不通\→修正 |
| 一致性 | 说话人合并/拆分 | SPEAKER_UNKNOWN → SPEAKER_00 |
| 意图 | 12类标签 | 陈述/提问/同意/反对/提议/总结/命令... |

- 支持 Claude / DeepSeek / MiniMax 多后端
- 3次API调用/turn，成本 ¥0.002/30s音频

---

## Slide 7: 多维复合检索（创新点3）
```
5 维检索:
  ├─ 语义 (BGE-M3 dense 1024d + FAISS)
  ├─ 关键词 (BGE-M3 sparse weights)
  ├─ 说话人 (精确ID匹配)
  ├─ 时间 (区间重叠+距离衰减)
  └─ 意图 (any-of匹配)
         ↓
  RRF (Reciprocal Rank Fusion) 融合
```
- 查询: "Speaker B 的反对意见" → 自动拆解为 speaker=B + intent=反对

---

## Slide 8: 技术栈与环境
| 模块 | 模型/技术 | 显存 |
|------|----------|------|
| 前端 | Demucs htdemucs_ft | ~2GB |
| Diarization | Pyannote 3.1 + WeSpeaker | ~2GB |
| ASR | Whisper-medium + wav2vec2 | ~4GB |
| LLM | DeepSeek/Claude API | 0 (云端) |
| 检索 | BGE-M3 (dense+sparse) | ~2GB |
| **峰值** | (串行执行) | **~4GB** |

---

## Slide 9: 数据集
| 数据集 | 语言 | 场景 | 测试集 |
|--------|------|------|--------|
| AISHELL-4 | 中文 | 4-8人会议 | 20文件 |
| AliMeeting | 中文 | 2-4人会议 | 8文件 |
| VoxConverse | 英文 | 影视/演讲 | dev+test |

---

## Slide 10: 实验结果 — DER
| 指标 | 值 |
|------|------|
| DER (avg) | 45.56% |
| DER (collar=0.25s) | 42.15% |
| DER 范围 | 10.74% ~ 100% |
| 文件数 | 20 |

- 60秒截断窗口；完整文件评测中
- 最佳文件 DER = 10.74%（2说话人清晰对话）

---

## Slide 11: 实验结果 — 时序与检索
**时序** (60s音频, RTX 4060):
| Stage | 基线 | 含LLM |
|-------|------|-------|
| Diarization | 2.5s | 4.9s |
| ASR | 5.9s | 5.2s |
| LLM | — | 9.6s |
| Total | 12.4s | 23.8s |

**检索**:
- 索引构建: 0.29s (5段)
- 查询延迟: 38ms avg
- 5维复合适用

---

## Slide 12: Demo 1 — 影视 BGM 鲁棒性
- 输入: 含BGM的影视片段（《漫长的季节》/ 英剧）
- 展示: Demucs 分离前后波形对比
- 中英多语言转写 + 说话人标注

---

## Slide 13: Demo 2 — 会议流式 + 检索
- 流式: 30s块处理 → 实时滚动字幕
- 检索: "反对意见" → Speaker_00 的反对发言列表
- Gradio GUI 交互展示

---

## Slide 14: 总结与贡献
1. ✅ 完整的多说话人转写管线（6阶段串联）
2. ✅ VAD AND 时间戳对齐（消除漂移）
3. ✅ LLM 三层后处理（纠错+一致性+意图）
4. ✅ BGE-M3 五维复合检索（RRF融合）
5. ✅ Gradio 交互式 GUI（3大Demo场景）

---

## Slide 15: 局限性与展望
- **实时性**: 30s块 → 逐帧流式（faster-whisper）
- **说话人识别**: 匿名ID → 人物身份绑定
- **检索基准**: 定性 → 标准评测集
- **多语言**: 中英 → 99种语言（Whisper-large-v3）

---

## Slide 16: Q&A
联系方式 / 项目链接 / 感谢
