# 多说话人语音转写与智能检索系统 —— 项目方案

## 一、项目核心定位

**一句话描述**:构建一个面向影视片段与会议场景的多说话人语音转写系统,集成 Speaker Diarization、ASR、LLM 后处理纠错与多维度智能检索,输出"谁在什么时候说了什么"的结构化结果,并支持自然语言+多维度的检索查询。

**目标场景**:
- 影视片段(中、英两种语言)的角色对话识别
- 多人会议录音的发言人归属与内容转写(使用 AISHELL-4 / AliMeeting / VoxConverse 数据集)
- 处理**轻度噪声 + 背景音乐(BGM)**干扰

**输出格式**:
```
[00:00:12 - 00:00:15] Speaker A: 我觉得这个方案成本太高了
[00:00:16 - 00:00:20] Speaker B: 不一定,我们应该先看效果
[00:00:21 - 00:00:25] Speaker A: 但是预算只有这么多
```

**叠加智能能力**:
- LLM 后处理纠错(N-best 重打分 + 跨说话人语境一致性)
- 多维度复合检索(**语义 / 关键词 / 说话人 / 时间 / 意图** 5 维)

**支持语言**:中文、英文(中文为主战场,英文作为多语言验证)。Whisper-large-v3 本身支持 99 种语言,系统架构可扩展。

**运行环境**:
- 本地:Windows 11 + Anaconda + PyTorch + RTX 4060 8GB(开发调试)
- 云端:Cloud Studio T4 16GB(完整管线 + 评测 + 演示)
- V100 32GB:仅在关键节点使用(机时较贵)

**项目策略**:
- **不做模型微调**(公开 SOTA 模型已足够强)
- **本地开发调试 + 云端全链路运行**:本地用 Whisper-medium,云端切到 Whisper-large-v3
- **效果优先**:选用各模块当前 SOTA 预训练模型
- **核心策略**:SOTA 预训练模型 + 工程组合 + LLM 后处理 + 智能检索

---

## 二、课程契合度论证

| 课程评分维度 | 本项目对应 |
|---|---|
| **课程主题"语音识别"** | ✅ 核心任务就是 ASR + Speaker Diarization |
| **近 2 年课题方向** | ✅ Pyannote 3.1 (2024)、Whisper-large-v3 (2023-2024)、WhisperX (2024)、BGE-M3 (2024)、LLM 后处理 (ICASSP 2025 主题) |
| **方向 1 端到端 Transformer** | ✅ Whisper、Pyannote 均为 Transformer 架构 |
| **方向 5 噪声鲁棒性** | ✅ Demucs 前端 + 噪声/BGM 鲁棒性实验 |
| **方向 6 实时语音转写** | ✅ 流式准实时模式(30 秒块处理) |
| **报告"国内外研究现状"** | ✅ Diarization 演进 + ASR 演进 + 检索演进 + LLM 后处理趋势 |
| **演示"技术深度"** | ✅ 多模型协作 + 时间戳对齐 + 多维度检索复合 |

---

## 三、模型清单

### 3.1 ASR 模型

| 用途 | 模型全称 | 下载地址 | 显存(fp16) |
|---|---|---|---|
| **本地调试** | openai/whisper-medium | https://huggingface.co/openai/whisper-medium | ~3GB |
| **云端运行** | openai/whisper-large-v3 | https://huggingface.co/openai/whisper-large-v3 | ~6GB |

**本地与云端区别**:仅 ASR 模型不同,通过配置文件切换:
```yaml
# configs/models.yaml
asr_model: "openai/whisper-medium"      # 本地
asr_model: "openai/whisper-large-v3"    # 云端
```

### 3.2 Speaker Diarization 模型

| 模型 | 用途 | 下载地址 | 备注 |
|---|---|---|---|
| **pyannote/speaker-diarization-3.1** | 说话人分离主管线 | HF: https://huggingface.co/pyannote/speaker-diarization-3.1<br>ModelScope: https://www.modelscope.cn/models/pyannote/speaker-diarization-3.1 | HF 需同意用户协议(5 分钟搞定);ModelScope 镜像无需授权 |
| **pyannote/segmentation-3.0** | 分段子模型 | HF: https://huggingface.co/pyannote/segmentation-3.0<br>ModelScope: https://www.modelscope.cn/models/pyannote/segmentation-3.0 | 由上面的管线自动调用 |

**本地与云端相同**,显存均约 1-2GB。

### 3.3 WhisperX 强制对齐模型

WhisperX 用于将 Whisper 输出对齐到 word-level 时间戳,按语言加载不同的 wav2vec2 模型。

| 语言 | 模型 | 下载地址 |
|---|---|---|
| **中文** | jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn | https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn |
| **英文** | WAV2VEC2_ASR_BASE_960H | torchaudio 内置,首次使用自动下载 |

**本地与云端相同**,显存均约 1GB。

### 3.4 BGM 人声分离模型

| 模型 | 用途 | 下载方式 |
|---|---|---|
| **Demucs htdemucs_ft** | BGM 人声分离 | pip 安装包,模型首次运行自动下载到缓存目录 |

**安装与使用**:
```bash
pip install demucs
```

**首次运行触发自动下载**:
```bash
# 命令行用法
demucs --two-stems vocals -n htdemucs_ft your_audio.wav

# 模型权重自动下载到:
# Linux/Mac: ~/.cache/torch/hub/checkpoints/htdemucs_ft.th
# Windows:   C:\Users\<用户>\.cache\torch\hub\checkpoints\htdemucs_ft.th
# 大小约 320MB
```


**Python API 调用**:
```python
from demucs.api import Separator
separator = Separator(model="htdemucs_ft")
origin, separated = separator.separate_audio_file("audio.wav")
vocals = separated["vocals"]  # 提取的人声
```

**本地与云端相同**,显存约 2-3GB。

### 3.5 检索 Embedding 模型

| 模型 | 用途 | 下载地址 |
|---|---|---|
| **BAAI/bge-m3** | 检索 embedding(dense + sparse 一体) | HF: https://huggingface.co/BAAI/bge-m3<br>ModelScope: https://www.modelscope.cn/models/AI-ModelScope/bge-m3 |

**本地与云端相同**,显存约 2GB。一个模型同时提供:
- Dense embedding(1024 维)→ 语义检索
- Sparse weights(BM25-like)→ 关键词检索

### 3.6 LLM(API 调用,无需下载)

**API Key 从项目根目录 `.env` 文件加载**

---

## 四、数据集清单

### 4.1 中文会议数据集

#### AISHELL-4
- **全称**:An Open Source Dataset for Speech Enhancement, Separation, Recognition and Speaker Diarization in Conference Scenario
- **规模**:120 小时,211 个会议,4-8 人/会议
- **下载地址**:
  - 主站: https://www.openslr.org/111/
  - 国内镜像: http://www.aishelltech.com/aishell_4
- **本项目实际下载**:**仅评测集**(~5GB),不要训练集

#### AliMeeting
- **全称**:ICASSP 2022 Multi-channel Multi-party Meeting Transcription Challenge (M2MeT) 数据集
- **规模**:118.75 小时,240 个会议,2-4 人/会议
- **下载地址**:https://www.openslr.org/119/
- **本项目实际下载**:
  - **Eval_Ali.tar.gz**(3.42GB)
  - **Test_Ali.tar.gz**(8.90GB)
  - 训练集 96GB 不需要下载
- **优先选远场(far)数据**(更接近真实会议场景)

### 4.2 英文影视/演讲数据集

#### VoxConverse
- **全称**:VoxConverse: A Speaker Diarisation Dataset from Open-source Media
- **规模**:50 小时+,来自 YouTube 影视和演讲
- **下载地址**:
  - https://huggingface.co/datasets/diarizers-community/voxconverse- 
  - VoxConverse 官方主页关了下载入口，使用社区维护的镜像。音频和 RTTM 标注一起打包成 parquet 格式，总大小 7.3GB，包含 dev (216 个样本) 和 test (232 个样本)，时间戳和说话人标注都已对齐。

### 4.3 演示用素材(自己截取)

| 内容 | 数量 | 时长 | 用途 |
|---|---|---|---|
| 中文影视片段 | 2-3 段 | 每段 30 秒 - 1 分钟 | Demo 演示(选 BGM 较弱的对话戏,如《漫长的季节》《狂飙》) |
| 英文影视片段 | 2-3 段 | 每段 30 秒 - 1 分钟 | Demo 演示(英剧/美剧)|

**仅用于课堂演示**,无需作为数据集。

---

## 五、本地 vs 云端的下载与运行划分

### 5.1 本地(4060 8GB)需要下载

```
模型:
✅ openai/whisper-medium                                # ASR 调试用
✅ pyannote/speaker-diarization-3.1
✅ pyannote/segmentation-3.0
✅ jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn
✅ BAAI/bge-m3
✅ Demucs htdemucs_ft(pip 安装后自动)

数据集:
✅ AISHELL-4 测试集中 1-2 个会议(约 200MB)用于单元测试

不下:
❌ whisper-large-v3(本地显存紧张,不需要)
❌ 完整 AISHELL-4 / AliMeeting / VoxConverse 测试集(留到云端跑)
```

**本地总硬盘占用**:约 10GB

### 5.2 云端(Cloud Studio T4 16GB)需要下载

```
模型:
✅ openai/whisper-large-v3                              # ASR 主力
✅ pyannote/speaker-diarization-3.1                     # 同本地
✅ pyannote/segmentation-3.0                            # 同本地
✅ jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn  # 同本地
✅ BAAI/bge-m3                                          # 同本地
✅ Demucs htdemucs_ft(pip 安装后自动)

数据集:
✅ AISHELL-4 完整测试集(~5GB)
✅ AliMeeting Eval+Test(~12GB)
✅ VoxConverse dev+test(~5-10GB)
```

**云端总硬盘占用**:模型 ~15GB + 数据集 ~25GB = **约 40GB**(Cloud Studio 默认 50GB 数据盘够用)

---

## 六、显存预算

### 本地(4060 8GB)

**串行执行,单模块运行**:

| 阶段 | 同时驻留 | 显存 |
|---|---|---|
| BGM 分离 | Demucs | ~2GB |
| Diarization | Pyannote 3.1 + segmentation | ~2GB |
| ASR | Whisper-medium + 中文 wav2vec2 | ~4GB |
| Embedding | BGE-M3 | ~2GB |

**峰值约 4GB**,余量充足。

### 云端(Cloud Studio T4 16GB)

**串行执行 + 主动释放显存**:

| 阶段 | 同时驻留 | 显存 |
|---|---|---|
| BGM 分离 | Demucs | ~3GB |
| Diarization | Pyannote 3.1 + segmentation | ~2GB |
| ASR | Whisper-large-v3 + 中文 wav2vec2 | ~7GB |
| Embedding | BGE-M3 | ~2GB |

**峰值约 8GB**,余量 8GB 充足。

**关键工程实践**:每阶段处理完后 `del model + torch.cuda.empty_cache()` 释放显存,避免全部驻留导致 OOM。

---

## 七、技术栈核心管线

```
输入:影视片段 / 会议录音
    ↓
┌─────────────────────────────────────────────────┐
│  [可选] 前端处理:Demucs htdemucs_ft             │
│  └─ 影视场景去 BGM,会议场景跳过                  │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│  Speaker Diarization:谁在什么时候说              │
│  └─ Pyannote 3.1                                │
│     输出 RTTM:[start, end, speaker_id]          │
└──────────────┬──────────────────────────────────┘
               ↓ 时间戳对齐(VAD AND 策略)
┌─────────────────────────────────────────────────┐
│  ASR:说了什么                                   │
│  └─ WhisperX (Whisper-medium / large-v3 + wav2vec2)│
│     输出:word-level timestamps + N-best         │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│  Diarization × ASR 合并                         │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│  LLM 后处理(创新层 1)                           │
│  ├─ N-best 重打分纠错                           │
│  ├─ 跨说话人语境一致性修正                       │
│  ├─ 说话人身份一致性检查                         │
│  └─ 意图标注(为检索准备)                        │
└──────────────┬──────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────┐
│  多维度复合检索(创新层 2)                       │
│  ├─ 语义检索(BGE-M3 dense embedding)            │
│  ├─ 关键词检索(BGE-M3 sparse / BM25)            │
│  ├─ 说话人过滤(元数据)                          │
│  ├─ 时间范围过滤(元数据)                        │
│  ├─ 意图过滤(LLM 预先标注)                      │
│  └─ Reciprocal Rank Fusion(RRF) 融合多路结果   │
└──────────────┬──────────────────────────────────┘
               ↓
GUI 展示(Gradio):转写结果 + 检索框 + 时间线播放
```

---

## 八、三大创新点

### 创新点 1:多模型协同管线 + 时间戳对齐策略

**问题**:Pyannote 和 Whisper 用不同的 VAD,直接组合会有时间戳漂移,导致"说话人 A 说的话被切到说话人 B 那里"。

**方案**:采用 **WhisperX VAD 与 Pyannote VAD 的逻辑 AND 交集**——把 Pyannote 的聚类结果严格对齐到 WhisperX 使用的 Silero VAD 边界。

### 创新点 2:LLM 增强的多层后处理

- **N-best 重打分纠错**:Whisper 输出 5 个候选,LLM 看说话人身份 + 对话上下文重新选最合理的
- **跨说话人语境一致性修正**:LLM 识别说话人角色(医生/患者、上司/下属),用角色信息反过来修正 ASR 错误
- **说话人身份一致性检查**:LLM 看说话风格判断 Pyannote 是否把同一个人错分成两个 ID
- **意图标注**:LLM 给每个发言打标签(`提问 / 陈述 / 反对 / 同意 / 提议 / 总结 / 闲谈 / 命令`)供检索使用

### 创新点 3:多维度复合检索系统

5 个检索维度(语义 / 关键词 / 说话人 / 时间 / 意图),通过 **Reciprocal Rank Fusion (RRF)** 算法融合多路结果。

| 用户查询 | 系统拆解 |
|---|---|
| "Speaker B 在后半段说过的反对意见" | 说话人=B + 时间=后 50% + 意图=反对 + 语义=反对 |
| "包含'预算'的反对发言" | 关键词=预算 + 意图=反对 |
| "找出所有提问句" | 意图=提问 |
| "类似'方案不可行'的发言" | 语义=方案不可行 |

---

## 九、三层演示效果

### Demo 1:多语言影视片段(BGM 鲁棒性 + 多语言)
中文片段(《漫长的季节》《狂飙》)+ 英文片段(英剧/美剧),展示去 BGM 前后效果对比 + 中英双语识别。

### Demo 2:会议数据集流式准实时
用 AliMeeting / AISHELL-4 测试集片段,演示 30 秒块流式处理,实时滚动出带说话人的字幕。

### Demo 3:多维度智能检索
基于 Demo 2 的转写结果,现场输入复合查询展示语义/关键词/说话人/时间/意图 5 维检索能力。

---

## 十、6 周开发计划

| 周次 | 任务 | 主战场 | 交付物 |
|---|---|---|---|
| **第 1 周** | 本地环境搭建 + Pyannote + Whisper-medium + WhisperX 各自跑通 + 数据集小子集测试 | 本地 | 单一短音频跑通 demo |
| **第 2 周** | 集成 Pyannote + WhisperX + 时间戳对齐(VAD AND 策略)+ 本地小测试集基线测试 | 本地 | 完整管线 + 基线 DER/WER 数据 |
| **第 3 周** | LLM 适配层 + N-best 重打分 + 跨说话人一致性修正 + 意图标注 prompt 设计 | 本地 | LLM 后处理模块 + 意图标注结果 |
| **第 4 周** | 流式准实时模式(30 秒块) + Demucs 集成 + 噪声/BGM 鲁棒性测试 + **首次上云端**(切到 Whisper-large-v3 跑完整管线) | 本地 + 云端首次 | 流式 demo + 噪声鲁棒性实验 |
| **第 5 周** | 多维度复合检索(BGE-M3 + 元数据 + 意图 + RRF) + Gradio GUI + 中英多语言影视片段 demo | 本地 + 云端集成测试 | 完整 GUI + 检索系统 |
| **第 6 周** | 完整数据集评测(云端) + 报告撰写(3000+ 字) + 答辩 PPT + 演示视频 + Live Demo 排练 | 云端 + V100 | 最终交付物 |

**云端机时预算**(总额 50+ 机时,T4 1.2 机时/小时,V100 3.6 机时/小时):

| 阶段 | 设备 | 机时消耗 |
|---|---|---|
| 第 4 周首次上云 | T4 | ~5 机时(4 小时) |
| 第 5 周集成测试 | T4 | ~10 机时(8 小时) |
| 第 6 周完整评测 | T4 | ~15 机时(12 小时) |
| 第 6 周最终 demo | V100 | ~10 机时(2.7 小时,演示稳定性优先)|
| **预留余量** | - | ~10 机时 |

---

## 十一、项目目录结构

```
multi-speaker-asr/
├── .env                            # API Keys(本地存放,不提交 git)
├── .env.example                    # API Key 模板
├── src/
│   ├── frontend/                   # 音频前端处理
│   │   └── demucs_separator.py     # BGM 人声分离
│   ├── diarization/                # 说话人分离
│   │   └── pyannote_pipeline.py
│   ├── asr/                        # 语音识别
│   │   ├── whisperx_pipeline.py
│   │   └── nbest_extractor.py
│   ├── alignment/                  # 时间戳对齐
│   │   └── vad_intersection.py     # VAD AND 策略
│   ├── llm/                        # LLM 适配与后处理
│   │   ├── base.py                 # LLMAdapter 抽象基类
│   │   ├── deepseek.py
│   │   ├── claude.py
│   │   ├── factory.py
│   │   ├── rescoring.py            # N-best 重打分
│   │   ├── consistency.py          # 跨说话人一致性
│   │   └── intent_tagger.py        # 意图标注
│   ├── retrieval/                  # 多维度检索(创新核心)
│   │   ├── embedding.py            # BGE-M3 封装
│   │   ├── semantic_search.py
│   │   ├── keyword_search.py
│   │   ├── metadata_filter.py
│   │   ├── rrf_fusion.py
│   │   └── retriever.py
│   ├── streaming/                  # 流式处理
│   │   └── chunk_processor.py      # 30 秒块处理
│   └── pipeline.py                 # 全链路串联
├── gui/
│   ├── app.py                      # Gradio 主程序
│   └── components/
├── configs/
│   ├── models.yaml                 # 本地/云端模型切换
│   ├── languages.yaml
│   └── retrieval.yaml
├── data/
│   ├── alimeeting/
│   ├── aishell4/
│   ├── voxconverse/
│   └── movie_clips/                # 演示用影视片段
├── experiments/                    # 实验脚本
│   ├── baseline_eval.py
│   ├── alignment_ablation.py
│   ├── llm_correction_eval.py
│   └── noise_robustness.py
├── models/                         # 模型权重缓存(可选,默认走 HF 缓存)
└── requirements.txt
```

---

## 十二、API Key 配置

项目根目录创建 `.env` 文件:
```
# Claude(七牛云中转)
CLAUDE_API_KEY=your_qnaigc_key_here
CLAUDE_BASE_URL=https://api.qnaigc.com/anthropic

# DeepSeek 官方
DEEPSEEK_API_KEY=your_deepseek_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# HuggingFace(Pyannote 授权)
HF_TOKEN=your_huggingface_token

# 国内镜像(可选,加速 HuggingFace 下载)
HF_ENDPOINT=https://hf-mirror.com
```

代码加载:
```python
from dotenv import load_dotenv
import os
load_dotenv()

claude_key = os.getenv("CLAUDE_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
hf_token = os.getenv("HF_TOKEN")
```

`.env` 文件**不提交 git**(在 `.gitignore` 中排除),提供 `.env.example` 作为模板。

---

## 十三、关键依赖

```
# requirements.txt
pyannote.audio==3.1.1
whisperx>=3.1.1
faster-whisper>=1.0.0
demucs>=4.0.0
FlagEmbedding>=1.2.10              # BGE-M3
rank-bm25>=0.2.2
gradio>=5.0.0
python-dotenv>=1.0.0
anthropic>=0.40.0                  # Claude SDK
openai>=1.50.0                     # DeepSeek 兼容 OpenAI 协议
torch>=2.5.0
torchaudio>=2.5.0
transformers>=4.45.0
huggingface-hub>=0.25.0
```

---

## 十四、关键决策记录

| 决策点 | 选择 | 理由 |
|---|---|---|
| 项目方向 | 多说话人转写 + LLM 增强 + 多维检索 | 真实痛点 + 课程契合 + 演示效果 |
| 是否微调 | 不微调 | 公开 SOTA 已足够强 |
| ASR 模型 | medium(本地)/ large-v3(云端) | 效果优先 + 本地调试便利 |
| Diarization | Pyannote 3.1 | 开源最佳平衡 |
| BGM 分离 | Demucs htdemucs_ft | 影视场景必需 |
| Embedding | BGE-M3 | 一模型覆盖 dense + sparse,支持 100+ 语言 |
| LLM 主力 | Claude Sonnet 4.6(七牛云中转) | 角色扮演 + JSON 输出最稳 |
| LLM 兜底 | DeepSeek V4-Flash(官方直连) | 便宜、中文好 |
| 支持语言 | 中文 + 英文 | 聚焦主战场,避免精力分散 |
| 检索维度 | 语义 / 关键词 / 说话人 / 时间 / 意图 | 5 维(不含情感) |
| 检索融合 | Reciprocal Rank Fusion (RRF) | 经典且效果好 |
| 本地策略 | 4060 调试 + Whisper-medium | 不耗云端机时 |
| 云端策略 | T4 跑完整管线 + V100 用于关键节点 | 50 机时精打细算 |
| GUI 框架 | Gradio 5.x | 上传音频 + 多模态展示 |
