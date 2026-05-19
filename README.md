# 多说话人语音转写与智能检索系统

面向影视片段与会议场景的多说话人语音转写系统，集成 Speaker Diarization、ASR、LLM 后处理纠错与多维度智能检索。

📄 **最终报告**：`docs/final_report.md`（3,748 字）
📊 **答辩 PPT 大纲**：`docs/presentation_outline.md`（16 页）

---

## 项目结构

```
E:/ASR/
├── final_plan.md                  ← 完整技术方案与模型选型文档
├── README.md                      ← 本文件
│
├── configs/
│   ├── models.yaml                ← 模型配置（local / cloud 双 profile）
│   ├── languages.yaml             ← 语言配置（zh/en + 意图标签体系）
│   └── retrieval.yaml             ← 检索配置（5维权重、RRF参数）
│
├── src/
│   ├── pipeline.py                ← 全链路编排器（6 阶段）
│   ├── asr/                       ← ASR 后端（当前主用 Transformers Whisper）
│   │   ├── transformers_whisper_backend.py
│   │   └── openai_whisper_backend.py       （备选）
│   ├── diarization/               ← Pyannote 说话人分离（实现位于 __init__.py）
│   ├── alignment/                 ← VAD AND 时间戳对齐策略（实现位于 __init__.py）
│   ├── frontend/                  ← Demucs BGM 人声分离（实现位于 __init__.py）
│   ├── llm/                       ← LLM 三层后处理
│   │   ├── base.py / claude.py / deepseek.py / factory.py  ← 适配器
│   │   ├── corrector.py           ← 上下文 ASR 纠错
│   │   ├── consistency.py         ← 说话人一致性检查（合并/拆分建议）
│   │   ├── intent_tagger.py       ← 意图标注（12 类多标签）
│   │   └── summarizer.py          ← 会议摘要生成
│   ├── retrieval/                 ← 多维复合检索（Week 5）
│   │   ├── embedding.py           ← BGE-M3 编码器（纯 transformers，dense+sparse）
│   │   ├── indexer.py             ← FAISS 索引构建/加载
│   │   ├── retriever.py           ← 5 维检索器 + RRF 融合
│   │   └── __init__.py
│   ├── streaming/                 ← 30 秒块流式处理（实现位于 __init__.py）
│   └── utils/                     ← 配置加载 / 日志 / CUDA DLL 修复
│       ├── config.py
│       ├── cuda_dlls.py
│       └── logger.py
│
├── scripts/
│   ├── check_env.py               ← 环境完整性检查
│   ├── import_test.py             ← 依赖导入测试
│   ├── demo_pipeline.py           ← 端到端管线测试
│   ├── demo_llm.py                ← LLM 后处理测试
│   ├── demo_frontend.py           ← Demucs 分离测试
│   ├── demo_streaming.py          ← 流式处理测试
│   ├── demo_asr.py                ← ASR 冒烟测试
│   ├── demo_retrieval.py          ← 检索端到端测试
│   ├── eval_diarization.py        ← 单文件 DER 评测
│   ├── run_test.py                ← 一键测试（指定文件/目录）
│   └── realtime_mic.py            ← 实时麦克风流式转写
│
├── experiments/
│   ├── comprehensive_eval.py      ← 完整评测驱动（DER+WER/CER+检索+消融）
│   └── generate_report.py         ← 评测数据注入报告脚本
│
├── gui/
│   └── app.py                     ← Gradio GUI（3 标签页：文件转写/实时麦克风/检索）
│
├── docs/
│   ├── final_report.md            ← 最终报告（3,748 字）
│   ├── presentation_outline.md    ← 答辩 PPT 大纲（16 页）
│   └── llm_postprocessing.md      ← LLM 后处理模块说明文档
│
├── outputs/                       ← 评测数据与管线输出
│   ├── eval_baseline_aishell4.json  ← E1 基线评测（20文件）
│   ├── eval_full_aishell4.json      ← E2 完整管线评测（20文件）
│   └── eval_test.json               ← 检索评测（单文件）
│
├── dataset/                       ← 数据集
│   ├── AISHELL-4/test/            ← 20 个中文会议音频 + TextGrid/RTTM
│   └── AIMeeting/Eval_Ali/        ← 8 个 AliMeeting 远场音频 + TextGrid
│
└── models/                        ← 模型权重（本地缓存）
    ├── whisper-medium/
    ├── speaker-diarization-3.1/
    ├── segmentation-3.0/
    ├── pyannote-wespeaker-voxceleb-resnet34-LM/
    ├── wav2vec2-large-xlsr-53-chinese-zh-cn/
    └── bge-m3/                     ← 含 sparse_linear.pt / colbert_linear.pt
```

> **当前代码实现补充**：README 早期版本中提到的 `pyannote_pipeline.py`、`vad_intersection.py`、`demucs_separator.py`、`chunk_processor.py` 等文件名，在当前代码中已收敛到各模块的 `__init__.py`。当前主 ASR 后端为 `src/asr/transformers_whisper_backend.py`，不是 WhisperX 主链路。

---

## 管线流程

```
音频输入 (.wav/.flac/.mp3)
  │
  ├─ [Stage 1: Demucs 人声分离，可选]  → 去 BGM，提取人声轨
  ├─ [Stage 2: Pyannote 3.1 说话人分离] → {start, end, speaker_id}
  ├─ [Stage 3: Transformers Whisper ASR 识别] → {start, end, text}
  ├─ [Stage 4: VAD AND 时间戳对齐]      → {start, end, speaker, text}
  ├─ [GPU 模型释放：del + empty_cache()]
  ├─ [Stage 5: LLM 三层后处理]
  │     ├─ 5a. 上下文纠错 (N-best 重打分 + 同音字修正)
  │     ├─ 5b. 说话人一致性检查 (合并/拆分 SPEAKER_UNKNOWN)
  │     ├─ 5c. 意图标注 (12 类多标签：陈述/提问/同意/反对/提议/总结...)
  │     └─ 5d. 会议摘要生成（可选，失败不阻断主流程）
  │
  └─ [Stage 6: BGE-M3 检索索引构建，可选] → FAISS + sparse weights
       │
       └─ 输出: {segments: [{start, end, speaker, text, intent, ...}]}
```

---

## 运行环境

| 项目 | 值                                           |
|------|---------------------------------------------|
| **OS** | Windows 11                                  |
| **Python** | 3.10.20                                     |
| **GPU** | NVIDIA GeForce RTX 4060 Laptop 8.6GB        |
| **CUDA** | 12.4                                        |
| **PyTorch** | 2.5.1+cu124                                 |
| **transformers** | 4.51.3                                      |
| **运行方式** | Python 直接运行（非 `conda run`）                  |
| **额外依赖** | `sounddevice` (本地麦克风), `websockets` (云端麦克风) |

---

## 快速开始

```bash
conda activate TTS
cd E:/ASR

# 1. 环境检查
python scripts/check_env.py

# 2. 端到端管线（30 秒音频）
python scripts/demo_pipeline.py --seconds 30

# 3. 含 LLM 后处理
python scripts/demo_llm.py --seconds 30

# 4. Demucs 人声分离测试
python scripts/demo_frontend.py

# 5. 流式处理测试
python scripts/demo_streaming.py --seconds 30

# 6. 多维检索测试（先跑管线再检索）
python scripts/demo_retrieval.py --seconds 30

# 7. 一键测试（指定文件/目录）
python scripts/run_test.py --path ./my_audio.wav          # 单文件
python scripts/run_test.py --path ./audio_dir/            # 整个目录
python scripts/run_test.py --path ./dir/ --no-llm         # 跳过 LLM
python scripts/run_test.py --path ./dir/ --retrieval      # 含检索

# 8. Gradio 交互界面（三合一）
python gui/app.py                     # 启动 Web UI（文件转写 + 实时麦克风 + 检索）

# 9. 实时麦克风流式转写（命令行）
python scripts/realtime_mic.py                            # 默认 10s 块
python scripts/realtime_mic.py --chunk 15                 # 15 秒块
python scripts/realtime_mic.py --list-devices             # 列出麦克风设备
python scripts/realtime_mic.py --device 1                 # 指定设备
python scripts/realtime_mic.py --no-llm                   # 仅 ASR（无 LLM 纠错延迟）

# 10. 完整评测（AISHELL-4 20 文件）
python experiments/comprehensive_eval.py                    # 完整管线
python experiments/comprehensive_eval.py --no-llm           # 基线（无LLM）
python experiments/comprehensive_eval.py --retrieval        # 含检索评测
python experiments/comprehensive_eval.py --subset 5         # 仅前5文件快速验证
```

---

## 实时麦克风转写

系统提供两种麦克风转写方式：

| 方式 | 适用场景 | 启动命令 |
|------|---------|---------|
| **GUI 模式** (推荐) | 本地 + 云端通用 | `python gui/app.py` → 选择「🎤 实时麦克风」 |
| **命令行模式** | 仅本地（需物理声卡） | `python scripts/realtime_mic.py` |

### GUI 模式架构（浏览器采集 → WebSocket → 服务器 GPU）

```
用户浏览器                               GPU 服务器
├─ getUserMedia() 麦克风采集              ├─ WebSocket Server :7861
├─ AudioContext 16kHz 单声道             ├─ RealtimePipeline (复用模型)
├─ 每 10s: Float32 → WAV → base64        │   ├─ Diarization + ASR (~7s)
│                                         │   └─ LLM 纠错 (~6s)
├── WebSocket ws://host:7861 ───────────►│
│  ◄── {phase:"asr", segments} ─────────┤
│  ◄── {phase:"llm", segments} ─────────┤
└─ Gradio 轮询 (1.5s) → 界面刷新        └─ _rt_queue → rt_poll()
```

**LLM 原位刷新**：LLM 纠错结果按时间戳匹配对应的 ASR 段，直接在界面上替换原文
- ⚡ = ASR 即刻输出
- ✅ = LLM 已纠错（标注 `←原:...` 显示修正前后对比）

### 命令行模式

```bash
python scripts/realtime_mic.py                  # 默认 10s 块
python scripts/realtime_mic.py --chunk 15       # 15 秒处理块
python scripts/realtime_mic.py --device 0       # 指定麦克风
python scripts/realtime_mic.py --no-llm         # 仅 ASR（无 LLM 纠错）
python scripts/realtime_mic.py --list-devices   # 列出设备
```

---

## 模型清单

| 用途 | 模型 | 显存 |
|------|------|------|
| 前端 BGM 分离 | Demucs htdemucs_ft | ~2GB |
| 说话人分离 | Pyannote speaker-diarization-3.1 + segmentation-3.0 | ~2GB |
| 语音识别 | Whisper-medium (本地) / Whisper-large-v3 (云端) | ~4GB / ~7GB |
| 强制对齐 | wav2vec2-large-xlsr-53-chinese-zh-cn (中文) | ~1GB |
| 检索编码 | BAAI/bge-m3 (dense 1024d + sparse weights) | ~2GB |
| **本地峰值** | (串行执行 + 主动释放) | **~4GB** |

---

## 开发过程与关键问题解决

### Week 1: 环境搭建 + ASR

**问题**: Windows RTX 4060 + CUDA 12.4 环境下，faster-whisper 的 ctranslate2 需要 cuDNN DLL。

**解决**: `src/utils/cuda_dlls.py` 中通过 `os.add_dll_directory()` 注册 NVIDIA pip wheel 的 DLL 路径。同时将 faster-whisper 的导入强制在 pyannote（依赖 onnxruntime）之前，避免 cuDNN 初始化冲突（WinError 1114）。

**问题**: HF pipeline 的 `initial_prompt` 文本泄露到转写输出。

**解决**: `configs/languages.yaml` 中将 `asr_initial_prompt` 设为空字符串，仅靠 `language=zh` 强制中文输出。


### Week 2: Diarization + Alignment

**问题**: Pyannote speaker-diarization-3.1 的 config.yaml 路径解析机制特殊。

**解决**: 使用本地 `.bin` 文件路径替代 HF repo ID。`Pipeline.from_pretrained()` 加载时需 `os.chdir(project_root)` 确保 config.yaml 中相对路径正确解析——这是官方推荐方式而非 hack。

**问题**: 本地 `models/` 下缺少 pyannote-wespeaker-voxceleb-resnet34-LM embedding 模型。

**解决**: 从旧项目 `E:/ASR_TTS/models/diarization/` 复制到 `E:/ASR/models/`。目录名需包含 "pyannote" 前缀以避免 ONNX 加载误判（已知社区 issue #1660）。

**问题**: `librosa` 与 `pyannote` 的 speechbrain 依赖存在 lazy import 冲突。

**解决**: 先 `import librosa` 再导入 pyannote 相关模块。


### Week 3: LLM 后处理

**问题**: 如何设计意图标签体系才能满足多维检索需求。

**解决**: 从 8 类扩展到 12 类多标签（陈述/提问/同意/反对/提议/总结/命令/澄清/确认/打断/寒暄/回应）。每类附带详细语义定义注入 prompt。标签系统设计为列表格式支持一句话多意图（如 `["陈述", "提议"]`）。详见 `docs/llm_postprocessing.md`。

**问题**: DeepSeek 与 Claude 两套 API 协议不同，需要统一接口。

**解决**: `src/llm/base.py` 定义统一的 `LLMAdapter` 抽象接口：`deepseek.py` 走 OpenAI 兼容 SDK，`claude.py` 走 Anthropic SDK（含 system prompt 分离），`factory.py` 根据 `.env` 的 `ACTIVE_LLM` 变量动态创建适配器。


### Week 4: Demucs + 流式处理

**问题**: Demucs 4.0.1 的 `api.Separator` 高层接口在 pip 安装版本中实际不存在。

**解决**: 改用 `demucs.pretrained.get_model()` + `demucs.separate.load_track()` + `apply_model()` 底层 API。需要手动处理单声道→立体声转换和归一化。

**问题**: 流式处理时 pipeline 总是从文件头读取而非 chunk 位置。

**解决**: 给 `pipeline.run()` 新增 `waveform` 和 `sample_rate` 参数，支持内存直传 numpy 数组（float32, mono），完全绕过文件 I/O。


### Week 5: 多维检索 —— FlagEmbedding 兼容性问题（重要）

**背景**: 项目最初使用 `FlagEmbedding` 库（`BGEM3FlagModel`）加载 BGE-M3 模型，同时输出稠密向量和稀疏词权重（BM25-like 关键词检索）。

**问题 1 — dtype 参数兼容性**: FlagEmbedding 1.4.0（PyPI 最新版）与 transformers ≥ 4.51 存在参数名不兼容——FlagEmbedding 内部调用 `AutoModel.from_pretrained(dtype=...)`，但新版 transformers 已将该参数重命名为 `torch_dtype`。导致加载时报 `TypeError: unexpected keyword argument 'dtype'`。

**问题 2 — encode() IndexError**: 即使通过运行时 monkey-patch 修正了 dtype 参数，`BGEM3FlagModel.encode()` 在 tokenizer 处理某些输入时会抛出 `IndexError`（tokenizer 返回的 token IDs 超出模型词表范围）。

**问题 3 — CUDA 上下文冲突**: 管线在 Stage 4 结束后释放 ASR/Diarization 模型后，直接加载 BGE-M3 会触发 CUDA 段错误（segfault）。原因是 torch 的 CUDA 上下文未彻底清理。

**解决方案**:

#### 5a. 放弃 FlagEmbedding，改用纯 transformers 加载 BGE-M3

核心文件：`src/retrieval/embedding.py`（完全重写）。

**稠密向量（Dense）**：
- 使用 `AutoModel.from_pretrained()` 加载 BGE-M3 的 XLMRobertaModel 基座
- 取 CLS token 的隐藏状态 → L2 归一化 → 1024 维稠密向量
- 仅依赖 `transformers`，无需 FlagEmbedding 或 sentence-transformers

**稀疏权重（Sparse）**：
- BGE-M3 的稀疏词权重来自模型目录下的 `sparse_linear.pt` 文件（~3.5KB）
- 这是一个 `nn.Linear(1024, 1)` 层的权重，应用于所有 token 的隐藏状态
- 公式：`token_weight = ReLU(sparse_linear(hidden_state))`
- 过滤特殊 token（`<s>`, `</s>`, `<pad>`）和负权重，同 token 多次出现取最大值
- 词法匹配分数：`Σ(w_q[t] × w_d[t])` for 查询与文档的重叠 token

**关键收益**：
- ✅ **零 FlagEmbedding 依赖**：不加载 FlagEmbedding，不修改任何库源码
- ✅ **零 sentence-transformers 依赖**：纯 `transformers` + `torch` 实现
- ✅ **与现有代码 100% 兼容**：`EncodingResult`、`EmbeddingEncoder`、`compute_lexical_score()` 接口完全不变
- ✅ **`indexer.py` 和 `retriever.py` 无需任何修改**

#### 5b. CUDA 上下文冲突修复

在管线中导入 `src.retrieval` 前必须显式执行：
```python
import gc
gc.collect()
torch.cuda.empty_cache()
```
这确保 ASR 和 Diarization 模型的 CUDA 上下文被彻底释放，避免 BGE-M3 加载时发生段错误。此逻辑已在 `scripts/demo_retrieval.py` 中内置。


### Week 6: 完整评测 + 报告

**问题**: 缺少批量评测脚本，原项目仅有单文件 DER 评测（`scripts/eval_diarization.py`）。

**解决方案**:

#### 6a. 实现完整评测驱动

`experiments/comprehensive_eval.py` 支持：
- 批量评测（AISHELL-4 20 文件 / AliMeeting 8 文件）
- 四组实验配置：基线（无LLM）/ 完整管线 / 含检索 / 消融
- 多种指标：DER（含 collar=0.25s）、WER/CER（via jiwer）、时序分析、检索性能
- JSON 报告输出 + Rich 格式化终端展示

#### 6b. 评测结果摘要

在 AISHELL-4 测试集 20 文件（每文件截取 60 秒）上：

| 指标 | 基线（无LLM） | 完整管线（含LLM） |
|------|:---:|:---:|
| DER (avg) | 45.56% | 47.67% |
| DER (collar=0.25s) | 42.15% | 44.01% |
| CER (avg) | 59.03% | 58.80% |
| 管线耗时 | 12.4s | 16.8s |
| **最佳文件 DER** | **10.74%** | **10.74%** |

检索性能（单文件 5 段文本）：
- 索引构建：0.29s
- 平均查询延迟：38.3ms
- 语义查询：86.3ms
- 说话人/意图过滤：~28ms

#### 6c. 报告生成

`experiments/generate_report.py` 从 `outputs/eval_*.json` 读取评测数据，自动注入 `docs/final_report.md` 的实验结果表格，并统计字数。


## 云端部署（Cloud Studio T4）

### 配置切换

1. 修改 `configs/models.yaml` 中 `profile: cloud`
2. 在 `.env` 中配置 `HF_TOKEN`
3. 模型自动从 HuggingFace Hub 下载
4. Whisper-medium → Whisper-large-v3 自动切换

### Cloud Studio 端口转发

Cloud Studio 需要借助 **VS Code 端口转发** 将服务器本地回环映射到公网：

```
Cloud Studio (内网)              VS Code 端口转发           用户浏览器
├─ GUI :7860 (127.0.0.1)  ────→  公网 URL_A  ────→  http://URL_A
├─ WS  :7861 (127.0.0.1)  ────→  公网 URL_B  ────→  ws://URL_B (JS自动)
```

**操作步骤**：
1. Cloud Studio 终端启动：
   ```bash
   python gui/app.py --host 127.0.0.1 --port 7860
   ```
2. 在 VS Code 中打开「端口」面板（Ctrl+Shift+P → Ports: Focus on Ports View）
3. 添加端口转发：`7860` → 右键「设置端口可见性」→ Public
4. 添加端口转发：`7861` → 右键「设置端口可见性」→ Public
5. 浏览器打开 7860 的公网地址即可使用

> **注意**：WebSocket 端口 (7861) 需要单独转发。如果仅需要文件转写+检索，可用 `--no-ws` 跳过 WebSocket，只需转发 7860 一个端口。

### 仅文件转写模式（只需一个端口）

```bash
python gui/app.py --host 127.0.0.1 --port 7860 --no-ws
```

### 端口说明

| 端口 | 用途 | 必须转发? |
|------|------|----------|
| 7860 | Gradio Web UI | ✅ 是 |
| 7861 | WebSocket 音频流 (实时麦克风) | ✅ 是（文件转写模式不需要） |

---

## API Key 配置

项目根目录创建 `.env` 文件（参考 `.env.example`）：

```bash
# LLM 后端
DEEPSEEK_API_KEY=your_key_here
CLAUDE_API_KEY=your_key_here
CLAUDE_BASE_URL=https://api.qnaigc.com/anthropic
ACTIVE_LLM=deepseek        # deepseek | claude | minimax

# HuggingFace
HF_TOKEN=your_hf_token

# 可选：国内镜像加速
HF_ENDPOINT=https://hf-mirror.com
```

> **安全提示**：不要把真实 API Key 提交到仓库或共享文档中；`.env.example` 应只保留占位符。

---

## 关键文件速查

| 文件 | 用途 |
|------|------|
| `final_plan.md` | 完整项目方案文档 |
| `docs/final_report.md` | 最终报告（3,748 字） |
| `docs/presentation_outline.md` | 答辩 PPT 大纲 |
| `docs/llm_postprocessing.md` | LLM 后处理模块说明 |
| `configs/models.yaml` | 模型配置切换 |
| `src/pipeline.py` | 6 阶段管线编排器 |
| `src/asr/transformers_whisper_backend.py` | 当前主 ASR 后端 |
| `src/diarization/__init__.py` | Pyannote 说话人分离 |
| `src/alignment/__init__.py` | 时间戳对齐策略 |
| `src/retrieval/embedding.py` | 纯 transformers BGE-M3 编码器 |
| `experiments/comprehensive_eval.py` | 完整评测脚本 |
| `outputs/eval_*.json` | 评测结果数据 |
