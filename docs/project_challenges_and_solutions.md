# 项目挑战与解决方案记录

本文记录项目开发过程中遇到的主要困难、根因分析、修复办法和参考资料。重点覆盖转写文本与音频对齐、多说话人分离、ASR 幻觉、短尾词缺失、GUI 人工修正、检索过滤等问题。

## 1. 转写文本与音频不对齐

### 现象

早期浏览器输出中，文本时间戳和实际播放音频严重不匹配：

```text
▸ ⚡ [00:00.0 - 00:02.5] Speaker C: 今天我们公司新出了一款手机产品进行一下这个研讨会...
▸ ⚡ [00:02.5 - 00:23.9] Speaker C: 对因为他可能是去的地方...
```

实际音频里第一句大约从 6-7 秒才开始，但系统把大量文本压到 0-2 秒。另一些 `.flac` 数据中还出现文本被压到 1-2 秒内、点击播放完全对不上原音频的问题。

### 根因

旧管线主要依赖 segment-level 对齐：

```text
Whisper ASR segment
+ Pyannote diarization segment
-> 按重叠比例分配 speaker
```

如果 ASR segment 本身时间戳错误或跨度过大，后续 speaker alignment 只能在错误边界上做分配，无法恢复真实词级边界。

此外，Hugging Face transformers Whisper pipeline 的 chunk timestamp 在长音频、中文会议和 `.flac` 数据上不够稳定，容易出现漂移、重复和静音幻觉。

### 修复

当前离线文件转写采用 openai-whisper sequential long-form 解码作为 ASR 主时钟：

- `configs/models.yaml`: 新增 `offline_backend: openai-whisper`
- `src/pipeline.py`: 文件转写优先使用 `offline_backend`
- `src/asr/openai_whisper_backend.py`: 启用 `condition_on_previous_text=False` 和 `hallucination_silence_threshold=2.0`

离线文件不再使用 transformers chunk timestamp 作为主时间轴。实时路径仍保留 transformers，以保证实时速度。

### 验证

`L_R003S01C02.flac` 前 120 秒，未开 LLM，修复后输出：

```text
0.000-6.000 SPEAKER_00: 零零六
6.000-9.000 SPEAKER_00: 零一三
9.000-10.000 SPEAKER_00: 零一四
10.000-13.000 SPEAKER_00: 零一九
13.000-16.000 SPEAKER_00: 零一六
16.000-23.000 SPEAKER_00: 零一八
23.000-25.000 SPEAKER_00: 今天把各位都叫过来啊
```

之前的 `比喻嗎`、正文压缩到 1-2 秒、长句错位等问题消失。

## 2. 时间跨度过大

### 现象

早期输出经常出现 40-60 秒的单段：

```text
[00:02.5 - 01:06.0] Speaker B: ...
```

点击播放时，一个段落覆盖多句话甚至多个说话人。

### 根因

Whisper 的 segment 边界较粗，且 segment-level speaker assignment 无法在一个长 ASR segment 内细分说话人。

### 修复

Stage 4 改为优先使用 WhisperX forced alignment：

```text
ASR text
-> WhisperX forced alignment
-> word timestamps
-> whisperx.assign_word_speakers()
-> aggregate words to speaker turns
```

新增 `_aggregate_words_to_turns()`，按 speaker、gap、最大时长和最大字数聚合：

- 同 speaker 且 gap 小于阈值时合并
- speaker 变化、gap 过大、段长超过限制时切分
- 段长控制在约 15 秒、80 字以内

### 额外保护

WhisperX alignment 在中文长句上偶尔会失败，把几十个字压到极短时间。新增 `_turns_need_segment_fallback()` 检测物理不合理结果：

- 长文本字速过高
- 单字拖太久
- 空段或倒置时间戳

检测失败时自动回退到 ASR segment timestamp，避免暴露错误 word timestamp。

## 3. forced alignment 前预切分导致错位

### 现象

曾经为控制段长，在 forced alignment 前按字符比例预切分：

```text
原 ASR: [0-60s] 很长文本
预切分: [0-15s], [15-30s], ...
```

这会把文本强行塞进伪造时间窗，导致 alignment 只能在错误范围内找匹配。

### 修复

forced alignment 前不再预切分。现在流程是：

```text
ASR 原始文本和粗时间戳
-> WhisperX forced alignment
-> word-level timestamp
-> turn aggregation
```

`_presplit_segments()` 仅用于 fallback 或实时快速路径，不再用于 forced alignment 主路径。

## 4. 采样率不一致导致时间漂移

### 现象

`.wav`、`.flac` 文件采样率不一致时，ASR、diarization 和 alignment 的时间轴可能漂移。

### 根因

`TransformersWhisperBackend.transcribe()` 曾对 numpy 输入硬编码：

```python
{"raw": audio, "sampling_rate": 16000}
```

但 pipeline 读取文件后没有统一重采样。

### 修复

在 `src/pipeline.py` 新增 `_resample_to_16k()`，文件读取后统一转为 mono float32 16k：

```python
waveform, sr = _resample_to_16k(waveform, sr)
```

同时 `TransformersWhisperBackend.transcribe()` 增加 `sample_rate` 参数，不再硬编码采样率。

## 5. 短尾词缺失

### 现象

无 LLM 情况下出现短尾词被截断：

```text
客户群这        # 应为：客户群这一块
一个看          # 应为：一个看法
再谈一          # 应为：再谈一下
```

### 根因

VAD/diarization 边界过紧时，ASR 输入片段尾部被裁掉。Whisper 没听到尾音，LLM 后处理也不应凭空补。

### 修复

ASR 输入增加前后 padding，并且 energy VAD 只作为 gate，不再用紧边界裁剪真实语音：

```text
energy VAD 判断是否有真实语音
通过 gate 后保留完整 diarization span
ASR 输入额外带 padding
输出时间戳 clamp 回有效语音区间
```

短尾词优先在模型识别层修复，而不是依赖 LLM。

## 6. 静音或无人说话区域出现文本

### 现象

原音频中没人说话，但输出了文本：

```text
[00:29.8 - 00:32.3] Speaker B: 基
[00:35.2 - 00:38.3] Speaker B: 本上就是...
```

或开头噪声被识别为：

```text
[00:00.0 - 00:00.5] SPEAKER_00: 完
```

### 根因

Pyannote 可能把低能量噪声误检为 speech span。若这些短噪声片段被送入 Whisper，Whisper 容易产生常见幻觉。

### 修复

- `_filter_asr_artifacts()` 清理空文本、括号 artifact、纯标点、物理不合理短段
- `clean_hallucination()` 清理常见 Whisper 片尾幻觉
- openai-whisper 离线路径启用 `hallucination_silence_threshold=2.0`
- transformers 实时路径启用通用解码防重复参数，如 `condition_on_prev_tokens=False`

注意：不再针对某个具体错误词做样例补丁。修复原则是过滤通用幻觉模式和不可信声学区域。

## 7. `.flac` 数据集效果明显变差

### 现象

切换到 AISHELL-4 `.flac` 后，出现：

```text
00:16.0 - 00:16.5: 比喻嗎
```

实际应为会议编号 `018`。同时正文出现大段错位和重复。

### 根因

诊断对比发现：

- 直接 long-form Whisper 能识别开头编号串
- 旧的 diarization-gated 短片段 ASR 会把缺少上下文的短编号误听成中文词
- WhisperX forced alignment 有时会把长中文段压到错误时间点

### 修复

根本修复不是针对 `018` 或 `比喻嗎` 写规则，而是切换离线 ASR 主时钟：

```text
离线文件: openai-whisper sequential long-form ASR
实时路径: transformers fast path
WhisperX alignment: 成功则用 word-level；失败则 fallback 到 ASR segment timestamp
```

这避免了短片段缺上下文和 HF chunk timestamp 不稳定的问题。

## 8. 播放前导静音过长

### 现象

文本正确，但点击播放时从段落前很久的静音开始：

```text
[00:00.0 - 00:13.0] SPEAKER_02:
今天咱们公司新出了一款手机产品...
```

实际说话从 7 秒左右开始。

### 根因

openai-whisper long-form segment 起点有时贴到窗口起点或前导静音。ASR 文本是正确的，但点击播放使用了过粗 start timestamp。

### 修复

新增 `_trim_asr_segments_to_speech()`，在 ASR 后、speaker alignment 前修剪播放边界：

- 优先使用 diarization speech activity 收紧边界
- 没有 diarization 重叠时回退 energy VAD
- 不让 diarization 决定是否转写，只用于修正播放 start/end
- 边界保守留白，避免截断语音

当前参数：

```text
start 前留 0.35s
end 后留 0.45s
只有偏移超过 0.5s 才修剪
修剪后过短则保留 ASR 原始边界
```

验证结果：

```text
修复前: 00:00.000-00:13.000
修复后: 00:06.566-...
```

这样既避免从 0 秒播放长静音，也降低截断首字和尾字的风险。

## 9. 说话人标签不稳定和人工修正需求

### 现象

同一个人可能被分成多个 speaker，例如 `SPEAKER_02` 和 `SPEAKER_03`。另外，部分文本识别错但语义看起来合理，LLM 也不会自动改正。

### 修复

GUI 增加人工修正能力：

- 转写结果下方提供可编辑文本框
- 支持逐段修改转写文本
- 支持直接修改每段 speaker
- 支持批量把某个 speaker 改为另一个 speaker 或自定义姓名
- 说话人重命名面板支持折叠/展开
- 手工修改后保存到 `manual_corrected_latest.json`

speaker 显示也从 A/B/C 改回 `SPEAKER_00`、`SPEAKER_01`，减少和原始 diarization 结果之间的映射混乱。

## 10. 多文件转写时播放器切换

### 现象

一次转写多个文件时，下方播放器只能绑定默认音频，无法手动切换当前播放文件。

### 修复

GUI 增加“播放器音频”下拉框：

- 处理多个文件时复制每个音频到输出目录
- 下拉框列出所有已处理音频
- 切换后播放器绑定对应文件
- 点击段落仍可跳转播放对应隐藏音频或当前播放器音频

## 11. 检索过滤不准确

### 现象

检索时按 speaker、time、intent 等条件过滤不稳定，可能返回不符合过滤条件的段。

### 修复

`src/retrieval/retriever.py` 修复硬过滤逻辑：

- speaker 查询支持 `Speaker B`、`B`、数字编号到 `SPEAKER_XX`
- intent 查询增加 `_intent_matches()`
- speaker/time/intent 作为 hard filter 时，不符合条件的结果不再进入最终排序

## 12. LLM 后处理的定位

### 原则

LLM 可以做：

- 标点恢复
- 繁体转简体
- 明显同音字/错词纠正
- 意图标注
- 会议摘要

LLM 不应该做：

- 修复时间戳
- 补齐 ASR 没听到的短尾词
- 决定 speaker turn 边界
- 代替 VAD/ASR/alignment 处理声学问题

### 当前实现

`src/llm/corrector.py`：

- 扩充繁体转简体常用字映射
- prompt 强调标点恢复
- 清理常见 Whisper 幻觉短语

默认 LLM 模型为 `deepseek-v4-flash`，GUI 中可切换。

## 13. 当前管线概要

离线文件路径：

```text
读音频 -> mono/16k
-> diarization
-> openai-whisper sequential long-form ASR
-> ASR 段边界保守修剪
-> ASR artifact 清理
-> WhisperX forced alignment
-> assign_word_speakers
-> word turn 聚合
-> alignment 质量检测
-> 失败则 fallback 到 ASR segment timestamp
-> LLM 后处理（可选）
-> 检索索引（可选）
```

实时路径：

```text
麦克风 chunk
-> diarization
-> transformers Whisper fast ASR
-> segment-level speaker assignment
-> 基础清理和聚合
-> LLM 上下文纠错（可选）
```

## 14. 参考资料

- [WhisperX: Time-Accurate Speech Transcription of Long-Form Audio](https://arxiv.org/abs/2303.00747): 使用 VAD、batched Whisper inference、forced phoneme alignment 和 speaker diarization 获取更准确的长音频词级时间戳。
- [WhisperX paper page on Hugging Face](https://huggingface.co/papers/2303.00747): WhisperX 方法和论文入口。
- [OpenAI Whisper `transcribe.py`](https://github.com/openai/whisper/blob/main/whisper/transcribe.py): `condition_on_previous_text`、`hallucination_silence_threshold`、`word_timestamps` 等长音频转写参数的实现来源。
- [NVIDIA NeMo Speaker Diarization intro](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/asr/speaker_diarization/intro.html): VAD、speaker embedding、clustering 等 diarization 基本组成。
- [NVIDIA NeMo diarization configs](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/configs.html): ASR-based VAD、word timestamp 与 VAD 结合等配置思路。

## 15. 后续改进方向

- 将 faster-whisper 模型转换为 CTranslate2 格式后，评估其 word timestamp 和速度表现。
- 对不同数据集分别记录 WER、DER、点击播放边界误差。
- 增加自动回归样例，覆盖 `.wav` 近讲麦、`.flac` 远场会议、静音开头、多人重叠说话。
- 将人工修正结果回写为可再次检索和导出的正式 transcript。
