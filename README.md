# 长音频转写对齐与智能检索系统

本项目是一个面向长音频资料的本地语音转写系统，支持文件转写、说话人识别、词级时间轴对齐、人工修正、历史结果管理、实时转写预览和智能检索。系统输出结构化 JSON，包含时间戳、说话人、转写文本、LLM 后处理结果、低置信度标记和检索所需字段。

项目托管地址：<https://github.com/uMemory/ASR.git>

## 功能概览

- 文件转写：手动添加一个或多个音频文件，生成带时间戳和说话人标签的转写结果。
- 点击回放：点击文本片段播放对应音频区间，用于核对音频与文本。
- 说话人管理：支持说话人重命名、批量替换和手动合并。
- 文本修正：支持人工修改转写文本，并保存修正后的 JSON。
- 历史结果：保存转写结果，刷新页面后可重新加载；同名文件重新转写时覆盖旧结果。
- LLM 后处理：可选启用文本纠错、标点恢复、意图标注和摘要生成。
- 实时转写：支持麦克风录音和测试音频的实时转写预览，保存后可继续离线精修。
- 智能检索：支持语义、关键词、说话人、时间、意图等维度的复合检索。
- 跨语言查询：检索前可使用 LLM 将查询改写为中英语义和关键词变体。
- 自动评估：支持对已保存 JSON 计算 CER/WER、LLM 前后对比、低置信度率和 fallback 率。

## 处理流程

```text
音频输入
  -> ASR 转写
  -> 说话人识别
  -> word-level 时间戳对齐
  -> 低置信度检测
  -> speaker turns 聚合
  -> 可选 LLM 后处理
  -> 保存 JSON / 历史结果
  -> 可选构建检索索引
```

文件转写以准确性和可回放校验为主；实时转写以低延迟预览为主。实时录音保存后，可以再执行离线精修流程。

## 环境要求

- Windows 11
- Python 3.10
- Conda 环境名
- NVIDIA GPU，推荐 8GB 显存以上
- PyTorch 2.5.1 + CUDA 12.4
- Gradio 5.x

安装依赖：

```powershell
conda create -n TTS python=3.10 -y
conda activate TTS

pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
pip install whisperx==3.4.2 --no-deps
```

依赖版本说明见 [requirements.txt](requirements.txt)。

## 模型下载

模型默认放在 `models/` 目录下，路径由 [configs/models.yaml](configs/models.yaml) 配置：

- ASR：`models/whisper-medium`
- 说话人识别：`models/speaker-diarization-3.1` 与 `models/segmentation-3.0`
- 中文强制对齐：`models/wav2vec2-large-xlsr-53-chinese-zh-cn`
- 英文强制对齐：`WAV2VEC2_ASR_BASE_960H`
- 检索编码：`models/bge-m3`

下载默认模型：

```powershell
python download.py
```

只下载部分模型：

```powershell
python download.py --only whisper-medium bge-m3
```

Pyannote 相关模型需要 Hugging Face token，并需要先在模型页面接受使用协议：

```powershell
$env:HF_TOKEN="your_hf_token"
python download.py --only pyannote-diarization pyannote-segmentation
```

## 配置

主要配置文件：

- [configs/models.yaml](configs/models.yaml)：模型路径、设备、LLM、检索开关。
- [configs/languages.yaml](configs/languages.yaml)：语言和意图标签。
- [configs/retrieval.yaml](configs/retrieval.yaml)：检索权重和 query rewrite 配置。

项目根目录创建 `.env`，可参考 `.env.example`：

```env
ACTIVE_LLM=deepseek
DEEPSEEK_API_KEY=your_key_here
CLAUDE_API_KEY=your_key_here
CLAUDE_BASE_URL=https://api.anthropic.com
HF_TOKEN=your_hf_token
```

GUI 中可以选择是否启用 LLM。关闭 LLM 时执行 ASR、说话人识别、对齐和基础过滤；开启 LLM 后增加文本纠错、意图标注和摘要生成。

## 快速开始

检查环境：

```powershell
conda activate TTS
cd E:\ASR
python scripts/check_env.py
python scripts/import_test.py
```

启动图形界面：

```powershell
python gui/app.py
```

默认访问地址：

```text
http://127.0.0.1:7860
```

不使用实时麦克风功能时：

```powershell
python gui/app.py --no-ws
```

## 文件转写

推荐使用 GUI 的“文件转写”模式：

1. 添加音频文件。
2. 选择语言：`zh` 或 `en`。
3. 选择是否启用 LLM 后处理。
4. 开始转写。
5. 点击文本段落核对对应音频。
6. 修改说话人或文本。
7. 保存 JSON。

命令行处理：

```powershell
python scripts/run_test.py --path .\audio.wav --language zh --seconds 240 --no-llm
python scripts/run_test.py --path .\audio_dir --language en --seconds 180
python scripts/run_test.py --path .\audio_dir --language zh --retrieval
```

`--seconds 0` 表示处理完整文件。

## 实时转写

GUI 的“实时麦克风”用于实时预览。实时结果会显示 ASR 直接结果和临时稿，停止并保存后可以继续执行离线精修。

命令行实时录音：

```powershell
python scripts/realtime_mic.py --language zh --chunk 10
python scripts/realtime_mic.py --language en --chunk 15 --no-llm
python scripts/realtime_mic.py --list-devices
python scripts/realtime_mic.py --device 1 --save-audio outputs/realtime.wav
```

块时长建议：

- `8-10s`：延迟较低，适合快速预览。
- `12-15s`：上下文更充分，适合较长句子。
- `20s+`：延迟较高，不建议作为默认值。

## 智能检索

检索模块读取转写段落并构建索引，支持：

- 查询改写：LLM 生成中英语义查询和关键词扩展，失败时使用规则回退。
- 语义检索：BGE-M3 dense embedding。
- 关键词检索：BGE-M3 sparse token weights 或 BM25 回退。
- 结构过滤：说话人、时间范围、意图标签。
- 排序融合：RRF 融合多路结果。

命令行检索示例：

```powershell
python scripts/demo_retrieval.py --audio .\audio.wav --seconds 120 --language zh
```

检索索引默认保存在 `outputs/index`。

查询示例：

```text
价格太贵的讨论
discussion about expensive rent
SPEAKER_00 对交通的建议
主播带货对购物体验的影响
```

## 评估

评估脚本读取已保存的转写 JSON，不重新运行模型。当前主要指标：

- 中文文本识别：`CER`
- 英文文本识别：`WER`
- LLM 前后对比：比较 `text_before_llm` 与最终文本
- 错误类型：插入、删除、替换
- 可靠性：低置信度率、word-level fallback 率

评估说明见 [experiments/README.md](experiments/README.md)。

评估已有 JSON：

```powershell
python experiments/evaluate_system.py `
  --dataset aishell1 `
  --pred-dir outputs/batch_eval_llm/results/aishell1 `
  --compare-llm `
  --output outputs/eval_aishell1.json
```

批量处理数据集并评估：

```powershell
python scripts/batch_dataset_eval.py `
  --dataset aishell1 `
  --batch-size 50 `
  --max-files 50 `
  --seconds 0 `
  --llm `
  --output-root outputs/batch_eval_llm
```

## AMI Parquet 音频导出

AMI 的 Parquet 文件需要先导出为 WAV：

```powershell
python scripts/export_ami_parquet_audio.py `
  --input dataset/AMI/SDM `
  --output dataset/AMI/exported_audio/SDM `
  --max-meetings 2 `
  --duration 180

python scripts/run_test.py `
  --path dataset/AMI/exported_audio/SDM `
  --language en `
  --seconds 180 `
  --no-llm
```

## 输出格式

管线输出为 JSON，核心字段示例：

```json
{
  "audio_path": "audio.wav",
  "language": "zh",
  "segments": [
    {
      "start": 6.9,
      "end": 19.9,
      "speaker": "SPEAKER_00",
      "text": "转写文本",
      "text_before_llm": "LLM 修正前文本",
      "intent": ["陈述"],
      "low_confidence": false
    }
  ],
  "timing": {
    "asr": 12.3,
    "diarization": 5.4,
    "alignment": 2.1
  },
  "metadata": {}
}
```

不同入口保存的 JSON 可能包含额外字段，例如历史转写、音频文件名、人工修正状态、低置信度原因或检索信息。

## 目录结构

```text
E:/ASR/
├── README.md
├── requirements.txt
├── download.py
├── configs/
│   ├── models.yaml
│   ├── languages.yaml
│   └── retrieval.yaml
├── src/
│   ├── pipeline.py
│   ├── asr/
│   ├── diarization/
│   ├── alignment/
│   ├── llm/
│   ├── retrieval/
│   ├── streaming/
│   └── utils/
├── gui/
│   ├── app.py
│   ├── realtime_backend.py
│   ├── realtime_controller.py
│   └── static_js.py
├── scripts/
│   ├── check_env.py
│   ├── import_test.py
│   ├── run_test.py
│   ├── realtime_mic.py
│   ├── batch_dataset_eval.py
│   ├── export_ami_parquet_audio.py
│   ├── demo_asr.py
│   ├── demo_pipeline.py
│   ├── demo_llm.py
│   ├── demo_retrieval.py
│   └── demo_streaming.py
├── experiments/
│   ├── README.md
│   ├── evaluate_system.py
│   └── comprehensive_eval.py
├── models/
├── dataset/
├── outputs/
└── tests/
```
- 原始音频、视频和运行日志
- 模型权重和数据集文件

`models/` 与 `dataset/` 会保留目录骨架，但目录内的实际大文件由 `.gitignore` 忽略。模型可通过 [download.py](download.py) 下载，数据集需按需要放入对应目录。
