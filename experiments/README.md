# 评估说明

本目录存放项目评估脚本，用于读取已经保存的转写 JSON 结果并计算指标。默认情况下，评估脚本不会重新运行 ASR，而是直接读取已有输出。

## 当前评估定位

本系统输出的是面向用户使用的 speaker turn：每个片段都应当可阅读、可点击播放、可人工修改、可保存和可检索。AISHELL-4、AliMeeting 等公开数据集常常提供更细粒度的 TextGrid 或说话人标注，而系统输出粒度更接近可阅读片段。因此，当参考标注粒度与系统输出粒度不一致时，直接计算时间边界误差、DER 或 JER 可能会低估系统的实际可用性。

当前阶段建议把以下指标作为主要结论：

- 中文文本识别：`CER`、插入、删除、替换错误。
- 英文文本识别：`WER`、插入、删除、替换错误。
- LLM 影响：比较 `text_before_llm` 与最终文本。
- 可靠性：低置信度片段比例、word-level fallback 率。
- 检索可用性：基于已保存转写文本的 Top-K 检索指标。
- 无标准参考的英文或多说话人样本：用于人工核验转写、说话人切换、点击播放和跨语言检索效果。

以下指标可作为辅助观察：

- `Start MAE`、`End MAE`、`Hit@0.5s`、`Hit@1.0s`。
- `DER`、`JER`、speaker turn accuracy。

这些辅助指标仍可由脚本计算，但当参考与预测的切分粒度不一致时，不应作为报告的核心结论。

## 主评估脚本

```powershell
conda run -n TTS python -B experiments/evaluate_system.py
```

## 本项目使用的保存结果

当前评估主要基于已经处理完成的结果目录：

```text
outputs/batch_eval_llm/results
```

早期人工测试结果可能保存在：

```text
tests/test_results
```

只有具备本地参考文本或参考标注的结果才能自动计算文本指标。没有标准参考的英文样本和多说话人样本主要用于定性展示与人工核验。

## 文本准确率评估

AISHELL-1 中文样本：

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset aishell1 `
  --pred-dir tests/test_results `
  --language zh `
  --compare-llm `
  --output outputs/eval_aishell1.json
```

LibriSpeech 英文样本：

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset librispeech `
  --pred-dir tests/test_results `
  --language en `
  --compare-llm `
  --output outputs/eval_librispeech.json
```

旧版 TextGrid 数据集示例：

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset alimeeting-near `
  --pred-dir tests/test_results `
  --compare-llm `
  --output outputs/eval_alimeeting_near.json
```

AISHELL-4 样本示例：

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset aishell4 `
  --pred tests/test_results/L_R004S01C01.flac.json `
         tests/test_results/M_R003S01C01.flac.json `
         tests/test_results/S_R003S01C01.flac.json `
  --duration 240 `
  --compare-llm `
  --output outputs/eval_aishell4_240.json
```

远场样本计算说话人指标可能较慢，建议先指定固定时长：

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset alimeeting-far `
  --pred tests/test_results/R8007_M8010_MS803.wav.json `
  --duration 240 `
  --compare-llm `
  --output outputs/eval_alimeeting_far_240.json
```

## LLM 前后对比

如果保存的 JSON 中包含 `text_before_llm`，使用 `--compare-llm` 后会同时报告：

- LLM 后 `CER/WER`。
- LLM 前 `CER/WER`。
- 错误率变化。
- 插入、删除、替换错误变化。

错误率下降说明 LLM 降低了严格文本错误率；错误率上升并不一定代表文本不可读，因为 LLM 可能进行了标点恢复、繁简转换、大小写规范或轻微改写，而这些变化可能与参考文本格式不完全一致。

## 检索评估

检索评估读取已保存的转写 JSON，并使用 Web UI 相同的 BGE-M3 检索路径构建索引：

```powershell
conda run -n TTS python -B experiments/evaluate_retrieval_saved.py `
  --encoder bge `
  --result-root outputs/batch_eval_llm/results `
  --extra-result-dirs tests/test_results `
  --output-dir outputs/retrieval_eval_bge_multidim_with_test
```

评估分为两组：

- 链路自检查询：使用完整原文查询和片段原文查询，目标片段已知，用于验证 BGE-M3 编码、FAISS 索引、稀疏关键词权重、RRF 融合和 Top-K 返回流程是否正常。
- 模拟用户查询：根据转写内容设计自然语言问题，覆盖语义、关键词、跨语言、说话人约束、时间范围约束、意图约束和混合约束。

其中 `outputs/batch_eval_llm/results` 主要提供带参考文本的数据集转写结果，`tests/test_results` 提供 Web UI 历史转写结果。多维混合检索指标会额外使用 `tests/test_results` 中的多说话人样本，例如 `ahnss.wav.json`、`cjfer.wav.json`、`R8007_M8010_N_SPK8050.wav.json` 等。

两组评估都报告 `Hit@1`、`Hit@5`、`MRR@5` 和平均查询延迟。模拟用户查询比完整原文查询/片段原文查询更接近真实使用方式，但它仍是基于当前保存结果设计的小规模评估集，不等同于大规模人工相关性基准。

当前检索器实际支持的结构化约束包括 speaker、time range 和 intent。数据集名称只在评估脚本中用于判断相关性，不是 Web UI 当前的过滤维度。

如果只需要调试检索流程，也可以使用轻量哈希编码器，不加载 BGE-M3：

```powershell
conda run -n TTS python -B experiments/evaluate_retrieval_saved.py `
  --encoder simple `
  --result-root outputs/batch_eval_llm/results `
  --output-dir outputs/retrieval_eval
```

当前主要输出文件：

```text
outputs/retrieval_eval_bge_multidim/retrieval_eval_summary.md
outputs/retrieval_eval_bge_multidim/retrieval_eval_summary.json
outputs/retrieval_eval_bge_multidim_with_test/retrieval_eval_summary.md
outputs/retrieval_eval_bge_multidim_with_test/retrieval_eval_summary.json
outputs/retrieval_eval_bge/retrieval_eval_summary.md
outputs/retrieval_eval_bge/retrieval_eval_summary.json
```

## 英文数据集说明

当前定量英文 ASR 指标使用 LibriSpeech。LibriSpeech 提供官方参考文本，适合计算英文 WER。

`tests/test_results` 中的部分英文样本没有本地官方参考文本，适合做定性展示：

- 检查多说话人场景下的展示效果。
- 检查连续音频的点击播放对齐。
- 展示跨语言检索和自然语言检索。

如果后续需要更完整的英文定量评估，建议使用：

- LibriSpeech `test-clean` 或 `test-other`：用于英文 WER。
- AMI 原始连续音频和官方标注：用于会议式英文评估。

VoxConverse 风格数据更适合展示说话人日志能力，因为它通常提供 RTTM 说话人活动标注，而不是完整人工转写文本。

## 人工参考模板

对于短英文片段或自定义音频，可以先导出人工标注模板：

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset alimeeting-near `
  --pred tests/test_results/ahnss.wav.json `
  --export-reference-template references/ahnss_reference_template.json
```

然后人工编辑 `segments[start,end,speaker,text]`，再使用参考 JSON 评估：

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset alimeeting-near `
  --pred tests/test_results/ahnss.wav.json `
  --reference-json references/ahnss_reference_template.json `
  --language en
```
