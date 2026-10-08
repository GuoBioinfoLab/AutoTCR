# AutoTCR

**对 TRBV–CDR3β–TRBJ 序列进行预测，并计算免疫组库的自身免疫相关评分。**

模型权重位于 [Hugging Face：loveCloud/AutoTCR](https://huggingface.co/loveCloud/AutoTCR)。本仓库已经包含对应的真实结构配置、103项词表、特殊 token 设置与推理参数，用户无需手动填写配置。

[英文说明](README.md) · [接口文档](docs/API.md) · [数据格式](docs/DATA_FORMATS.md) · [测试记录](VALIDATION.md)

## 快速使用

在仓库根目录执行：

```bash
python -m pip install -e ".[runtime]"
autotcr download-model --output-dir models/autotcr

autotcr repertoire \
  --model-dir models/autotcr \
  --input examples/repertoire.csv \
  --summary-output outputs/summary.csv \
  --predictions-output outputs/predictions.csv.gz \
  --device cpu
```

已有兼容 PyTorch 环境可直接安装本包。新建 CPU 环境时，可以先运行：

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

GPU 用户先按照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装适合驱动的版本，再用 `--device cuda` 或 `--device cuda:0`。`auto` 自动选择可用设备。默认 batch size 为64，可通过 `--batch-size` 修改。

## 模型下载与离线运行

`download-model` 将 Hugging Face 的 AutoTCR.pth 与包内配置组合为完整本地目录，校验权重 SHA-256，并记录来源。固定发布 commit 为：

`909bcd38137e73b8a94e709f3758b5d1a8180ebd`

| 方式 | 用法 |
|---|---|
| 使用本地完整模型 | 加 `--model-dir models/autotcr`，推理不联网 |
| 自动使用 Hugging Face 缓存 | 不指定模型来源，首次自动下载 |
| 已有缓存但无网络 | 加 `--local-files-only` |
| 自定义本地路径 | 加 `--settings configs/inference.yaml` |

CPU/GPU 计算节点无网络时，在联网节点完成下载，再把整个模型文件夹复制过去。无需原服务器的 pretrained_path，也不需要额外的预训练权重文件。

## 序列级预测

输入包含 `sequence` 或 `TcRb_vj` 列；标签可有可无，不进入模型。

```bash
autotcr sequences \
  --model-dir models/autotcr \
  --input examples/sequences.csv \
  --output outputs/sequences.csv \
  --device cpu
```

输出保留原行顺序与元数据，增加两类概率、预测标签及 `autoimmune_probability`。预测标签为概率更高的类别，不是临床诊断阈值。

## 免疫组库评分

一个文件放一个处理后的组库，至少包含 `TcRb_vj` 和 `Freq`。Freq 可以是计数或频率；程序按照总和归一化。

ARS = Σ(丰度 × class-1 概率) / Σ丰度。

`freq_prob` 是 ARS 的同值标量列。输出还包括 top100_mean、top_diff_mean 和其他兼容评分。所有评分只用一次模型推理计算；重复记录保留。逐条结果中还包含 normalized_abundance 和 ars_contribution。

## 批量队列

新样本没有标签时，提供 `sample,file_path` 清单。相对路径从清单所在目录解析：

```bash
autotcr cohort \
  --model-dir models/autotcr \
  --manifest examples/cohort_manifest.csv \
  --output outputs/cohort.csv \
  --device cpu
```

原来的 internal/external 数据文件夹可以直接使用：

```bash
autotcr cohort \
  --model-dir models/autotcr \
  --metadata-dir /path/to/dataset_pv2 \
  --evaluation-set all \
  --disease-dir /path/to/clustered/disease \
  --healthy-dir /path/to/hc_clustered_freq \
  --output outputs/internal_external_scores.csv \
  --predictions-dir outputs/per_sample_predictions \
  --device auto
```

包括带 _add 后缀的 internal/external 文件，排除 clinical。true_label 为1时从病例目录找文件，为0时从健康目录找文件，文件名为 `{sample}_input_seq_vj.csv`。标签只决定文件路径，不进入模型。已有评分会重新计算。找不到文件时默认报错；可用 `--skip-missing` 保留缺失状态行。

## Python 接口

```python
from autotcr import AutoTCRPredictor

predictor = AutoTCRPredictor.from_pretrained(
    "loveCloud/AutoTCR", device="cpu", batch_size=64
)
result = predictor.predict_repertoire("examples/repertoire.csv")
print(result.summary["ars"])

# 已下载的离线模型：
predictor = AutoTCRPredictor.from_model_dir("models/autotcr", device="cpu")
```

## 保留的原始模型逻辑

- 特殊 token 与 ID：PAD=$/0，MASK=./1，UNK=?/2，SEP=|/3，CLS=*/4。
- 103项词表，保持原行顺序；不添加 ## 前缀，不转换大小写。
- 固定输入长度32，包含特殊 token。结构配置允许64个位置，不代表原推理使用64。
- first-last-avg 使用第一个编码层与最后一个编码层，保留包含补齐位置的平均池化。
- 分类头为768→128→32→2，dropout=0.3，评估时关闭。
- 完整权重严格加载，支持原 module. 前缀；默认不截断超长输入。

包不执行原始测序处理、V/J 注释或 GLIPH2 聚类。复现文章结果需要相同的上游处理与序列选择。

## 验证与上传

测试记录见 [VALIDATION.md](VALIDATION.md)。真实模型比对工具为 `scripts/validate_released_model.py`；单元测试无需下载大模型。

GitHub 只需上传本包的代码、配置和文档；大权重与缓存已被忽略，无需 Git LFS。详见 [作者上传指南](docs/UPLOAD_GUIDE.zh-CN.md)。源码与权重的使用许可证请作者明确；分词器原有 Apache-2.0 声明已保留。
