# AutoTCR

**Sequence and immune-repertoire inference for autoimmune-associated TCR patterns.**

AutoTCR scores structured **TRBV–CDR3β–TRBJ** sequences with a pretrained BERT encoder and aggregates their probabilities into an abundance-weighted autoimmune repertoire score (**ARS**).

This repository provides the inference software, matched tokenizer and configuration files. The trained weights are hosted at **[loveCloud/AutoTCR on Hugging Face](https://huggingface.co/loveCloud/AutoTCR)**.

[中文说明](README.zh-CN.md) · [Python API](docs/API.md) · [Input and output formats](docs/DATA_FORMATS.md) · [Validation](VALIDATION.md)

## Quick start

Requires **Python ≥3.10**. Run these commands from the repository root:

```bash
# 1. Install (a compatible PyTorch installation is required).
python -m pip install -e ".[runtime]"

# 2. Download the model once.
autotcr download-model --output-dir models/autotcr

# 3. Score the example repertoire on CPU.
autotcr repertoire \
  --model-dir models/autotcr \
  --input examples/repertoire.csv \
  --summary-output outputs/summary.csv \
  --predictions-output outputs/predictions.csv.gz \
  --device cpu
```

The example contains illustrative records, not a validation cohort. For sequence predictions, use the `sequences` command below.

**No manual model configuration is needed.** The download command combines the verified checkpoint with the configuration and vocabulary included in this package. It creates a complete local bundle, records the source revision and checksums, and leaves existing different assets untouched.

## Installation

### CPU

For a new Linux/macOS environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[runtime]"
```

If PyTorch is already installed, install only this package. On Windows, activate your virtual environment using `.venv\Scripts\activate`, or use an existing conda environment.

### GPU

Install the PyTorch build appropriate for your driver using the [official installation selector](https://pytorch.org/get-started/locally/), then run:

```bash
python -m pip install -e ".[runtime]"
```

Use `--device cuda` or `--device cuda:0`. The default `auto` selects CUDA when available and otherwise uses CPU. Adjust batch size with `--batch-size 64`; reducing it can help with GPU memory limits.

The package uses Transformers **4.45.2**, matching the supplied encoder configuration. It requires no TensorFlow, Scanpy, MiXCR or training utilities.

## Model loading

Choose the workflow appropriate for your environment:

| Workflow | Command options | Network access |
|---|---|---|
| Complete local bundle | `--model-dir models/autotcr` | None during inference |
| Automatic Hub cache | Omit model-source options | Downloads on first use; reuses the cache |
| Existing Hub cache offline | `--local-files-only` | None |
| Custom local settings | `--settings configs/inference.yaml` | Local asset paths |

The release pins Hugging Face commit **`909bcd38137e73b8a94e709f3758b5d1a8180ebd`** and verifies the weight file's SHA-256. An update to the Hub's main branch does not silently change the checkpoint used by this software version.

The local bundle contains `AutoTCR.pth`, `config.json`, `vocab.txt`, `tokenizer_config.json`, `inference_config.yaml`, `model_source.json` and `checksums.json`. Older local bundles named `10k.pth` remain supported.

On a compute node without internet, download the bundle on a connected machine and copy the **entire folder** to the node.

## Sequence prediction

CSV/TSV input needs one sequence column:

```csv
sequence
TRBV18CASSSTSDTDTQYFTRBJ2-3
TRBV7-8CASSSSGTTEAFFTRBJ1-1
```

```bash
autotcr sequences \
  --model-dir models/autotcr \
  --input examples/sequences.csv \
  --output outputs/sequences.csv \
  --device cpu
```

Recognized column names include `sequence` and `TcRb_vj`. Use `--sequence-column` for another name. Labels are optional and never supplied to the network.

Outputs retain input order, duplicate records and metadata:

| Added column | Meaning |
|---|---|
| `input_row` | Zero-based input row |
| `prob_class_0` | Healthy-reference class probability |
| `prob_class_1` | Autoimmune-associated class probability |
| `autoimmune_probability` | Alias for the positive-class probability |
| `predicted_label` | Class with the greater probability; not a clinical operating threshold |

## Repertoire scoring

Each input file represents **one processed repertoire**, with a sequence column and an abundance column. Common names are `TcRb_vj` and `Freq`. Both counts and relative frequencies are accepted; total abundance must be positive.

```bash
autotcr repertoire \
  --model-dir models/autotcr \
  --input examples/repertoire.csv \
  --sample-id example \
  --summary-output outputs/repertoire_summary.csv \
  --predictions-output outputs/repertoire_predictions.csv.gz \
  --device cpu
```

For clonotype i with abundance a_i and class-1 probability p_i:

**ARS = Σ(a_i × p_i) / Σa_i.**

| Summary field | Meaning |
|---|---|
| `ars` | Primary abundance-weighted repertoire score |
| `freq_prob` | Identical scalar compatibility alias for ARS |
| `top100_mean` | Mean of the highest min(100,N) sequence probabilities |
| `top_diff_mean` | Highest floor(0.1N) mean minus overall mean; zero when N<10 |
| `softmax_sharp`, `softmax_freq` | Optional legacy aggregation rules |
| `n_input_records` | Input row count, including duplicates |
| `n_unique_sequences` | Distinct structured sequence count |
| `total_abundance` | Sum of the supplied abundances |

All scores come from one model inference pass. Per-sequence outputs include `normalized_abundance` and `ars_contribution`; contributions sum to ARS. The legacy `n_clonotypes` field is a row-count alias.

ARS is a continuous repertoire score. The software does not impose a disease-specific or clinical decision threshold.

## Batch inference

### New samples without known labels

Create a manifest with `sample,file_path`. Relative file paths resolve against the manifest's directory.

```csv
sample,file_path
example,repertoire.csv
```

```bash
autotcr cohort \
  --model-dir models/autotcr \
  --manifest examples/cohort_manifest.csv \
  --output outputs/cohort.csv \
  --predictions-dir outputs/per_sample_predictions \
  --device cpu
```

### Original internal/external evaluation layout

A metadata directory can contain files such as `AIH_internal.csv`, `AIH_external.csv`, `RA_internal_add.csv` and `T1D_external_add.csv`.

```bash
autotcr cohort \
  --model-dir models/autotcr \
  --metadata-dir /path/to/dataset_pv2 \
  --evaluation-set all \
  --disease-dir /path/to/clustered/disease \
  --healthy-dir /path/to/hc_clustered_freq \
  --output outputs/internal_external_scores.csv \
  --device auto
```

For metadata with `sample,true_label`, label 1 selects the disease directory and label 0 selects the healthy directory; filenames are `{sample}_input_seq_vj.csv`. Labels select the files and are not model inputs. Explicit `file_path` takes precedence.

Discovery loads internal/external files, including `_add` suffixes, and excludes clinical files. Existing scores are recomputed. Repeated paths reuse one inference result while retaining the metadata rows.

Missing files fail by default. `--skip-missing` retains them with `status=missing` and empty scores. Optional per-sequence outputs use path hashes; `predictions_file` maps the summary rows to their gzip files.

## Python API

```python
from autotcr import AutoTCRPredictor

# Download once and reuse the Hugging Face cache.
predictor = AutoTCRPredictor.from_pretrained(
    "loveCloud/AutoTCR",
    device="cpu",
    batch_size=64,
)

sequence_scores = predictor.predict_sequences("examples/sequences.csv")
result = predictor.predict_repertoire("examples/repertoire.csv", sample_id="example")
print(result.summary["ars"])

result.save(
    summary_path="outputs/summary.csv",
    predictions_path="outputs/predictions.csv.gz",
)
```

For an offline local bundle:

```python
predictor = AutoTCRPredictor.from_model_dir("models/autotcr", device="cpu")
```

See [API.md](docs/API.md) for signatures, settings and cohort options. Actual network loading is lazy after assets are resolved.

## Model configuration and compatibility

| Setting | Released value |
|---|---|
| Encoder | 12 BERT layers, hidden size 768, 12 attention heads |
| Intermediate dimension | 1536 |
| Vocabulary size | 103 |
| Fixed inference length | 32 tokens, including special tokens |
| Pooling | `first-last-avg` |
| Classifier head | 768 → 128 → 32 → 2 |
| Classifier dropout | 0.3; disabled during evaluation |

Special tokens are **PAD=`$` (0), MASK=`.` (1), UNK=`?` (2), SEP=`|` (3), CLS=`*` (4)**.

The custom tokenizer preserves longest-match segmentation without adding WordPiece `##` prefixes. Pooling uses encoder layer 1 (`hidden_states[1]`) and the final layer, averaging the full fixed length including padding and special tokens. Parameters load strictly into the original `bert`, `fc1`, `fc2` and `fc3` names.

Input case, duplicate records and abundance values are preserved. Overlength tokenized inputs raise an error; padding length is not increased automatically. Inference uses FP32, and score aggregation uses float64.

The architecture JSON retains its original `BertForMaskedLM` metadata. Loading uses the packaged BERT encoder and custom classifier head to match the complete classification checkpoint.

## Repertoire preparation

The software accepts **already processed** TRBV–CDR3β–TRBJ records. It does not perform raw-read assembly, V/J assignment or GLIPH2 clustering.

For comparisons with the article, use the same upstream processing and candidate-clonotype selection. Scores from arbitrary unprocessed repertoires may have different distributions. Sequence associations alone do not establish antigen specificity or clinical diagnosis.

## Testing and reproducibility

```bash
python -m unittest discover -s tests -v
python scripts/validate_released_model.py --cache-dir /path/to/hf_cache
```

The first command runs unit and synthetic runtime tests without downloading the real checkpoint. The second checks the published model and compares probabilities with the supplied original tokenizer and original pooling formulas.

[VALIDATION.md](VALIDATION.md) records the completed tests. For independent reference outputs exported by your original environment:

```bash
python scripts/compare_reference_predictions.py \
  --settings configs/inference.yaml --reference original_predictions.csv
```

## Repository and release

[Model assets](models/README.md) · [Author upload guide](docs/UPLOAD_GUIDE.zh-CN.md) · [Model card](MODEL_CARD.md)

Large weight files and download caches are ignored by Git; the GitHub repository distributes code and small configuration files. Model weights remain on Hugging Face. Git LFS is not required for this code repository.

`CITATION.cff` contains confirmed software metadata. Add the final article DOI and author list when available. The custom tokenizer retains its Apache-2.0 attribution. The authors should choose an explicit license for their original code and model weights before advertising unrestricted reuse.
