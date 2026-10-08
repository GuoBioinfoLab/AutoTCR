# Validation of AutoTCR v1.1.0

Validation date: 2026-10-08.

## Environment

Python 3.12; PyTorch 2.5.1+cpu; Transformers 4.45.2; huggingface_hub 0.36.2; NumPy 2.3.5; pandas 2.2.3; PyYAML 6.0.3.

## Published checkpoint

The public loveCloud/AutoTCR model was retrieved at commit 909bcd38137e73b8a94e709f3758b5d1a8180ebd.

| Check | Observed result |
|---|---|
| Filename | AutoTCR.pth |
| Size | 230268116 bytes |
| SHA-256 | 8949531ceee16f9919b90165d388460c3700d6971958866894ab893076bd58c3 |
| Complete state-dictionary loading | Passed with strict=True |
| Model parameters | 57,549,026 |
| Vocabulary | 103 tokens |
| Fixed inference length | 32 |
| Pooling | first-last-avg |
| PAD/MASK/UNK/SEP/CLS IDs | 0/1/2/3/4 |

## Real-checkpoint regression

The original supplied tokenizer was loaded with the author-supplied tokenizer configuration. Its token IDs were compared with packaged inference for four structured receptor sequences. The supplied fixed-length padding, first-last pooling and classifier operations were independently transcribed and evaluated using the same loaded encoder/head parameters.

- Token IDs matched.
- Maximum absolute difference in class probabilities: 0.
- Absolute ARS difference: 0.

The machine-readable report is in docs/released_model_validation.json. The reusable command is scripts/validate_released_model.py.

These comparisons establish agreement with original tokenizer/pooling operations on the tested inputs. They share the loaded model parameters and do not replace an independent full-dataset comparison with outputs exported from the original training environment.

## Automated software tests

All **39 unique tests** passed. They cover data parsing, order/duplicate preservation, abundance aggregation, internal/external metadata discovery, case/control routing, unlabelled manifests, shipped release assets, download plumbing, fixed padding, the three supported pooling modes, batch-size consistency and strict checkpoint loading.

Synthetic integration tests exercise validate-config, sequences, repertoire and cohort on CPU. They create random tiny checkpoints in temporary directories and require no real-checkpoint download. Ruff and Python compilation checks passed.

## Offline and distribution checks

The download-model CLI created a complete real offline bundle from the verified Hub cache. The repertoire CLI then scored the example using only local assets. Wheel and source distributions built successfully. The wheel includes all four bundled asset files and the tokenizer license. After installing the wheel and running outside the source directory, the CLI reported version 1.1.0, validate-config strictly loaded the real checkpoint, and the original-tokenizer/pooling regression passed with zero differences.

## Scope

CPU execution was tested. CUDA selection is implemented, but no GPU execution was available here. Remote GitHub CI was configured but not run during this task.

The examples and regression sequences are software checks, not biological-performance benchmarks. Upstream repertoire processing and candidate selection must match the intended evaluation workflow.
