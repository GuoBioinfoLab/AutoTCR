# AutoTCR model files

Weights are published at [loveCloud/AutoTCR](https://huggingface.co/loveCloud/AutoTCR) under the filename `AutoTCR.pth`. This GitHub repository ships matched small assets in `src/autotcr/assets/` and does not include the large checkpoint.

Download a complete offline bundle:

```bash
autotcr download-model --output-dir models/autotcr
autotcr validate-config --model-dir models/autotcr --device cpu
```

The directory will contain AutoTCR.pth, config.json, vocab.txt, tokenizer_config.json, inference_config.yaml, model_source.json and checksums.json. Use the whole directory for offline deployment.

The official release is pinned to commit `909bcd38137e73b8a94e709f3758b5d1a8180ebd`. Checkpoint size is 230268116 bytes and SHA-256 is:

`8949531ceee16f9919b90165d388460c3700d6971958866894ab893076bd58c3`

The vocabulary contains 103 entries, preserving author-supplied order. Special IDs: PAD $=0, MASK .=1, UNK ?=2, SEP |=3, CLS *=4. Fixed inference length is32 and pooling is first-last-avg.

The full checkpoint includes BERT and the classifier, so a second pretrained weight file is not needed. Historical bundles containing 10k.pth are accepted by from_model_dir. Existing different model assets are not overwritten by the official download command.

Large weights and caches are ignored in this code repository. The author should separately state model-weight usage terms on the Hugging Face model card.

