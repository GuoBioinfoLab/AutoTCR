# AutoTCR model card

## Released model

- Weight repository: https://huggingface.co/loveCloud/AutoTCR
- Filename: AutoTCR.pth
- Pinned revision: 909bcd38137e73b8a94e709f3758b5d1a8180ebd
- SHA-256: 8949531ceee16f9919b90165d388460c3700d6971958866894ab893076bd58c3
- Checkpoint size: 230268116 bytes

The model contains a 12-layer BERT encoder (hidden size 768, 12 heads, intermediate dimension 1536) and a 768→128→32→2 MLP head. Parameter count is 57,549,026. Dropout between classifier layers is 0.3 and is disabled during evaluation.

## Representation and prediction

Inputs concatenate TRBV, CDR3β and TRBJ strings. The author-supplied vocabulary has 103 tokens: PAD=$/0, MASK=./1, UNK=?/2, SEP=|/3 and CLS=*/4. The custom tokenizer applies longest-match segmentation without ## prefixes. Inference pads to 32 tokens and uses first-last averaging over the full length.

Class 1 denotes autoimmune-associated sequence patterns. ARS is the observed-abundance-weighted mean of class-1 probabilities. This model does not identify a specific autoimmune disease.

## Input processing

The software operates on already processed structured sequences and repertoires. Raw-read processing, V/J assignment and GLIPH2 clustering are outside the package. Comparisons with the manuscript require matching upstream processing and candidate-clonotype selection.

## Software validation

The real published checkpoint was obtained through the Hub and loaded strictly on CPU. See VALIDATION.md for tokenizer, formula-regression, unit-test and installed-package checks. Regression against original operations is a software check, not a benchmark of biological performance.

CUDA execution is supported but was not tested in the available environment. Real-checkpoint verification defaults to the pinned revision; a future weight update should be distributed with matched configurations and a new software version.

## Interpretation and publication metadata

Sequence scores capture learned associations, and repertoire scores summarize the supplied processed repertoire. Neither output alone verifies antigen specificity or establishes clinical diagnosis.

Training sources, split design and disease-specific performance should be taken from the final article and supplementary materials. Confirm the final paper DOI, author list and model usage license before public release. Existing third-party tokenizer attribution is retained.

