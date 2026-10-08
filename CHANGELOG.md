# Changelog

## 1.1.0 — 2026-10-08

- Load the published loveCloud/AutoTCR checkpoint, pinned by Hub revision and SHA-256.
- Ship the matched 103-token vocabulary, BERT configuration and original inference parameters.
- Add from_pretrained(), download_model() and the download-model CLI command.
- Support complete local offline bundles and cached Hub inference.
- Set the released default batch size to64; retain fixed inference length32.
- Add real-checkpoint tokenizer/formula regression tooling and expand the README.
- Keep large weight files out of the GitHub repository.

## 1.0.0

Initial packaged local inference interfaces, original tokenizer/model compatibility, cohort routing, API documentation and synthetic CPU tests.
