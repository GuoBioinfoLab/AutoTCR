# Python API

## Load the published model

`AutoTCRPredictor.from_pretrained(repo_id="loveCloud/AutoTCR", *, device="auto", batch_size=64, cache_dir=None, local_files_only=False) -> AutoTCRPredictor`

Downloads the pinned weight release through huggingface_hub, verifies size and SHA-256, and combines it with the bundled author-supplied configuration. Subsequent calls reuse the cache. local_files_only=True prohibits network access. The supplied assets match loveCloud/AutoTCR; other models should be loaded with their own matched local bundle.

`download_model(output_dir, *, cache_dir=None, local_files_only=False) -> pathlib.Path`

Creates a complete offline model bundle and source/checksum metadata. Existing different weight or configuration files are not overwritten.

## Load local assets

`AutoTCRPredictor.from_model_dir(model_dir, *, device="auto", batch_size=64) -> AutoTCRPredictor`

Uses AutoTCR.pth (or legacy 10k.pth), config.json, vocab.txt and inference_config.yaml from the directory. Inference does not access the network.

`AutoTCRPredictor.from_settings(settings_or_path)`

Accepts an InferenceSettings object or YAML/JSON path. Relative paths resolve against the settings file. Actual network construction is lazy until prediction or backend.describe().

## predict_sequences

`predict_sequences(table_or_path, *, sequence_column=None) -> pandas.DataFrame`

Input is a non-empty DataFrame or CSV/TSV/gzip path. Common sequence columns are sequence and TcRb_vj. Original row order, metadata and duplicates are retained. Labels are not inputs. Outputs add input_row, prob_class_0, prob_class_1, predicted_label and autoimmune_probability. input_row is regenerated as a zero-based source-row identifier.

## predict_repertoire

`predict_repertoire(table_or_path, *, sample_id=None, sequence_column=None, frequency_column=None, sharp_temperature=10.0) -> RepertoirePrediction`

Requires one processed repertoire with nonnegative finite abundance and positive total. Common columns are TcRb_vj and Freq. A Sample/sample column with multiple distinct IDs is rejected. Sample ID can be explicit or inferred from the table or filename.

| Result member | Content |
|---|---|
| summary | Dictionary of sample ID, counts, total abundance and aggregate scores |
| per_sequence | Table of probabilities, normalized_abundance and ars_contribution |
| save(summary_path=None, predictions_path=None) | Writes either or both CSVs; .gz enables compression |

Summary scores: ars (abundance-weighted class-1 probability), freq_prob (identical scalar), top100_mean, top_diff_mean, softmax_sharp and softmax_freq. Count fields distinguish n_input_records from n_unique_sequences; n_clonotypes is a legacy input-row count.

## predict_cohort

`predict_cohort(manifest_or_path, *, disease_dir=None, healthy_dir=None, sample_column="sample", label_column="true_label", path_column="file_path", file_suffix="_input_seq_vj.csv", sequence_column=None, frequency_column=None, manifest_base_dir=None, skip_missing=False, predictions_dir=None) -> pandas.DataFrame`

Use sample,file_path for unlabelled samples, or sample,true_label for original case/control directory routing. Explicit paths take precedence. Relative paths resolve against the manifest file; DataFrame inputs require manifest_base_dir for relative paths. Labels, if provided, must be binary.

Preserves metadata and replaces scores for successfully processed samples. Identical input paths are inferred once. Missing files fail unless skip_missing=True, which creates status=missing and empty score fields. Optional gzip prediction outputs are linked by predictions_file.

## Settings and backend

`InferenceSettings.from_file(path)`, `from_model_dir(path, **runtime_options)`, `validate(check_paths=True)`, and `model_options()` provide portable local settings.

`TorchBackend(settings).predict_proba(sequences) -> numpy.ndarray` returns an ordered (N,2) probability array. `describe()` strictly loads the model and reports its device, parameter count, vocabulary, fixed length and pooling.

Defaults: batch_size=64, device=auto, positive_class_index=1, long_sequence_policy=error, allow_unsafe_checkpoint=False. The published model uses sent_max_len=32, dropout=0.3 and first-last-avg.

A custom backend can be passed to `AutoTCRPredictor(backend)` for unit testing or another execution engine. LegacyTorchBackend is a compatibility alias.

