# Input and output formats

All commands accept comma- or tab-delimited tables, including gzip-compressed files. A custom sequence or frequency column can be named explicitly.

## Sequence table

Required: `sequence` or `TcRb_vj`.

Optional fields such as `true_label`, `split`, or project metadata are retained. Labels do not enter the model. Input strings should concatenate the original TRBV gene, CDR3β amino-acid sequence and TRBJ gene without adding new separators.

## One repertoire

Required: `TcRb_vj` and `Freq`, or equivalent explicit column names. Optional: `Sample`, `TcRb`, `V`, `J`. Each file represents one processed repertoire.

Frequencies must be finite and nonnegative, with positive total. Zero-frequency and duplicate records are retained. Sequence inference uses the structured sequence column; V/J columns are not independently substituted.

## Cohort manifest

Recommended for new samples: `sample,file_path`. Real labels are not required.

For original evaluation data: `project,disease,sample,true_label` plus optional previous score columns. With no explicit file_path, label 1 selects disease_dir and label 0 selects healthy_dir. Previous score columns are overwritten for successful predictions.

An explicit file_path always takes precedence. Relative paths resolve against the manifest directory. When passing a DataFrame, supply manifest_base_dir for relative paths.

## Outputs

Sequence results preserve input order and add zero-based input_row, probability columns and predicted_label. The positive class is index 1.

Repertoire summaries contain ARS and aggregate compatibility scores, input record count, unique sequence count and total abundance. The scalar freq_prob column equals ARS; the original two-class list output is not stored as a string.

Optional per-sequence repertoire results include normalized_abundance and ars_contribution. Each cohort summary has a status field. Skipped missing files have status=missing and missing score values, including previously provided scores.

Output paths are created as needed. Existing output CSVs are replaced when the same destination is reused. Outputs may contain local resolved input paths; inspect them before sharing results.
