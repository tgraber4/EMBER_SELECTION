# Example Row: File Guide

All data files below describe the same sample: a benign Win32 file with SHA-256 `0008071d…`.

- **`example_full_row.json`**: The complete row from the JSONL dataset, pretty-printed. It holds all 32 top-level keys: metadata, labels, the 12 raw feature blocks, and the Capa tags (`caps`, `ttps`, `mbc`).
- **`example_row_toplevel.json`**: An outline of the row that lists the same 32 keys in order, with lists and objects emptied and hashes shortened. It shows the row's layout, not its data.
- **`example_row_non_feature_data.json`**: The 20 keys that are never turned into features: hashes, VirusTotal dates, `detection_ratio`, labels, `file_type`, `week_id`, and the Capa tags. Together with the raw feature data file, it makes up the full row.
- **`example_row_raw_feature_data.json`**: The raw features: the 12 feature blocks (`histogram` through `pefilewarnings`) before vectorization, in their original JSON types. These are the inputs to `PEFeatureExtractor.process_raw_features`.
- **`example_row_raw_feature_types.json`**: The type and meaning of every field in the raw feature data file, nested the same way and checked against all 123,024 sampled rows. Each value is either a nested object or a string written as `type[, length N] (meaning)`, and keys like `sections[]` describe the items in a list.
- **`example_row_feature_vector.json`**: The feature vector for this row: the raw feature data after vectorization, as 2,568 float32 values. It also records the `sha256`, `label`, `dim`, and `dtype`.
- **`example_row_summary.md`**: A written walkthrough of what each section of the row means, plus notes on this sample. It refers to an older file name (`PE_both_sampled.jsonl`) and wrongly says the warnings come from `lief` (they come from `pefile`).
- **`example_row_file_guide.md`**: This file, a short guide to what each file in the folder is.
