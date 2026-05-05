# `basic_layout.txt` Attribute Classification


## Included in the feature vector

- `general` (size, entropy, is_pe, start_bytes)
- `histogram` (256-bin normalized byte histogram)
- `byteentropy` (2D byte/entropy histogram)
- `strings` (string stats, printable dist, regex counts)
- `header` (COFF / Optional / DOS header fields)
- `section` (section names, sizes, entropies, overlay)
- `imports` (hashed libraries and imported functions)
- `exports` (hashed exported symbols)
- `datadirectories` (sizes + virtual addrs of data dirs)
- `richheader` (hashed rich-header value pairs)
- `authenticode` (signature / cert chain stats)
- `pefilewarnings` (normalized pefile warning one-hots)

## Not included in the feature vector

- `md5` — file hash (identifier / metadata)
- `sha1` — file hash (identifier / metadata)
- `sha256` — file hash; stored as a row key, not vectorized
- `tlsh` — locality-sensitive hash (metadata)
- `first_submission_date` — VirusTotal metadata
- `last_analysis_date` — VirusTotal metadata
- `detection_ratio` — VirusTotal detection ratio (metadata)
- `label` — binary malicious/benign target (used as `y`, not `X`)
- `file_type` — file-type tag; also a valid `label_type`
- `family` — malware family target (multiclass label)
- `family_confidence` — confidence score for the family label (metadata)
- `behavior` — multilabel target
- `file_property` — multilabel target
- `packer` — multilabel target
- `exploit` — multilabel target
- `group` — threat-group multilabel target
- `week_id` — data-collection week identifier (metadata)
- `caps` — Capa capabilities (threat-intel tags, not vectorized)
- `ttps` — Capa / MITRE ATT&CK techniques (threat-intel tags)
- `mbc` — Malware Behavior Catalogue entries (threat-intel tags)



## Categories of attributes not included in the feature vector

### File identifiers
- `md5`
- `sha1`
- `sha256`
- `tlsh`

### VirusTotal metadata
- `first_submission_date`
- `last_analysis_date`
- `detection_ratio`

### Labels
- `label` — binary malicious/benign
- `family` — malware family (multiclass)
- `behavior` — multilabel
- `file_property` — multilabel
- `packer` — multilabel
- `exploit` — multilabel
- `group` — threat-group multilabel
- `family_confidence` — confidence score for the `family` label


### Capa / threat-intel tags
- `caps` — Capa capabilities
- `ttps` — Capa / MITRE ATT&CK techniques
- `mbc` — Malware Behavior Catalogue entries

### Extra
- `file_type` — file-type tag; also a valid `label_type`
- `week_id` — data-collection week identifier