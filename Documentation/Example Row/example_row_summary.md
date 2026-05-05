# PE_both_sampled.jsonl — Row Layout Summary

Each line in `PE_both_sampled.jsonl` is **one PE (Portable Executable) file sample** as a JSON object. The object is a single flat dict at the top level, mixing metadata with nested feature groups. This is EMBER-2024's feature schema for Windows PE binaries.

An example row has been written (pretty-printed) to `example_row.json`.

## Top-level sections

### 1. Sample identity & labels
- **`md5`, `sha1`, `sha256`, `tlsh`** — hashes identifying the binary (TLSH is a fuzzy/locality-sensitive hash).
- **`first_submission_date`, `last_analysis_date`** — Unix timestamps from VirusTotal.
- **`detection_ratio`** — e.g. `"0/76"`: how many AV engines flagged it.
- **`label`** — `0` = benign, `1` = malicious (likely `-1` = unlabeled elsewhere).
- **`file_type`** — e.g. `"Win32"`.
- **`family`, `family_confidence`** — malware family if labeled (null for benign).
- **`behavior`, `file_property`, `packer`, `exploit`, `group`** — VT-derived tags.
- **`week_id`** — time bucket for temporal train/test splits.

### 2. Raw byte-level features
- **`histogram`** — 256-entry array: count of each byte value (0x00–0xFF) in the file.
- **`byteentropy`** — 256-entry array: 2D byte/entropy histogram flattened (16 entropy bins × 16 byte-value bins).

### 3. `strings` — printable-string statistics
- `numstrings`, `avlength`, `printables`, `entropy` — counts, average length, and Shannon entropy of extracted ASCII strings.
- `printabledist` — 96-entry histogram over printable ASCII characters.
- `string_counts` — occurrences of suspicious/indicative keywords (`file`, `process`, `mutex`, `password`, `ipv4_addr`, `download`, etc.).

### 4. `general` — top-level PE info
- `size`, `entropy` of full file, `is_pe` flag, and `start_bytes` (first bytes of the file — `[77, 90, 144, 0]` = `MZ\x90\x00`, the DOS header magic).

### 5. `header` — parsed PE header
- **`coff`** — COFF file header: machine type, timestamp, section count, characteristics flags.
- **`optional`** — optional header: subsystem, linker version, code/data sizes, entry-point RVA, image base, stack/heap sizes, DLL characteristics.
- **`dos`** — classic MZ DOS header fields (`e_magic`, `e_lfanew`, etc.).

### 6. `section` — PE sections
- `entry` — section containing the entry point.
- `sections` — list with `name`, `size`, `entropy`, `vsize`, size/vsize ratios, and `props` (permissions like `MEM_EXECUTE`, `MEM_WRITE`).
- `overlay` — data appended after the last section (size + entropy).

### 7. `imports` / `exports`
- **`imports`** — dict mapping each imported DLL → list of imported function names.
- **`exports`** — list of exported functions (often empty for EXEs).

### 8. `datadirectories`
The 16 PE data directory entries (`IMPORT`, `RESOURCE`, `DEBUG`, `IAT`, `TLS`, `SECURITY`, etc.) with their RVA + size. Index 0 also carries `has_relocs` / `has_dynamic_relocs` flags.

### 9. `richheader`
Rich header values — undocumented Microsoft toolchain fingerprint, useful for compiler/linker attribution.

### 10. `authenticode`
Code-signing metadata: `num_certs`, `self_signed`, chain depth, signing time. All zero → unsigned.

### 11. `pefilewarnings`
Parser warnings from `lief` (malformed headers, suspicious fields).

### 12. `caps` — CAPA capabilities
High-level behaviors detected by [Mandiant CAPA](https://github.com/mandiant/capa). Each entry has:
- `Capability` (e.g. *"Create thread"*, *"Log keystrokes via input method manager"*, *"Encode data using xor"*)
- `Namespace` (taxonomy, e.g. `host-interaction/thread/create`)
- `Addrs` — addresses where the capability was detected.

### 13. `ttps` — MITRE ATT&CK
List of `{Tactic, Technique}` pairs mapping capabilities to the ATT&CK framework (e.g. `DISCOVERY / System Information Discovery [T1082]`).

### 14. `mbc` — Malware Behavior Catalog
`{Objective, Behavior}` pairs using the MBC taxonomy (e.g. `PROCESS / Create Mutex [C0042]`, `ANTI-BEHAVIORAL ANALYSIS / Debugger Detection`).

## Example row at a glance

- **Hash:** `0008071d...` / MD5 `4c6fc7d0...`
- **Label:** `0` (benign), **detections:** `0/76`
- **32-bit Windows GUI executable**, ~2 MB, built ~April 2000 (COFF timestamp `955462309`), linked with MSVC 6
- Imports suggest a **DirectX-era Windows game**: `DDRAW.dll`, `DINPUT.dll`, Miles Sound System (`mss32`), Smacker video (`smackw32`)
- Unsigned; entry-point in `.text`
- CAPA still flags generic behaviors (threads, mutexes, XOR encoding, GetTickCount timing) — common in legit software, which is why it's correctly labeled benign despite a noisy capability list.
