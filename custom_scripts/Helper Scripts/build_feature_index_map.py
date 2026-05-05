"""Generate ember_data/feature_index_map.json describing every index 0..dim-1
of the vector produced by thrember.PEFeatureExtractor.

Each entry has:
  - index
  - block (feature-class name: general, histogram, ...)
  - block_index (position within the block)
  - field (human-readable sub-field, or hashed-block tag)
  - hashed (True if index lives inside a FeatureHasher bucket)
"""

import json
from thrember import PEFeatureExtractor
from thrember.features import (
    GeneralFileInfo, ByteHistogram, ByteEntropyHistogram,
    StringExtractor, HeaderFileInfo, SectionInfo, ImportsInfo,
    ExportsInfo, DataDirectories, RichHeader, AuthenticodeSignature,
    PEFormatWarnings,
)

extractor = PEFeatureExtractor()

# Helper to build per-block label lists -------------------------------------

def general_labels():
    return ["size", "entropy", "is_pe",
            "start_bytes[0]", "start_bytes[1]", "start_bytes[2]", "start_bytes[3]"]

def histogram_labels():
    return [f"histogram_0x{b:02X}" for b in range(256)]

def byteentropy_labels():
    # 16 entropy bins x 16 byte bins (flattened row-major)
    return [f"byteentropy_H{h}_byte{b}" for h in range(16) for b in range(16)]

def strings_labels():
    se = StringExtractor()
    regex_keys = sorted(se._regexes.keys())  # matches regex_idxs
    labels = ["numstrings", "avlength", "printables"]
    labels += [f"printabledist_0x{0x20+i:02X}" for i in range(96)]
    labels += ["strings_entropy"]
    labels += [f"string_count_{k}" for k in regex_keys]
    return labels

def header_labels():
    h = HeaderFileInfo()
    scalar = [
        "coff.timestamp", "coff.number_of_sections", "coff.number_of_symbols",
        "coff.sizeof_optional_header", "coff.pointer_to_symbol_table",
        "coff.machine(cat)", "optional.subsystem(cat)",
        "optional.major_image_version", "optional.minor_image_version",
        "optional.major_linker_version", "optional.minor_linker_version",
        "optional.major_operating_system_version", "optional.minor_operating_system_version",
        "optional.major_subsystem_version", "optional.minor_subsystem_version",
        "optional.sizeof_code", "optional.sizeof_headers", "optional.sizeof_image",
        "optional.sizeof_initialized_data", "optional.sizeof_uninitialized_data",
        "optional.sizeof_stack_reserve", "optional.sizeof_stack_commit",
        "optional.sizeof_heap_reserve", "optional.sizeof_heap_commit",
        "optional.address_of_entrypoint", "optional.base_of_code",
        "optional.image_base", "optional.section_alignment", "optional.checksum",
        "optional.number_of_rvas_and_sizes",
    ]
    char_flags   = [f"coff.characteristics:{c}" for c in h._image_characteristics]   # 16
    dll_flags    = [f"optional.dll_characteristics:{c}" for c in h._dll_characteristics]  # 11
    dos_fields   = [f"dos.{m}" for m in h._dos_members]  # 17
    return scalar + char_flags + dll_flags + dos_fields

def section_labels():
    general = [
        "section.n_sections", "section.n_zero_size", "section.n_empty_name",
        "section.n_rx", "section.n_w",
        "section.entropy_max", "section.entropy_min",
        "section.size_ratio_max", "section.size_ratio_min",
        "section.vsize_ratio_max", "section.vsize_ratio_min",
    ]
    hashed = (
        [f"section.sizes_hashed[{i}]"      for i in range(50)] +
        [f"section.vsize_hashed[{i}]"      for i in range(50)] +
        [f"section.entropy_hashed[{i}]"    for i in range(50)] +
        [f"section.characteristics_hashed[{i}]" for i in range(50)] +
        [f"section.entry_name_hashed[{i}]" for i in range(10)]
    )
    overlay = ["overlay.size", "overlay.size_ratio", "overlay.entropy"]
    return general + hashed + overlay

def imports_labels():
    return (
        ["imports.n_functions", "imports.n_libraries"]
        + [f"imports.libraries_hashed[{i}]" for i in range(256)]
        + [f"imports.lib_func_hashed[{i}]"  for i in range(1024)]
    )

def exports_labels():
    # NOTE: first value is len(exports_hashed) (always 128), a quirk of the code
    return ["exports.count_quirk"] + [f"exports.names_hashed[{i}]" for i in range(128)]

def datadirectories_labels():
    dd = DataDirectories()
    out = []
    for name in dd._name_order:
        out.append(f"datadir.{name}.size")
        out.append(f"datadir.{name}.virtual_address")
    out += ["has_relocs", "has_dynamic_relocs"]
    return out

def richheader_labels():
    return ["richheader.number_of_pairs"] + [f"richheader.pair_hashed[{i}]" for i in range(32)]

def authenticode_labels():
    return [
        "auth.num_certs", "auth.self_signed", "auth.empty_program_name",
        "auth.no_countersigner", "auth.parse_error", "auth.chain_max_depth",
        "auth.latest_signing_time", "auth.signing_time_diff",
    ]

def pefilewarnings_labels():
    pw = PEFormatWarnings.__new__(PEFormatWarnings)  # bypass __init__ path logic
    import os as _os
    from pathlib import Path as _Path
    wf = _Path(_os.path.join(_os.path.dirname(__file__), "src", "thrember", "pefile_warnings.txt"))
    pw.__init__(wf)
    # warning_ids is {template: id}; invert
    inv = [None] * (max(pw.warning_ids.values()) + 1)
    for k, v in pw.warning_ids.items():
        inv[v] = k
    labels = [f"pewarn:{w}" for w in inv]
    labels.append("pewarn.total_count")
    return labels

# Hashed-block detection -----------------------------------------------------
HASHED_SUBSTRINGS = (
    "_hashed[", "pair_hashed[",
)

def is_hashed(label: str) -> bool:
    return any(s in label for s in HASHED_SUBSTRINGS)

# Assemble -------------------------------------------------------------------
BLOCKS = [
    ("general",         general_labels()),
    ("histogram",       histogram_labels()),
    ("byteentropy",     byteentropy_labels()),
    ("strings",         strings_labels()),
    ("header",          header_labels()),
    ("section",         section_labels()),
    ("imports",         imports_labels()),
    ("exports",         exports_labels()),
    ("datadirectories", datadirectories_labels()),
    ("richheader",      richheader_labels()),
    ("authenticode",    authenticode_labels()),
    ("pefilewarnings",  pefilewarnings_labels()),
]

# Sanity-check dims against the extractor
expected = {fe.name: fe.dim for fe in extractor.features}
for name, labels in BLOCKS:
    assert len(labels) == expected[name], (
        f"{name}: labeled {len(labels)} but extractor dim is {expected[name]}"
    )

entries = []
idx = 0
block_ranges = {}
for name, labels in BLOCKS:
    start = idx
    for j, lab in enumerate(labels):
        entries.append({
            "index": idx,
            "block": name,
            "block_index": j,
            "field": lab,
            "hashed": is_hashed(lab),
        })
        idx += 1
    block_ranges[name] = [start, idx - 1]

assert idx == extractor.dim, (idx, extractor.dim)

out = {
    "dim": extractor.dim,
    "block_ranges": block_ranges,  # {block: [start, end_inclusive]}
    "entries": entries,
}

path = "ember_data/feature_index_map.json"
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print(f"wrote {path}")
print(f"dim = {out['dim']}")
for b, (s, e) in block_ranges.items():
    n_hash = sum(1 for i in range(s, e + 1) if entries[i]["hashed"])
    print(f"  {b:16s} [{s:>4}..{e:>4}]  size={e-s+1:<5} hashed={n_hash}")
