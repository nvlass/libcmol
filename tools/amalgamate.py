#!/usr/bin/env python3
"""
amalgamate.py — generate the single-header release artifact for libcmol.

Output (stdout):  cmol.h  with the full implementation injected under
                  #ifdef CMOL_IMPLEMENTATION.

Usage:
    python3 tools/amalgamate.py > cmol_amalgam.h
    # or via make:
    make amalgamate
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Source files included in the implementation block, topological order.
SOURCES = [
    "src/platform.c",
    "src/arena.c",
    "src/gguf.c",
    "src/tokenizer.c",
    "src/quant.c",
    "src/attn.c",
    "src/model.c",
    "src/sampler.c",
    "src/api.c",
]

# Internal headers folded into the implementation block.
INTERNAL_HEADERS = [
    "src/platform.h",
    "src/cmol_internal.h",
    "src/arena.h",
    "src/gguf.h",
    "src/tokenizer.h",
    "src/quant.h",
    "src/attn.h",
    "src/model.h",
    "src/sampler.h",
]

# Patterns to strip from individual files when merging.
_STRIP_INCLUDE_INTERNAL = re.compile(
    r'^\s*#\s*include\s+"[^"]*"\s*$', re.MULTILINE
)
_STRIP_INCLUDE_GUARD_START = re.compile(
    r'^\s*#\s*ifndef\s+CMOL_\w+_H\b.*\n\s*#\s*define\s+CMOL_\w+_H\b.*\n',
    re.MULTILINE,
)
_STRIP_INCLUDE_GUARD_END = re.compile(
    r'\n\s*#\s*endif\s*/\*\s*CMOL_\w+_H\s*\*/\s*$'
)

def strip_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    text = _STRIP_INCLUDE_INTERNAL.sub("", text)
    text = _STRIP_INCLUDE_GUARD_START.sub("", text)
    text = _STRIP_INCLUDE_GUARD_END.sub("", text)
    return text.strip()


def main() -> None:
    out = sys.stdout

    # ── Header banner ────────────────────────────────────────────────────
    out.write("/*\n")
    out.write(" * cmol.h — libcmol single-header release\n")
    out.write(" *\n")
    out.write(" * AUTO-GENERATED — do not edit.\n")
    out.write(" * Regenerate with:  make amalgamate\n")
    out.write(" *\n")
    out.write(" * Usage:\n")
    out.write(" *   In exactly ONE translation unit:\n")
    out.write(" *       #define CMOL_IMPLEMENTATION\n")
    out.write(" *       #include \"cmol.h\"\n")
    out.write(" *   In all other files:\n")
    out.write(" *       #include \"cmol.h\"\n")
    out.write(" */\n\n")

    # ── Public API header (strip the CMOL_IMPLEMENTATION placeholder) ────
    api_path = ROOT / "include" / "cmol.h"
    api_text = api_path.read_text(encoding="utf-8")

    # Remove the CMOL_IMPLEMENTATION placeholder block
    api_text = re.sub(
        r'#ifdef CMOL_IMPLEMENTATION\s*/\*.*?\*/\s*#endif /\* CMOL_IMPLEMENTATION \*/',
        "",
        api_text,
        flags=re.DOTALL,
    )
    # Remove the closing #endif for CMOL_H so we can append implementation
    api_text = re.sub(r'\n#endif /\* CMOL_H \*/\s*$', "", api_text)

    out.write(api_text.rstrip())
    out.write("\n\n")

    # ── Implementation block ─────────────────────────────────────────────
    out.write("#ifdef CMOL_IMPLEMENTATION\n\n")

    # Internal headers first (type definitions used by all .c files)
    for hdr in INTERNAL_HEADERS:
        path = ROOT / hdr
        if not path.exists():
            print(f"warning: {hdr} not found, skipping", file=sys.stderr)
            continue
        out.write(f"/* ===== {hdr} ===== */\n")
        out.write(strip_file(path))
        out.write("\n\n")

    # Source files
    for src in SOURCES:
        path = ROOT / src
        if not path.exists():
            print(f"warning: {src} not found, skipping", file=sys.stderr)
            continue
        out.write(f"/* ===== {src} ===== */\n")
        out.write(strip_file(path))
        out.write("\n\n")

    out.write("#endif /* CMOL_IMPLEMENTATION */\n\n")
    out.write("#endif /* CMOL_H */\n")


if __name__ == "__main__":
    main()
