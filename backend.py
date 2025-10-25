#!/usr/bin/env python3
"""
sora_hex_flag.py
Scan a binary for the string "Sora" (case-insensitive), including UTF-16LE/BE,
and print ONLY:
  - FOUND
  - NOT FOUND

Exit codes:
  0 = scanned successfully, no matches
  1 = scanned successfully, matches found (flag condition)
  2 = usage or runtime error (file missing, I/O error, etc.)
"""

import argparse
import os
import sys
from typing import List, Tuple

# Byte patterns for "Sora"
PATTERNS: List[bytes] = [
    b"Sora",                         # ASCII (we'll search case-insensitive)
    b"S\x00o\x00r\x00a\x00",         # UTF-16LE
    b"\x00S\x00o\x00r\x00a",         # UTF-16BE
]

def scan_file(path: str, targets: List[bytes], chunk_size: int = 1024 * 1024, max_hits: int = 1) -> List[Tuple[int, bytes]]:
    """
    Incrementally scan a file for any of the target byte patterns (case-insensitive for ASCII).
    Returns a list of (absolute_offset, pattern) for the first max_hits matches (defaults to 1 for speed).
    """
    hits: List[Tuple[int, bytes]] = []
    if not targets:
        return hits

    max_pat = max(len(t) for t in targets)
    overlap = max(max_pat - 1, 0)
    offset = 0

    with open(path, "rb") as f:
        prev = b""
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            data = prev + chunk
            lower = data.lower()

            for pat in targets:
                # ASCII search: case-insensitive by searching in .lower() stream
                if pat.isascii():
                    needle = pat.lower()
                else:
                    # UTF-16 patterns are binary-specific; search exact
                    needle = pat

                i = 0
                while True:
                    if pat.isascii():
                        j = lower.find(needle, i)
                    else:
                        j = data.find(needle, i)
                    if j == -1:
                        break
                    abs_pos = offset - len(prev) + j
                    hits.append((abs_pos, pat))
                    if len(hits) >= max_hits:
                        return hits
                    i = j + 1

            # keep last `overlap` bytes for boundary matches
            prev = data[-overlap:] if overlap and len(data) >= overlap else data
            offset += len(chunk)

    return hits

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Flag if a file embeds 'Sora' (ASCII/UTF-16). Prints ONLY 'FOUND' or 'NOT FOUND'."
    )
    ap.add_argument("file", help="Path to the file to scan (.mp4 or any binary).")
    ap.add_argument("--chunk-size", type=int, default=1024 * 1024,
                    help="Read size in bytes per chunk (default: 1048576).")
    # keep fast by default: stop after the first hit
    ap.add_argument("--max-hits", type=int, default=1,
                    help="Stop after this many matches (default: 1).")
    return ap.parse_args()

def main() -> None:
    args = parse_args()

    # validate path
    if not os.path.isfile(args.file):
        print(f"Error: {args.file} not found.", file=sys.stderr)
        sys.exit(2)

    try:
        hits = scan_file(
            path=args.file,
            targets=PATTERNS,
            chunk_size=args.chunk_size,
            max_hits=args.max_hits,
        )
    except Exception as e:
        print(f"Error scanning file: {e}", file=sys.stderr)
        sys.exit(2)

    if hits:
        # stdout ONLY
        print("FOUND")
        sys.exit(1)
    else:
        print("NOT FOUND")
        sys.exit(0)

if __name__ == "__main__":
    main()
