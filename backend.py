import argparse, sys, os

def hexdump_slice(buf: bytes, start: int, length: int = 64):
    s = max(0, start - 32)
    e = min(len(buf), start + length - 32)
    window = buf[s:e]
    # hex
    hexbytes = " ".join(f"{b:02x}" for b in window)
    # ascii printable
    ascii_print = "".join(chr(b) if 32 <= b <= 126 else "." for b in window)
    # caret under match position
    caret_pos = (start - s) * 3  # each "xx " is 3 chars
    caret_line = " " * caret_pos + "^"
    return hexbytes, ascii_print, caret_line

def scan_file(path: str, targets: list[bytes], chunk_size: int = 1024 * 1024):
    hits = []
    # read in chunks with overlap to catch boundary matches
    max_pat = max(len(t) for t in targets)
    overlap = max_pat - 1
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
                pat_lower = pat.lower()
                i = 0
                while True:
                    j = lower.find(pat_lower, i)
                    if j == -1:
                        break
                    abs_pos = offset - len(prev) + j
                    hits.append((abs_pos, pat))
                    i = j + 1
            # prepare next overlap
            if len(data) >= overlap:
                prev = data[-overlap:]
            else:
                prev = data
            offset += len(chunk)
    return hits

def main():
    ap = argparse.ArgumentParser(description="Flag MP4 if it embeds 'Sora' (any case).")
    ap.add_argument("file", help="Path to the .mp4 (or any binary) to scan")
    ap.add_argument("--flag-only", action="store_true",
                    help="Print FOUND/NOT FOUND and exit code (1 if found, 0 if not).")
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        print(f"Error: {args.file} not found.", file=sys.stderr)
        sys.exit(2)

    # patterns: ASCII 'Sora' and UTF-16LE/BE variants
    targets = [
        b"Sora",          # ASCII (case-insensitive search is applied)
        b"S\x00o\x00r\x00a\x00",  # UTF-16LE
        b"\x00S\x00o\x00r\x00a",  # UTF-16BE
    ]

    hits = scan_file(args.file, targets)

    if args.flag_only:
        if hits:
            print("FOUND")
            sys.exit(1)  # non-zero so it’s easy to alert in scripts/CI
        else:
            print("NOT FOUND")
            sys.exit(0)

    if not hits:
        print("No occurrences of 'Sora' found (ASCII/UTF-16).")
        sys.exit(0)

    print(f"Found {len(hits)} occurrence(s):")
    with open(args.file, "rb") as f:
        for idx, (pos, pat) in enumerate(hits, 1):
            f.seek(max(0, pos - 32))
            snippet = f.read(96)
            hexbytes, ascii_print, caret_line = hexdump_slice(snippet, 32)
            print(f"\n[{idx}] Offset: {pos} (0x{pos:08x})  Pattern: {pat!r}")
            print(hexbytes)
            print(ascii_print)
            print(caret_line)
    # Exit 1 to indicate a flag condition
    sys.exit(1)

if __name__ == "__main__":
    main()
