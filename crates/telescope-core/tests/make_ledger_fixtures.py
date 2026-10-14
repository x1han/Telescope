#!/usr/bin/env python3
"""Regenerate the three raw-BAM fixtures behind the ledger-guard tests.

WHY RAW BYTES. The three tests in `src/evidence.rs` that exercise the
production `ledger_deficit() == 0` assertion need records whose `refID` and
`pos` are -1, and whose read name is the two bytes `*\\0`. None of those is
expressible through noodles' record API -- all three fields are `usize` /
non-empty there -- so the fixtures are embedded as encoded BAM bytes and
`bam::io::Writer` is never involved. This script is how those bytes were
produced, and how to re-derive them.

WHAT IT DOES. Writes a minimal single-record BAM per case to `--out-dir` and
prints the Rust `const` arrays, so the copies in `evidence.rs` are diffed
rather than trusted:

    python3 make_ledger_fixtures.py --out-dir /tmp/ledger_fx \\
        --check ../src/evidence.rs

`--check` compares each printed array against the `const` in the given source
file and exits non-zero on any difference. It is a plain substring comparison
-- the array text is the artifact, so the array text is what is compared.

BGZF. The output is gzip-framed with the BC extra subfield, as BAM requires.
An uncompressed but otherwise valid BAM is accepted by samtools and REJECTED
by noodles with "Failed to read BAM header: failed to fill whole buffer", and
samtools will not round-trip such a file either -- `samtools view -b -o out.bam
in.bam` also fails. Framing it correctly here is what makes the fixtures
loadable at all.
"""
import argparse
import os
import struct
import sys
import zlib

# BGZF EOF marker: an empty deflate block in a BGZF-framed gzip member.
BGZF_EOF_MARKER = bytes([
    0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00,
    0x42, 0x43, 0x02, 0x00, 0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00])

TX_LEN = 1000     # @SQ LN
POS = 980         # 0-based start; ref_end = 980+20-1 = 999, +35 slack = 1034
                  # >= 1000, so the record clears the 3' TES window.
CIGAR = [(20, "M")]  # (length, op)
SEQ = "ACGTACGTACGTACGTACGT"
QUAL = [30] * 20

CIGAR_OPS = "MIDNSHP=X"


def bgzf_block(payload):
    """One BGZF block: gzip member carrying the BC extra subfield.

    Total block size is 26 + len(deflate) and must not exceed 65536.
    """
    co = zlib.compressobj(6, zlib.DEFLATED, -zlib.MAX_WBITS)
    comp = co.compress(payload) + co.flush()
    bsize = 26 + len(comp)
    assert bsize <= 65536, bsize
    return (b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff"
            + struct.pack("<H", 6)
            + b"BC" + struct.pack("<H", 2) + struct.pack("<H", bsize - 1)
            + comp
            + struct.pack("<I", zlib.crc32(payload) & 0xffffffff)
            + struct.pack("<I", len(payload) & 0xffffffff))


def bgzf(data):
    out, i = bytearray(), 0
    while i < len(data):
        out += bgzf_block(data[i:i + 0xff00])
        i += 0xff00
    return bytes(out) + BGZF_EOF_MARKER


def cigar_ops(pairs):
    out = b""
    for ln, op in pairs:
        out += struct.pack("<I", (ln << 4) | CIGAR_OPS.index(op))
    return out


def enc_seq(seq):
    code = "=ACMGRSVTWYHKDBN"
    bits = "".join("%x" % code.index(c) for c in seq)
    bits += "0" * (-len(bits) % 2)
    return bytes(int(bits[i:i + 2], 16) for i in range(0, len(bits), 2))


def tag_i(name, val):
    return name.encode() + b"i" + struct.pack("<i", val)


def record(name_bytes, refid=0, pos=POS):
    """One alignment record. FLAG 0 = mapped, forward, primary."""
    rn = name_bytes
    body = struct.pack("<iiBBHHHiiii",
                       refid,         # refID
                       pos,           # pos (0-based start)
                       len(rn),       # l_read_name, uint8
                       60,            # mapq
                       0,             # bin
                       len(CIGAR),    # n_cigar_op
                       0,             # flag
                       len(SEQ),      # l_seq
                       -1, 0, 0)      # next_refID, next_pos, tlen
    body += rn + cigar_ops(CIGAR) + enc_seq(SEQ) + bytes(QUAL) + tag_i("pt", 70)
    return struct.pack("<i", len(body)) + body


def write(path, records):
    text = b"@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:tx1\tLN:%d\n" % TX_LEN
    out = b"BAM\x01" + struct.pack("<i", len(text)) + text
    nm = b"tx1\x00"
    out += struct.pack("<i", 1)                    # n_ref
    out += struct.pack("<i", len(nm)) + nm + struct.pack("<i", TX_LEN)
    for r in records:
        out += r
    open(path, "wb").write(bgzf(out))


def rust_array(name, path):
    b = open(path, "rb").read()
    lines = ["    const %s: &[u8] = &[" % name]
    for i in range(0, len(b), 16):
        lines.append("        " + " ".join("0x%02x," % c for c in b[i:i + 16]))
    lines.append("    ];")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/ledger_fx")
    ap.add_argument("--check", default=None,
                    help="path to evidence.rs; compare the arrays against it")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    written = {
        # refID = -1 on a record whose FLAG says mapped -> `_ => continue` at :570
        "BAM_NO_REFID": os.path.join(args.out_dir, "notid.bam"),
        # pos = -1 -> alignment_start().get() yields nothing -> continue at :577
        "BAM_NO_POS": os.path.join(args.out_dir, "nopos.bam"),
        # read name = "*\0" -> record.name() == None -> continue at :642
        "BAM_NO_NAME": os.path.join(args.out_dir, "nameless.bam"),
    }
    write(written["BAM_NO_REFID"], [record(b"readA\x00", refid=-1)])
    write(written["BAM_NO_POS"], [record(b"readA\x00", pos=-1)])
    write(written["BAM_NO_NAME"], [record(b"*\x00")])

    bad = 0
    src = open(args.check, encoding="utf-8").read() if args.check else None
    for name, path in written.items():
        arr = rust_array(name, path)
        print("/* %s -- %d bytes */" % (os.path.basename(path),
                                        os.path.getsize(path)))
        print(arr)
        print()
        if src is not None and arr not in src:
            print("MISMATCH: %s does not appear verbatim in %s"
                  % (name, args.check), file=sys.stderr)
            bad += 1
    if src is not None:
        print("LEDGER_FIXTURE_CHECK: %d/3 arrays match" % (3 - bad),
              file=sys.stderr)
    sys.exit(1 if bad else 0)