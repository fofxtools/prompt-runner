#!/usr/bin/env python3
"""
stable-diffusion.cpp expects FLUX.2's attention Q/K-norm ("RMSNorm") scale
tensors to be named "...norm.key_norm.scale" / "...norm.query_norm.scale" -
both for FluxConfig::detect_from_weights() (which infers num_heads from one
such tensor's size, and silently falls back to a wrong default if it finds
none - causing a SIGFPE deeper in FluxRunner's init) and for the actual
tensor loader (which errors "tensor ... not in model file" for every one of
these it can't find under that name).

Community-quantized GGUFs (e.g. many civitai/huggingface FLUX.2 Klein
conversions) often use ".weight" as the suffix for every tensor uniformly,
including these norm scales. Fix: write a copy of the GGUF with every
"*.norm.{key,query}_norm.weight" tensor renamed (not duplicated - the loader
doesn't need both) to the ".scale" suffix it looks for. Everything else is
copied byte-for-byte, no re-quantization.

Usage:
    python scripts/patch_gguf_head_dim.py path/to/model.gguf
    python scripts/patch_gguf_head_dim.py path/to/model.gguf -o path/to/out.gguf
"""

import argparse
import re
import sys
from pathlib import Path

import gguf

RENAME_PATTERN = re.compile(r"\.norm\.(key_norm|query_norm)\.weight$")


def renamed(name: str) -> str:
    return RENAME_PATTERN.sub(r".norm.\1.scale", name)


def patch(src: str, dst: str) -> None:
    reader = gguf.GGUFReader(src)
    writer = gguf.GGUFWriter(dst, arch="flux2")

    # Note: t.data is already in GGUF's packed byte-shape for quantized types
    # (e.g. Q5_K rows store 176 bytes per 256-element block, not 4096 raw
    # floats). add_tensor's default raw_shape=tensor.shape expects exactly
    # that packed shape and converts it back to logical shape internally via
    # quant_shape_from_byte_shape - passing the logical shape (t.shape)
    # explicitly double-converts and breaks. So: pass t.data with no raw_shape
    # override at all.
    n_renamed = 0
    print(f"Copying {len(reader.tensors)} tensors...")
    for i, t in enumerate(reader.tensors):
        out_name = renamed(t.name)
        if out_name != t.name:
            n_renamed += 1
        writer.add_tensor(out_name, t.data, raw_dtype=t.tensor_type)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(reader.tensors)}")
    print(f"Renamed {n_renamed} norm-scale tensors (.weight -> .scale).")
    if n_renamed == 0:
        print(
            "ERROR: expected to rename some tensors, renamed none - this GGUF "
            "may not need patching, or its tensor names don't match the "
            "expected FLUX.2 pattern.",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Writing header...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"Done: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="Path to the source GGUF to patch")
    parser.add_argument(
        "-o",
        "--output",
        help="Output path (default: <src stem>.sdcpp-patched.gguf next to src)",
    )
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.exists():
        print(f"ERROR: source file not found: {src_path}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        dst_path = Path(args.output)
    else:
        dst_path = src_path.with_suffix("")
        dst_path = dst_path.with_name(dst_path.name + ".sdcpp-patched.gguf")

    patch(str(src_path), str(dst_path))


if __name__ == "__main__":
    main()
