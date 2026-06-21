#!/usr/bin/env python3
"""Convert roads_extraction_fp32.onnx → fp16 / fp4 (low-bit for inference)."""

from __future__ import annotations

import argparse
from pathlib import Path


def convert_fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(dst))
    print(f"FP16 saved: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def _count_matmul_nbits(path: Path) -> int:
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return sum(1 for n in model.graph.node if n.op_type == "MatMulNBits")


def _cleanup_sidecars(path: Path) -> None:
    side = path.with_suffix(path.suffix + ".data")
    if side.is_file():
        side.unlink()


def convert_fp4(src: Path, dst: Path, block_size: int = 128) -> str:
    """Low-bit export for compare_models.py.

    1. Try MatMulNBits 4-bit (works on LLM-style MatMul graphs).
    2. D-LinkNet is Conv-only → fall back to dynamic INT8 weight quant (~35 MB).
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        DefaultWeightOnlyQuantConfig,
        MatMulNBitsQuantizer,
    )
    from onnxruntime.quantization import QuantFormat

    dst.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_sidecars(dst)

    tmp = dst.with_suffix(".matmulnbits.onnx")
    cfg = DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=False,
        quant_format=QuantFormat.QOperator,
        bits=4,
    )
    quant = MatMulNBitsQuantizer(str(src), algo_config=cfg)
    quant.process()
    quant.model.save_model_to_file(str(tmp), True)

    nbits = _count_matmul_nbits(tmp)
    if nbits > 0:
        tmp.replace(dst)
        _cleanup_sidecars(dst)
        print(f"FP4 MatMulNBits ({nbits} ops) saved: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        return "matmul_nbits_4bit"

    tmp.unlink(missing_ok=True)
    _cleanup_sidecars(tmp)
    print("MatMulNBits: no MatMul layers (Conv CNN) — using dynamic INT8 weight quant")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    print(f"FP4 (INT8 dynamic fallback) saved: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return "int8_dynamic_fallback"


def main():
    root = Path(__file__).parent
    p = argparse.ArgumentParser(description="Quantize ONNX road model")
    p.add_argument("--src", type=Path, default=root / "models" / "roads_extraction_fp32.onnx")
    p.add_argument("--fp16-out", type=Path, default=root / "models" / "roads_extraction_fp16.onnx")
    p.add_argument("--fp4-out", type=Path, default=root / "models" / "roads_extraction_fp4.onnx")
    p.add_argument("--only", choices=["fp16", "fp4", "all"], default="all")
    p.add_argument("--block-size", type=int, default=128)
    args = p.parse_args()

    if not args.src.is_file():
        raise SystemExit(f"Source not found: {args.src}")

    if args.only in ("fp16", "all"):
        convert_fp16(args.src, args.fp16_out)
    if args.only in ("fp4", "all"):
        mode = convert_fp4(args.src, args.fp4_out, block_size=args.block_size)
        print(f"fp4 mode: {mode}")


if __name__ == "__main__":
    main()
