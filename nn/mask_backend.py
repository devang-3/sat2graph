"""Unified road mask backends for run_from_place / run_dataset_nn."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from nn.deeplab_inference import load_deeplab_onnx, predict_prob_map_onnx
from nn.inference import InferMode, load_onnx_model, predict_prob_map as dlink_predict

MaskBackendKind = Literal["dlink", "deeplab"]


@dataclass
class MaskBackend:
    kind: MaskBackendKind
    label: str
    model: object
    input_size: tuple[int, int]
    infer_mode: InferMode | None  # None = native (DeepLab)
    default_onnx: Path

    @property
    def mask_backend_tag(self) -> str:
        return f"{self.kind}_onnx_fp32"


def default_onnx_path(kind: MaskBackendKind, root: Path | None = None) -> Path:
    root = root or Path(__file__).resolve().parent.parent
    if kind == "deeplab":
        return root / "models" / "deeplab_fp32.onnx"
    return root / "models" / "roads_extraction_fp32.onnx"


def load_mask_backend(
    kind: MaskBackendKind = "dlink",
    onnx_path: Path | None = None,
    root: Path | None = None,
) -> MaskBackend:
    path = onnx_path or default_onnx_path(kind, root)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")

    if kind == "deeplab":
        return MaskBackend(
            kind="deeplab",
            label="DeepLabV3+ ONNX FP32",
            model=load_deeplab_onnx(path),
            input_size=(1024, 1024),
            infer_mode=None,
            default_onnx=path,
        )

    return MaskBackend(
        kind="dlink",
        label="D-LinkNet ONNX FP32",
        model=load_onnx_model(path),
        input_size=(256, 256),
        infer_mode="resize",
        default_onnx=path,
    )


def predict_road_prob(
    backend: MaskBackend,
    rgb: np.ndarray,
    *,
    output_size: tuple[int, int] | None = None,
    mode: InferMode = "resize",
    stride: int | None = None,
) -> np.ndarray:
    out = output_size or rgb.shape[:2]
    if backend.kind == "deeplab":
        return predict_prob_map_onnx(
            backend.model,
            rgb,
            input_size=backend.input_size,
            output_size=out,
        )
    return dlink_predict(
        backend.model,
        rgb,
        input_size=backend.input_size,
        output_size=out,
        mode=mode,
        stride=stride,
    )
