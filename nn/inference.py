"""Run D-LinkNet on a satellite tile (for tier_b/c vectorize pipeline)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

InferMode = Literal["resize", "sliding"]


@dataclass
class OnnxRoadModel:
    session: object
    input_name: str
    output_name: str


def load_road_model(weights_path: str | Path):
    from tensorflow.keras.models import load_model

    from .dlinknet_model import CUSTOM_OBJECTS

    return load_model(str(weights_path), custom_objects=CUSTOM_OBJECTS, compile=False)


def load_onnx_model(onnx_path: str | Path) -> OnnxRoadModel:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    sess = ort.InferenceSession(str(onnx_path), opts, providers=providers)
    return OnnxRoadModel(
        session=sess,
        input_name=sess.get_inputs()[0].name,
        output_name=sess.get_outputs()[0].name,
    )


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.float32:
        arr = rgb.astype(np.float32)
        if arr.max() > 1.5:
            arr = arr / 255.0
    else:
        arr = rgb.copy()
    return np.clip(arr, 0.0, 1.0)


def _prob_from_batch(model, batch: np.ndarray) -> np.ndarray:
    if isinstance(model, OnnxRoadModel):
        out = model.session.run([model.output_name], {model.input_name: batch})[0]
        return out[0, ..., 0]
    return model.predict(batch, verbose=0)[0, ..., 0]


def _blend_window(ph: int, pw: int) -> np.ndarray:
    """2D Hann window — smooth overlap between patches."""
    wy = np.hanning(ph) if ph > 1 else np.ones(1, dtype=np.float64)
    wx = np.hanning(pw) if pw > 1 else np.ones(1, dtype=np.float64)
    return np.outer(wy, wx)


def _window_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def _predict_prob_resize(
    model,
    arr: np.ndarray,
    input_size: tuple[int, int],
    out_h: int,
    out_w: int,
) -> np.ndarray:
    pil = Image.fromarray((arr * 255).astype(np.uint8))
    pil = pil.resize((input_size[1], input_size[0]), Image.BILINEAR)
    batch = np.asarray(pil, dtype=np.float32) / 255.0
    batch = batch[np.newaxis, ...]

    prob = _prob_from_batch(model, batch)
    if (out_h, out_w) != input_size:
        prob = np.array(
            Image.fromarray((prob * 255).astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)
        ) / 255.0
    return prob


def _predict_prob_sliding(
    model,
    arr: np.ndarray,
    input_size: tuple[int, int],
    stride: int,
) -> np.ndarray:
    h, w = arr.shape[:2]
    ph, pw = input_size
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    prob_acc = np.zeros((h, w), dtype=np.float64)
    weight_acc = np.zeros((h, w), dtype=np.float64)
    window = _blend_window(ph, pw)

    ys = _window_starts(h, ph, stride)
    xs = _window_starts(w, pw, stride)

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + ph, h)
            x1 = min(x0 + pw, w)
            valid_h = y1 - y0
            valid_w = x1 - x0

            patch = np.zeros((ph, pw, 3), dtype=np.float32)
            patch[:valid_h, :valid_w] = arr[y0:y1, x0:x1]
            batch = patch[np.newaxis, ...]
            prob_patch = _prob_from_batch(model, batch)

            w_slice = window[:valid_h, :valid_w]
            prob_acc[y0:y1, x0:x1] += prob_patch[:valid_h, :valid_w] * w_slice
            weight_acc[y0:y1, x0:x1] += w_slice

    return (prob_acc / np.maximum(weight_acc, 1e-8)).astype(np.float32)


def predict_prob_map(
    model,
    rgb: np.ndarray,
    input_size: tuple[int, int] = (256, 256),
    output_size: tuple[int, int] | None = None,
    *,
    mode: InferMode = "resize",
    stride: int | None = None,
) -> np.ndarray:
    """
    Probability map in [0, 1] at output_size (default: original H×W).

    mode:
      - resize: whole image → input_size → upsample (default; matches training)
      - sliding: native input_size patches + Hann blend (experimental; needs crop-trained model)
    """
    arr = _normalize_rgb(rgb)
    h, w = arr.shape[:2]
    out_h, out_w = output_size or (h, w)

    if mode == "resize":
        prob = _predict_prob_resize(model, arr, input_size, out_h, out_w)
    else:
        step = stride if stride is not None else input_size[0] // 2
        prob = _predict_prob_sliding(model, arr, input_size, step)
        if (out_h, out_w) != (h, w):
            prob = np.array(
                Image.fromarray((prob * 255).astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)
            ) / 255.0

    return prob


def predict_mask(
    model,
    rgb: np.ndarray,
    input_size: tuple[int, int] = (256, 256),
    threshold: float = 0.5,
    output_size: tuple[int, int] | None = None,
    *,
    mode: InferMode = "resize",
    stride: int | None = None,
) -> np.ndarray:
    prob = predict_prob_map(
        model,
        rgb,
        input_size=input_size,
        output_size=output_size,
        mode=mode,
        stride=stride,
    )
    return prob >= threshold
