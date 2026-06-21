"""DeepLabV3+ inference (segmentation_models_pytorch) for sat2graph pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ENCODER = "resnet50"
ENCODER_WEIGHTS = "imagenet"
ROAD_CLASS_INDEX = 1  # ['background', 'road']


@dataclass
class DeepLabModel:
    model: object
    device: str
    input_size: tuple[int, int] = (1024, 1024)

    def eval(self):
        self.model.eval()
        return self


def _build_deeplab_architecture():
    import segmentation_models_pytorch as smp

    return smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=None,
        classes=2,
        activation="sigmoid",
    )


def load_deeplab_model(
    weights_path: str | Path,
    device: str | None = None,
) -> DeepLabModel:
    """
    Load DeepLabV3+ checkpoint from the DeepLab notebook.

    Supports:
    - full model via torch.save(model) (best_model.pth)
    - state_dict only (best_model_new.pth from training scripts)
    """
    import torch

    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(f"DeepLab weights not found: {path}")

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(str(path), map_location=dev, weights_only=False)

    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif all(isinstance(k, str) for k in checkpoint.keys()):
            state_dict = checkpoint
        else:
            raise ValueError(f"Unrecognized DeepLab checkpoint dict keys: {list(checkpoint)[:8]}")
        model = _build_deeplab_architecture()
        model.load_state_dict(state_dict)
    else:
        raise TypeError(f"Unsupported DeepLab checkpoint type: {type(checkpoint)}")

    model.to(dev)
    model.eval()
    return DeepLabModel(model=model, device=dev)


def _preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8/float H×W×3 → CHW float32 with ImageNet normalization (ResNet50)."""
    import segmentation_models_pytorch as smp

    if rgb.dtype != np.uint8:
        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8) if rgb.max() <= 1.5 else rgb.astype(np.uint8)
    else:
        arr = rgb

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    x = preprocessing_fn(arr)
    return x.transpose(2, 0, 1).astype(np.float32)


def predict_prob_map(
    dl_model: DeepLabModel,
    rgb: np.ndarray,
    *,
    input_size: tuple[int, int] | None = None,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Road-class probability map in [0, 1] at output_size (default: original H×W).

    DeepLab was trained on native-resolution DeepGlobe tiles (~1024×1024) with
    ImageNet preprocessing — not the D-LinkNet 256× whole-tile resize.
    """
    import torch

    h, w = rgb.shape[:2]
    out_h, out_w = output_size or (h, w)
    infer_h, infer_w = input_size or dl_model.input_size

    if (h, w) != (infer_h, infer_w):
        pil = Image.fromarray(
            (rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        )
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil = pil.resize((infer_w, infer_h), Image.BILINEAR)
        rgb_infer = np.asarray(pil, dtype=np.uint8)
    else:
        rgb_infer = rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    batch = _preprocess_rgb(rgb_infer)
    x = torch.from_numpy(batch).unsqueeze(0).to(dl_model.device)

    with torch.no_grad():
        out = dl_model.model(x)

    prob = out.squeeze(0).detach().cpu().numpy()
    if prob.ndim == 3:
        prob = prob[ROAD_CLASS_INDEX]
    else:
        prob = prob

    if (prob.shape[0], prob.shape[1]) != (out_h, out_w):
        prob = np.array(
            Image.fromarray((prob * 255).astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)
        ) / 255.0

    return prob.astype(np.float32)


@dataclass
class DeepLabOnnxModel:
    session: object
    input_name: str
    output_name: str


def load_deeplab_onnx(onnx_path: str | Path) -> DeepLabOnnxModel:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(0, "CUDAExecutionProvider")
    sess = ort.InferenceSession(str(onnx_path), opts, providers=providers)
    return DeepLabOnnxModel(
        session=sess,
        input_name=sess.get_inputs()[0].name,
        output_name=sess.get_outputs()[0].name,
    )


def predict_prob_map_onnx(
    onnx_model: DeepLabOnnxModel,
    rgb: np.ndarray,
    *,
    input_size: tuple[int, int] = (1024, 1024),
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Road-class probability map via ONNX (ImageNet-preprocessed NCHW input)."""
    h, w = rgb.shape[:2]
    out_h, out_w = output_size or (h, w)
    infer_h, infer_w = input_size

    if (h, w) != (infer_h, infer_w):
        pil = Image.fromarray(
            (rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
        )
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        pil = pil.resize((infer_w, infer_h), Image.BILINEAR)
        rgb_infer = np.asarray(pil, dtype=np.uint8)
    else:
        rgb_infer = rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    batch = _preprocess_rgb(rgb_infer)[np.newaxis, ...]
    out = onnx_model.session.run([onnx_model.output_name], {onnx_model.input_name: batch})[0]
    prob = out.squeeze(0)
    if prob.ndim == 3:
        prob = prob[ROAD_CLASS_INDEX]

    if (prob.shape[0], prob.shape[1]) != (out_h, out_w):
        prob = np.array(
            Image.fromarray((prob * 255).astype(np.uint8)).resize((out_w, out_h), Image.BILINEAR)
        ) / 255.0

    return prob.astype(np.float32)
