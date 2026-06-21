# nn/ — road mask models

Two backends wired through `mask_backend.py` → `run_from_place.py`:

| Backend | ONNX | Input | Preprocess |
|---------|------|-------|------------|
| `deeplab` (default) | `models/deeplab_fp32.onnx` | 1024×1024 | ImageNet ResNet50 norm |
| `dlink` | `models/roads_extraction_fp32.onnx` | 256×256 | RGB [0,1], whole-tile resize |

## Export DeepLab to ONNX

```bash
pip install -r ../requirements-deeplab.txt
python3 ../convert_deeplab_onnx.py
# reads models/best_model.pth → models/deeplab_fp32.onnx
```

## Train D-LinkNet (Kaggle)

1. Add Mass Roads or flat `*_sat.jpg` + `*_mask.png` dataset on Kaggle
2. Open `notebook/dl-linknet.ipynb`, set `DATA_ROOT`, run
3. Export to ONNX for `--model dlink`

## Train DeepLabV3+

Open `notebook/road-extraction-from-satellite-images-deeplabv3.ipynb` (DeepGlobe-style, native 1024).

After training: `torch.save(model, 'best_model.pth')` then run `convert_deeplab_onnx.py`.

## Inference (Python)

```python
from nn.mask_backend import load_mask_backend, predict_road_prob
from PIL import Image
import numpy as np

rgb = np.array(Image.open("tile.jpg").convert("RGB"))
backend = load_mask_backend("deeplab")
prob = predict_road_prob(backend, rgb)
```
