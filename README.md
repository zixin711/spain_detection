# Wall Mold Detection System

Capture a wall with a phone, segment suspected mold, combine white-light and 395 nm UV evidence, assign an L1–L5 severity level, and generate remediation guidance and robot instructions.

See the [design](docs/DESIGN.md), [data download](docs/DATA_DOWNLOAD.md), and [deployment](docs/DEPLOY.md) guides.

## Features

- **Flexible image inputs:** Provide a white-light image, a UV image, or both. The system identifies the lighting and selects the matching detector; paired images are combined.
- **Evidence-based level ranges:** The output includes `level_range` and guidance for capturing additional images when the evidence cannot support a single level.
  - White light only: L2–L5; L1 is not visible to the naked eye.
  - UV only: L1–L4; L5 can only be marked as suspected.
  - Both modalities: L1–L5.
- **Runs without model weights:** The pipeline falls back to a conventional computer-vision baseline when trained weights are unavailable.
- **False-positive controls:** White-light negative classes include water stains, efflorescence, and dirt. The UV negative class `fluor_artifact` covers optical brighteners, plastic parts, and paint additives.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # For GPU installation, see docs/DEPLOY.md.
sudo apt install -y fonts-noto-cjk       # Required for existing CJK text in annotated images and PDF reports.

./run.sh
```

Connect your phone to the same Wi-Fi network and open the `http://<local-ip>:8000` address printed at startup.

## Data and training

For full GPU installation and training instructions, see [docs/TRAINING.md](docs/TRAINING.md). Common commands:

```bash
python scripts/check_env.py --train
python scripts/download_datasets.py
python scripts/prepare_dataset.py        # Outputs data/processed/{white,uv}/.
python scripts/train.py --domain white --device 1
python scripts/train.py --domain uv    --device 0
python scripts/evaluate.py --domain white
python scripts/export_model.py --domain white --formats onnx engine
```

## Repository layout

```text
app/                    Inference service and algorithms
  modality.py           White-light and UV classification
  inference.py          TensorRT, ONNX, PyTorch, and CV baseline backends
  channels.py           H (UV signal), M (coverage), and L (visual depth) scores
  grading.py            Severity rules, mold index, level ranges, and UNKNOWN fallback
  static/               Mobile web interface
configs/                Detection and grading settings
  grading.yaml          Channel weights, modality limits, area thresholds, and rules
scripts/                Data, training, evaluation, export, calibration, and review tools
docker/                 Inference service image
```

## Interpretation rules

1. **L1 requires UV evidence.** By definition, it cannot be established from a white-light image alone.
2. **Depth is estimated visually.** The 0.1, 0.5, and 2.0 mm thresholds are approximate categories, not measurements; the interface and report must say so.
3. **UV and white-light area thresholds differ.** UV response can extend beyond the visible area, so the two modalities use separately calibrated thresholds.
4. **The area denominator is the usable wall area.** If wall segmentation fails, the system falls back with lower confidence.
5. **Resolve rule conflicts conservatively.** When rules and the mold index disagree, use the more severe level.

## Results on calibration samples

The local `data/paired_calib/` directory contains five same-view white-light and UV pairs, one for each level. The thresholds were calibrated against these samples. The regression test is `tests/test_pipeline.py::test_calibration_samples_reproduce_levels`.

| Reference | White light only | UV only | Both |
|---|---|---|---|
| L1 | `L0_UNVERIFIED` [L0–L1] | **L1** | **L1** |
| L2 | L2 | L2 | L2 |
| L3 | L3 | L3 | L3 |
| L4 | L4 [L4–L5] | L4 [L4–L5] | L4 [L4–L5] |
| L5 | L4 [L4–L5] | L4 [L4–L5] | L4 [L4–L5] |

The white-light-only result cannot establish L1, so the system requests a UV image. L4 and L5 remain a range when the sample does not clearly show structural damage to the substrate.

## Finding from the UV samples

**Wall mold did not emit yellow-green fluorescence under 395 nm UV illumination.** The measured yellow-green fluorescent pixel fraction was **0.0000** across the five UV samples. Instead, optical brighteners in the wall paint emitted blue-white light, while mold absorbed or quenched it and appeared darker. Colony edges appeared as bright, lace-like filaments.

The UV detector therefore uses absolute deviation after illumination-field normalization to capture dark regions and bright filaments; it does not isolate fluorescence by hue. See section 3.5 of [docs/DESIGN.md](docs/DESIGN.md). Recalibrate the thresholds for wall finishes with different brightener content.

## Status and limitations

Milestones M0–M1 are implemented: the end-to-end pipeline runs, the data pipeline is ready, the CV baseline works, and severity thresholds have been calibrated on the sample pairs.

- **High baseline false-positive rate on clean walls:** The CV baseline marked two of five normal BD3 samples as mold (40%), above the acceptance target of 8% or less. A trained detector is needed to distinguish gray dirt from pale mold.
- **Visual depth is uncalibrated:** Depth categories include reference millimeter ranges, but are not physical measurements.
- **Too few samples for mold-index calibration:** There is one image pair per level. The index check can raise severity by at most one level while `configs/grading.yaml` has `index_check.calibrated: false`.
- **SQLite requires local storage:** A database on a network or FUSE mount can produce `disk I/O error`. Set `MOLD_STORAGE_DIR` to a local directory.

Downloaded datasets, calibration images, generated records, and trained weights are excluded from Git. See [docs/DATA_DOWNLOAD.md](docs/DATA_DOWNLOAD.md) for data acquisition.
