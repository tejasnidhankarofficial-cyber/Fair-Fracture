# 🦴 Fair-Fracture: Bone Fracture Diagnosis with YOLOv8 + U-Net

A hybrid deep learning framework for comprehensive bone fracture diagnosis from X-ray images, combining **YOLOv8 object detection** (multi-class fracture classification) with **U-Net semantic segmentation** (pixel-level fracture mapping).

> Built as part of the Applied Machine Learning course at Stevens Institute of Technology.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Limitations & Future Work](#limitations--future-work)
- [Team](#team)

---

## Overview

Bone fractures account for an estimated **178 million new cases per year** worldwide (WHO). Manual X-ray review is time-consuming and error-prone, especially under high patient load. Fair-Fracture automates this process by running two complementary models in parallel:

| Approach | Outputs |
|---|---|
| CNN Classifier only | Class label |
| Object Detector (YOLO) | Class + bounding box |
| Segmentation model | Pixel-level mask |
| **Fair-Fracture (ours)** | **Class + bounding box + pixel mask** |

The system is designed to **support**, not replace, radiologist judgment — providing three independent evidence signals that can be cross-verified.

---

## Architecture

```
Input X-Ray
     │
     ├──────────────────┬──────────────────┐
     │                  │                  │
     ▼                  ▼                  │
 YOLOv8-Nano        U-Net                 │
 (Detection)     (Segmentation)           │
     │                  │                  │
     ▼                  ▼                  │
 Class label +     Binary pixel            │
 Bounding box         mask                 │
     │                  │                  │
     └──────────────────┘                  │
              │                            │
              ▼                            │
     Gradio Dashboard (fused output) ◄─────┘
```

- **YOLOv8-Nano** identifies fracture type and location (bounding box) across 17 fracture classes.
- **U-Net** produces a pixel-level binary mask showing the exact extent of the fracture region.
- Where bounding boxes from both models **agree**, confidence is high. Where they **disagree**, the case is flagged for expert review.

---

## Datasets

### FracAtlas (U-Net branch)
- 4,083 anonymized musculoskeletal radiographs from hospitals in Bangladesh
- Annotated by two expert radiologists, validated by an orthopedic surgeon
- Pixel-level polygon masks for fracture regions (hand, leg, hip, shoulder)
- 82.4% non-fractured / 17.6% fractured (realistic clinical prevalence)
- Split: 70% train / 15% validation / 15% test

### Roboflow Bone Fracture Dataset (YOLOv8 branch)
- 2,000 annotated X-ray images across **17 fracture classes**:
  `avulsion`, `closed-simple`, `comminuted`, `compression-crush`, `fracture-dislocation`, `greenstick`, `hairline`, `impacted`, `intra-articular`, `longitudinal`, `oblique`, `open-compound`, `pathological`, `segmental`, `spiral`, `stress`, `transverse`
- Native YOLOv8 format with `data.yaml` config

---

## Results

### U-Net Segmentation

| Metric | Score |
|---|---|
| Test Pixel Accuracy | **99.26%** |
| Test Loss (BCE + Dice) | 0.3742 |
| Dice Score | ~0.30 |

> Note: High pixel accuracy is expected given that fracture pixels make up <5% of an X-ray. The Dice score is the more meaningful metric here, as it directly measures overlap on the fracture region.

### YOLOv8 Detection (overall)

| Metric | Score |
|---|---|
| Precision | 0.373 |
| Recall | 0.423 |
| mAP50 | **0.364** |
| mAP50-95 | 0.303 |

### YOLOv8 Per-Class Performance (selected)

| Class | mAP50 |
|---|---|
| Closed-simple | **0.891** |
| Compression-crush | **0.844** |
| Stress | 0.595 |
| Open-compound | 0.525 |
| Greenstick | 0.415 |
| Comminuted | 0.411 |
| Longitudinal | 0.081 |
| Oblique | 0.135 |

Performance is strongest on well-represented classes with visually distinctive patterns. Rare classes (7–18 training examples) show lower scores due to data scarcity and visual similarity to normal bone structures.

---

## Installation

```bash
git clone https://github.com/<your-org>/fair-fracture.git
cd fair-fracture
pip install -r requirements.txt
```

### Requirements

```
torch
torchvision
ultralytics
opencv-python
gradio
Pillow
numpy
```

> Training was performed on an Apple M-series GPU. CUDA is supported via PyTorch and Ultralytics.

---

## Usage

### Run the Gradio Dashboard (Inference)

```bash
python app.py
```

Upload an X-ray image and the dashboard will display:
- **YOLOv8 panel**: detected fracture types with bounding boxes and confidence scores
- **U-Net panel**: pixel-level segmentation mask overlaid on the original image
- **Detected Fracture Types** text box listing all identified classes

### Train U-Net

```bash
python train_unet.py \
  --data_dir /path/to/FracAtlas \
  --epochs 30 \
  --lr 1e-4 \
  --batch_size 8
```

### Train YOLOv8

```bash
python train_yolo.py \
  --data data.yaml \
  --epochs 50 \
  --imgsz 640 \
  --model yolov8n.pt
```

---

## Project Structure

```
fair-fracture/
├── app.py                  # Gradio inference dashboard
├── train_unet.py           # U-Net training script
├── train_yolo.py           # YOLOv8 training script
├── models/
│   ├── unet.py             # U-Net architecture (PyTorch, from scratch)
│   └── yolo/               # YOLOv8 weights and config
├── data/
│   ├── fracatlas/          # FracAtlas dataset (U-Net)
│   └── roboflow/           # Roboflow Bone Fracture dataset (YOLOv8)
├── utils/
│   ├── dataset.py          # FracAtlas DataLoader & mask generation
│   └── inference.py        # Inference pipeline & contour filtering
├── checkpoints/            # Saved model weights
├── requirements.txt
└── README.md
```

---

## Model Details

### U-Net
- Built from scratch in PyTorch
- Encoder: 4 downsampling stages, channel depth 64 → 128 → 256 → 512 → 1024
- Decoder: 4 upsampling stages with skip-connection concatenation
- Loss: `0.5 × BCE + 0.5 × Dice` (handles extreme class imbalance)
- Optimizer: Adam (lr = 1e-4), 30 epochs, batch size 8
- Input: 578 × 578, output: binary pixel mask
- Inference threshold: **0.30** (lower than default 0.5 to prioritize recall in a screening context)

### YOLOv8-Nano
- Pre-trained on COCO (`yolov8n.pt`), fine-tuned on Roboflow Bone Fracture dataset
- ~3.2M parameters; runs on standard hardware without a dedicated GPU
- Anchor-free detection head with separated classification and regression branches
- Input: 640 × 640 with mosaic augmentation, random flip, HSV perturbation
- Loss: Distribution Focal Loss (DFL) + CIoU + Binary Cross-Entropy

---

## Limitations & Future Work

**Current Limitations:**
- Class imbalance in the Roboflow dataset (some classes have <20 training examples)
- Single-view input only; clinical practice typically uses two orthogonal X-ray views
- No prospective clinical validation against radiologist interpretations
- Fairness audit not yet performed — dataset is geographically limited to Bangladesh

**Planned Improvements:**
- Expand to 200–500 images per fracture class
- Add multi-view (AP + lateral) fusion for better alignment with clinical workflow
- Systematic fairness audit stratified by patient age, sex, anatomical site, and hardware vendor
- Extend to volumetric CT data using 3D U-Net variants
- Prospective clinical study comparing system performance against radiologist reads on consecutive ED X-rays

---

## Team

| Name | Department | Email |
|---|---|---|
| Agnim Gupta | Computer Science | agupta57@stevens.edu |
| Jay Gajjar | Mathematical Science | jgajjar@stevens.edu |
| Tejas Nidhankar | Electrical & Computer Engineering | tnidhank@stevens.edu |

Stevens Institute of Technology, Hoboken, NJ

---

## Citation

If you use this work, please cite:

```bibtex
@article{fair-fracture-2024,
  title   = {Fair-Fracture: A YOLOv8 and U-Net Framework for Bone Fracture Diagnosis},
  author  = {Gupta, Agnim and Gajjar, Jay and Nidhankar, Tejas},
  school  = {Stevens Institute of Technology},
  year    = {2024}
}
```

---

## Acknowledgements

- [FracAtlas Dataset](https://github.com/iFtekharul-Islam/FracAtlas) — Abedeen et al., 2023
- [Roboflow Universe](https://universe.roboflow.com) — Bone Fracture dataset
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [U-Net](https://arxiv.org/abs/1505.04597) — Ronneberger et al., 2015
