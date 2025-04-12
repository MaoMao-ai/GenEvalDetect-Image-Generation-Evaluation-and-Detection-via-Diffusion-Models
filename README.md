# GenEvalDetect-Image-Generation-Evaluation-and-Detection-via-Diffusion-Models

## Overview

This project explores a comprehensive pipeline for diffusion-based image generation, multi-scale evaluation, and AI-generated image detection. It integrates:

- **Stable Diffusion** for subject-driven image synthesis
- **MSFID (Multi-Scale Fréchet Inception Distance)** for enhanced image quality assessment
- **ViT-based binary classifier** to detect AI-generated images
- **Dataset for Training MSFID and ViT-based binary classifier** can be obtained from https://github.com/ZhendongWang6/DIRE

---

## Directory Structure

```
.
├── DreamBooth/                # DreamBooth fine-tuning for subject-specific generation
├── MSFID/                     # MSFID computation using Inception-V3 layers
├── Detection/                 # ViT-based image classifier
│   ├── train_vit_classifier.py
│   ├── predict.py
│   ├── vit_model/             # Saved model checkpoints
│   └── datasets/
│       ├── train/adm, real/
│       ├── val/adm, real/
│       └── test/adm, real/
├── figures/                   # Paper figures and result images
└── paper.tex                  # LaTeX source for the project report
```

---

## 1. Image Generation with Stable Diffusion + DreamBooth

- Fine-tunes Stable Diffusion v1.5 using [DreamBooth](https://arxiv.org/abs/2208.12242) for identity-preserving subject synthesis
- Input: Few reference images of a specific object (e.g., a cat)
- Output: Customized image generation based on prompt + identity
- Reference: https://github.com/JoePenna/Dreambooth-Stable-Diffusion

**Training Usage** (via `diffusers` or `custom_dreambooth.py`):

```bash
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-5" \
  --instance_data_dir="path_to_instance_images" \
  --output_dir="./DreamBooth/output_model"
```

---

## 2. MSFID: Multi-Scale FID Evaluation

- Implements FID across multiple Inception-V3 layers: `Mixed_5d`, `Mixed_6e`, and `Mixed_7c`
- Captures both low-level texture fidelity and high-level semantics
- Provides layer-wise and averaged MSFID scores for realistic assessment

**Evaluation Usage**:

```bash
python compute_msfid.py --real_dir real/ --fake_dir generated/ --batch_size 50
```

---

## 3. ViT-Based Detection of AI-Generated Images

- Binary classifier using Vision Transformer (`vit-base-patch16-224-in21k`)
- Dataset is organized as `imagefolder` format with classes: `adm` (fake), `real` (authentic)
- Achieves ~95% accuracy in detecting diffusion-generated images

### Training

```bash
python train_vit_classifier.py
```

### Inference

```bash
python predict.py
```

Example output:

```
--- Predictions for: ./Detection/fake ---
0.png: adm
1.png: adm
...

--- Predictions for: ./Detection/real ---
0.JPEG: real
1.JPEG: real
...
```

---

## Dependencies

- Python 3.10+
- PyTorch 2.x
- Hugging Face `transformers` and `datasets`
- `torchvision`, `scipy`, `Pillow`, `scikit-learn`, `matplotlib`

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Results

| Task                | Method               | Metric         | Score     |
|---------------------|----------------------|----------------|-----------|
| Generation Quality  | Stable Diffusion     | MSFID (avg)    | 62.77     |
| Detection Accuracy  | ViT-Base Binary Clf  | Accuracy       | 95%       |

---

## Citation

If you find this project useful, please consider citing:

```
@misc{mao2025genevaldetect,
  title={GenEvalDetect: Image Generation, Evaluation, and Detection via Diffusion Models},
  author={Yiming Mao and Enmu Liu},
  year={2025},
  note={Course Project, ECE 580}
}
```

---

## License

MIT License
