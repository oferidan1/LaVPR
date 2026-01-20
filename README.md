# LaVPR: Benchmarking Language and Vision for Place Recognition

Official implementation of **LaVPR**, a comprehensive framework for bridging natural language and computer vision in the context of Visual Place Recognition (VPR).

---

## 🌟 Key Contributions

* **LaVPR Benchmark:** A massive, curated dataset extending standard VPR benchmarks with over **650,000 aligned natural language descriptions**.
* **Multi-Modal Models:** Two distinct architectural approaches:
1. **Multi-Modal Fusion:** Dynamic weighting of image and text features.
2. **Multi-Modal Alignment:** Cross-modal embedding alignment achieving State-of-the-Art (SOTA) performance.


* **Comprehensive Evaluation:** Support for image-only, text-only, and various fusion-based retrieval modes.

---

## 🛠 Setup

### Environment

This codebase has been tested with **PyTorch 2.9.0**, **CUDA 12.6**, and **Xformers**.

```bash
# Create and activate your environment (optional but recommended)
conda create -n lavpr python=3.12
conda activate lavpr

# Install dependencies
pip install -r requirements.txt

```

---

## 📊 Dataset Preparation

To reproduce our results, download the following datasets:

| Dataset | Purpose | Link |
| --- | --- | --- |
| **GSV-Cities** | Training (Source) | [Download](https://github.com/amaralibey/gsv-cities) |
| **MSLS** | Evaluation | [Download](https://github.com/FrederikWarburg/mapillary_sls) |
| **Pittsburgh** | Evaluation | [Download](https://data.ciirc.cvut.cz/public/projects/2015netVLAD/Pittsburgh250k/) |
| **LaVPR** | Text descriptions | Extract: datasets/descriptions.zip to datasets/descriptions |
| --- | --- | ---


---

## 🚀 Training

Training on **GSV-Cities** for 10 epochs takes approximately **10 hours** on a single NVIDIA RTX 3090.

### 1. Image-Text Fusion Model (Dynamic Weighting)

```bash
python train.py --fusion_type=dynamic_weighting \
                --is_text_pooling=1 \
                --vpr_dim=512 \
                --vpr_model_name=mixvpr \
                --text_dim=1024 \
                --text_model_name=BAAI/bge-large-en-v1.5

```

### 2. Image-Text Alignment Model (Cross-Modal)

```bash
python train.py --cross_modal=2 \
                --fusion_type=none \
                --vpr_model_name=Salesforce/blip-itm-base-coco \
                --vpr_dim=256 \
                --is_text_pooling=0 \
                --is_image_pooling=0 \
                --image_size=384 \
                --loss_name=MultiSimilarityLossCM \
                --is_trainable_text_encoder=1 \
                --lora_all_linear=1 \
                --lora_r=64

```

*Checkpoints and logs will be saved automatically to the `/logs` directory.*

---

## 🔍 Evaluation

We provide several evaluation modes to test the versatility of LaVPR.

| Mode | Command Snippet |
| --- | --- |
| **Image Only** | `python eval_vpr.py --encode_mode=image --is_encode_text=0` |
| **Text Only** | `python eval_vpr.py --encode_mode=text --is_encode_image=0` |
| **Fusion (Concat)** | `python eval_vpr.py --is_dual_encoder=1 --dual_encoder_fusion=cat` |
| **Fusion (ADS)** | `python eval_vpr.py --fusion_type=dynamic_weighting --is_text_pooling=1 --model_name=PATH_TO_CKPT` |
| **Cross-Modal** | `python eval_vpr.py --cross_modal=2 --vpr_dim=256 --image_size=384 --text_dim=256 --embeds_dim=256 --vpr_model_name=Salesforce/blip-itm-base-coco --lora_path=checkpoints/blip_lora_all_r64` |

---

## ❤️ Acknowledgements

This repository builds upon several excellent open-source projects:

* [MixVPR](https://github.com/amaralibey/MixVPR) - State-of-the-art VPR architecture.
* [GSV-Cities](https://github.com/amaralibey/gsv-cities) - Large-scale VPR dataset.
* [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) - Standardized VPR evaluation framework.

---


