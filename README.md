# LaVPR: Benchmarking Language and Vision for Place Recognition

> 🏆 Accepted to **ECCV 2026**

Official implementation of **LaVPR**, a comprehensive framework for bridging natural language and computer vision in the context of Visual Place Recognition (VPR).
[Paper link](https://www.arxiv.org/abs/2602.03253)

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
| **LaVPR** | Text descriptions | Extract: datasets/descriptions.zip to: datasets/descriptions|
| **LaVPR MSLS-Blur**| Blur augmentation (Will be provided upon paper acceptance) | Copy folder: datasets/msls_subsets/query_blur to: msls/val dataset location|
| **LaVPR MSLS-Weather** | Weather augmentation (Will be provided upon paper acceptance) | Copy folder: datasets/msls_subsets/query_weather to: msls/val dataset location|

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
                --text_model_name=BAAI/bge-large-en-v1.5 \
                --train_csv=datasets/descriptions/gsv_cities_descriptions.csv \
                --image_root=PATH_TO_GSV_CITIES_DATASET_LOCATION \
                --val_csv=datasets/descriptions/pitts30k_val_800_queries.csv \
                --val_image_root=PATH_TO_PITTS30K_VAL_DATASET_LOCATION

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
                --lora_r=64 \
                --train_csv=datasets/descriptions/gsv_cities_descriptions.csv \
                --image_root=PATH_TO_GSV_CITIES_DATASET_LOCATION \
                --val_csv=datasets/descriptions/pitts30k_val_800_queries.csv \
                --val_image_root=PATH_TO_PITTS30K_VAL_DATASET_LOCATION

```

*Checkpoints and logs will be saved automatically to the `/logs` directory.*

---

## 🔍 Evaluation

We provide several evaluation modes to test the versatility of LaVPR.

### 📂 Directory Structure
To ensure the paths are mapped correctly, organize your local dataset as follows:

```text
data/
└── amstertime/
    └── test/               <-- image_root
        ├── database/       <-- database_folder
        └── queries/        <-- queries_folder
```

```text
datasets/
└── descriptions/    
    amstertime_descriptions.csv              <-- amstertime descriptions texts
    amstertime_descriptions_subset.csv       <-- amstertime descriptions subset texts
    gsv_cities_descriptions.csv              <-- gsv cities descriptions texts
    msls_challenge_descriptions.csv          <-- msls challenge descriptions texts
    msls_val_descriptions.csv                <-- msls val descriptions texts
    msls_val_descriptions_blur.csv           <-- msls val descriptions blur texts
    msls_val_descriptions_weather.csv        <-- msls val descriptions weather texts
    pitts30k_test_descriptions.csv           <-- pitts30k test descriptions texts
    pitts30k_val_800_queries.csv             <-- pitts30k val 800 queries texts
    pitts30k_val_descriptions.csv            <-- pitts30k val descriptions texts
   
```

| Mode | Command Snippet |
| --- | --- |
| **Image Only** | `python eval_vpr.py --encode_mode=image --is_encode_text=0 --database_folder=PATH_TO_DB_IMAGES --queries_folder=PATH_TO_QUERY_IMAGES --image_root=PATH_TO_IMAGE_ROOT --queries_csv=PATH_TO_DESCRIPTION_CSV` |
| **Text Only** | `python eval_vpr.py --encode_mode=text --is_encode_image=0 --database_folder=PATH_TO_DB_IMAGES --queries_folder=PATH_TO_QUERY_IMAGES --image_root=PATH_TO_IMAGE_ROOT --queries_csv=PATH_TO_DESCRIPTION_CSV` |
| **Fusion (Concat)** | `python eval_vpr.py --is_dual_encoder=1 --dual_encoder_fusion=cat --database_folder=PATH_TO_DB_IMAGES --queries_folder=PATH_TO_QUERY_IMAGES --image_root=PATH_TO_IMAGE_ROOT --queries_csv=PATH_TO_DESCRIPTION_CSV` |
| **Fusion (ADS)** | `python eval_vpr.py --fusion_type=dynamic_weighting --is_text_pooling=1 --model_name=PATH_TO_CKPT --database_folder=PATH_TO_DB_IMAGES --queries_folder=PATH_TO_QUERY_IMAGES --image_root=PATH_TO_IMAGE_ROOT --queries_csv=PATH_TO_DESCRIPTION_CSV` |
| **Cross-Modal** | `python eval_vpr.py --cross_modal=2 --vpr_dim=256 --image_size=384 --text_dim=256 --embeds_dim=256 --vpr_model_name=Salesforce/blip-itm-base-coco --lora_path=checkpoints/blip_lora_all_r64 --database_folder=PATH_TO_DB_IMAGES --queries_folder=PATH_TO_QUERY_IMAGES --image_root=PATH_TO_IMAGES --queries_csv=PATH_TO_DESCRIPTION_CSV` |

---

## Licensing and Data Usage

This repository contains two types of assets: original text annotations and modified image sequences.

| Component | License | Notes |
| :--- | :--- | :--- |
| **Text Annotations** | [CC BY 4.0](LICENSE) | Original creative work by the author. |
| **Modified MSLS Images** | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) | Derivative work based on the MSLS dataset. |

### Third-Party Dataset References
While this repository provides annotations for the following datasets, it **does not** redistribute the original images for them. Users are responsible for ensuring compliance with the respective source licenses:

| Dataset | Original License | Status |
| :--- | :--- | :--- |
| **GSV Cities** | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) | Referenced |
| **Amstertime** | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) | Referenced |
| **Pitts30k** | Restricted (GSV Terms) | Referenced |

---

## ❤️ Acknowledgements

This repository builds upon several excellent open-source projects:

* [MixVPR](https://github.com/amaralibey/MixVPR) - State-of-the-art VPR architecture.
* [GSV-Cities](https://github.com/amaralibey/gsv-cities) - Large-scale VPR dataset.
* [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) - Standardized VPR evaluation framework.
  
## 📖 Cite us

If you find our work useful in your research, please consider citing:

```bibtex
@article{idan2026lavpr,
  title={LaVPR: Benchmarking Language and Vision for Place Recognition},
  author={Idan, Ofer and Badur, Dan and Keller, Yosi and Shavit, Yoli},
  journal={arXiv preprint arXiv:2602.03253},
  year={2026}
}


