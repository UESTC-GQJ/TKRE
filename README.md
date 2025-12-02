# TKRE: Two-Stage Knowledge-Guided Pre-training for Relation Extraction

<p align="center">
    <a href="https://ijcai.org/">
        <img alt="IJCAI 2025" src="https://img.shields.io/badge/IJCAI-2025-blue">
    </a>
    <a href="https://github.com/UESTC-GQJ/TKRE/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/UESTC-GQJ/TKRE">
    </a>
    <a href="https://huggingface.co/papers">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Paper-yellow">
    </a>
</p>

This repository contains the source code and datasets for the paper **"Bridging Generative and Discriminative Learning: Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training"**, accepted by **IJCAI 2025**.

## 🌟 Overview

**TKRE** addresses the data scarcity and limited generalization in Few-Shot Relation Extraction (FSRE) by integrating LLMs with traditional Small Language Models (SLMs).

* **Explanation-Driven Knowledge Generation (EDKG)**: Generates logic explanations for relations using LLMs.
* **Schema-Constrained Data Generation (SCDG)**: Generates synthetic training data.
* **Two-Stage Pre-training**:
    1.  **MSLM**: Masked Span Language Modeling.
    2.  **SCL**: Span-Level Contrastive Learning.

## 📂 Repository Structure

```text
TKRE/
├── baseline_models/            # Baseline implementations (e.g., GenPT)
├── codes/
│   ├── data_augmentation/      # [Module 1] LLM-based Data Generation
│   │   ├── automodel_da.py
│   │   ├── corpus.py
│   │   ├── data_augment.sh     # Script for SCDG
│   │   └── data_corpus.sh      # Script for EDKG
│   ├── demonstration/          # Demonstration retrieval utils
│   │   └── recall.sh
│   └── traditional_re/         # [Module 2] Main TKRE Model (Pre-training & Finetuning)
│       ├── fewshot_scripts/
│       │   ├── mslm_roberta.sh # Script for Pre-training (Stage 1 & 2)
│       │   └── run_roberta.sh  # Script for Fine-tuning
│       ├── mslm_model.py       # MSLM implementation
│       ├── cl_model.py         # Contrastive Learning (SCL) implementation
│       ├── train.py            # Main training loop
│       └── utils.py
└── datasets/
    ├── example_data/           # Sample datasets (dev, train, test)
    └── synthetic_data/         # Generated data from LLMs
        ├── llm_corpus.json
        └── llm_da_8.json
```

## 🚀 Requirements
To install the required dependencies:
```
pip install -r requirements.txt
```
Core Dependencies:
* Python >= 3.8
* PyTorch >= 1.13.0
* Transformers >= 4.30.0
* Scikit-learn

## 🛠️ Usage
1. Data Generation (Optional)
If you wish to reproduce the data generation process using LLMs (e.g., Llama-2, GPT-3.5), navigate to the augmentation directory:
```
cd codes/data_augmentation

# 1. Generate Explanation-Driven Knowledge (EDKG)
bash data_corpus.sh

# 2. Generate Schema-Constrained Synthetic Data (SCDG)
bash data_augment.sh
```
2. Knowledge-Guided Pre-training
Perform the two-stage pre-training (MSLM + SCL) on the synthetic and explanation-driven corpus.
```
python mslm_model.py

python cl_model.py
```
3. Task-Oriented Fine-tuningFine-tune the pre-trained model on the $K$-shot support sets ($K=8, 16, 32$).
```
cd codes/traditional_re

# Run the fine-tuning script
bash fewshot_scripts/run_roberta.sh
```

If you use the code in your project, please cite the following paper:
IJCAI'25 ([PDF](https://www.ijcai.org/proceedings/2025/897))
```bibtex
@inproceedings{ijcai2025p897,
  title     = {Bridging Generative and Discriminative Learning: Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training},
  author    = {Guo, Quanjiang and Zhang, Jinchuan and Wang, Sijie and Tian, Ling and Kang, Zhao and Yan, Bin and Xiao, Weidong},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on
               Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {James Kwok},
  pages     = {8068--8076},
  year      = {2025},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2025/897},
  url       = {https://doi.org/10.24963/ijcai.2025/897},
}
```

Please email to guochance1999@163.com for other inquiries.
