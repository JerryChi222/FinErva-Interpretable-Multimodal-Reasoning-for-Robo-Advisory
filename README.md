<h1 align="center">🌌 FinErva: Interpretable Multimodal Reasoning for Robo-Advisory</h1>

<p align="center">
  <em>A dataset & lightweight training framework that teaches small models to think like financial analysts.</em>
</p>

<p align="center">
  <img src="framework/CoT-pipeline.png" width="80%" />
</p>

FinErva — short for **FINancial-llm-with-minERVA-wisdom** — is a multimodal Chain-of-Thought (CoT) dataset and training pipeline designed explicitly for *financial* reasoning. It captures two of the most economically important tasks in investment decision-making:

- **Contract & disclosure understanding** (FinErva-Pact)  
- **Candlestick-chart technical analysis** (FinErva-Price)

And here’s the bigger reveal:  
> **FinErva enables models under 0.8B parameters to approach the reasoning ability of human finance professionals** — including step-by-step interpretability — while remaining cost-efficient and deployment-friendly.

This project aims to push forward the next generation of **auditable, multimodal, interpretable, and financially compliant AI systems** for robo-advisory, risk management, and regulation.

---

# 🎯 Key Features

- 🧠 **Multimodal Chain-of-Thought (CoT)**  
  The **first** financial dataset combining contracts, real-world financial images, and candlestick charts *with human-verified reasoning chains*.  

- 📊 **Realistic Financial Context**  
  Includes authentic documents, disclosures, screenshots, K-line charts — not synthetic toy data.  

- 🔍 **Explicit Interpretability**  
  Models trained on FinErva generate transparent, auditable reasoning aligned with financial practices and regulatory expectations.  

- 🪶 **Lightweight Models Only**  
  All experiments use sub-0.8B VLMs, trainable on commodity hardware (RTX 3090 ×2).  

- 🧩 **Two-Stage Training Pipeline**  
  1. **Supervised CoT Learning**  
  2. **Self-CoT Refinement**  
  A scalable recipe for boosting the reasoning quality of small models.  

- 📈 **Expert-Level Performance**  
  Fine-tuned models achieve accuracy comparable to human finance professionals on both subsets.

If you're from the finance world:  
> You will find FinErva to be the *first dataset that truly forces models to “think like analysts.”*

---

## 📦 Installation

Recommended Python version: **Python ≥ 3.9**

Clone the repo:

```bash
git clone https://github.com/JerryChi222/FinErva-Interpretable-Multimodal-Reasoning-for-Robo-Advisory.git
cd FinErva
```

Create environment:

### Conda

```bash
conda create -n finerva python=3.10
conda activate finerva
pip install -r requirements.txt
```

### venv

```bash
python3 -m venv finerva
source finerva/bin/activate
pip install -r requirements.txt
```

---

## 🗂️ Dataset Overview

FinErva contains **7,544** multimodal, manually verified samples across two tracks:

| Subset            | Samples | Description                             |
| ----------------- | ------- | --------------------------------------- |
| **FinErva-Pact**  | 5,488   | Contract & disclosure reasoning         |
| **FinErva-Price** | 2,056   | Candlestick analysis, technical signals |

Each data point includes:

- A real financial image (contracts, charts, screenshots, etc.)
- A finance-oriented QA pair with distractors
- A **human-validated Chain-of-Thought rationale**
- Balanced question types across numerical reasoning, textual comprehension, technical pattern recognition, etc.

📥 **Download dataset on HuggingFace:**\
👉 **\https://huggingface.co/datasets/jerrychi/FinErva**

---

## 🔧 Reproducing Experiments

To reproduce the core results in the FinErva framework, you only need to complete the following three steps.


### 1️⃣ Place Models and Data

Download and place the backbone language models (e.g., FLAN-T5, LaMini-T5, Alpaca-Flan) under:

```text
models/<model-name>/
```

Make sure the corresponding vision encoder (ViT with classification head removed) is available at:

```text
vision_features-pact/     
vision_features-price/
```

And FinErva dataset at:

```text
data-finerva
    ├── pact
    └── price  
```


### 2️⃣ Stage 1 — Supervised-CoT-Training

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --data_root data-finerva/price \
  --model models/lightweight_models/MBZUAI_LaMini-Flan-T5-77M \
  --use_fin price \
  --user_msg rationale \
  --img_type vit \
  --bs 2 \
  --eval_bs 4 \
  --epoch 20 \
  --lr 5e-5 \
  --output_len 512 \
  --use_caption \
  --use_generate \
  --prompt_format QCM-E \
  --output_dir experiments-1210-price-train
```

- `--data_root` Root directory of the training dataset. This directory should contain the processed FinErva data (or a customized replacement dataset).
- `--model` Path to the backbone language model used for fine-tuning. In our experiments, this typically points to a lightweight FLAN-T5–style model.
- `--use_fin` Specifies the FinErva subset used for training.  
  - `price`: FinErva-Price (candlestick chart and technical analysis tasks)  
  - `pact`: FinErva-Pact (contract and disclosure understanding tasks)
- `--user_msg` Specifies the supervision signal used during training.  
  - `rationale` indicates that Chain-of-Thought (reasoning) supervision is enabled.
- `--img_type` Type of visual features used as input.  
- `--bs` Training batch size per step.
- `--eval_bs` Batch size used during evaluation.
- `--epoch` Number of training epochs.
- `--lr` Learning rate for fine-tuning.
- `--output_len` Maximum length of the generated output sequence (including reasoning and answer).
- `--use_caption` Enables the use of image captions as additional textual input during training.
- `--use_generate` Enables generative training mode, allowing the model to generate both reasoning chains and final answers.
- `--prompt_format` Specifies the prompt template format used during training.  
  `QCM-E` corresponds to a multiple-choice question format with explicit reasoning. Others type of prompt shown in utils_prompt.py
- `--output_dir` Directory where training logs and model checkpoints will be saved.

### 3️⃣ Stage 2 — Self-CoT-Learning

```bash
CUDA_VISIBLE_DEVICES=0,1 python main.py \
  --data_root data-/pact \
  --model models/lightweight_models/MBZUAI_LaMini-Flan-T5-783M \
  --use_fin pact \
  --user_msg answer \
  --img_type vit \
  --bs 2 \
  --eval_bs 4 \
  --epoch 20 \
  --lr 5e-5 \
  --output_len 512 \
  --use_caption \
  --use_generate \
  --prompt_format QCMG-EA \
  --output_dir experiments-0806-pact-train \
  --eval_le experiments-0806-pact-train/pact_rationale_models-*model*-*setting*/predictions_ans_eval.json \
  --test_le experiments-0806-pact-train/pact_rationale_models-*model*-*setting*/predictions_ans_test.json
```

- `--eval_le`  
  Path to the evaluation result file generated during training or validation.  
  This file contains the model’s predicted answers and reasoning outputs on the **evaluation (validation) split**, and is used for computing accuracy and reasoning-related metrics.
- `--test_le`  
  Path to the test result file generated after training is completed.  
  This file contains the model’s predicted answers and reasoning outputs on the **held-out test split**, and is used for final performance reporting.


---

## 🧪 Personalized Exploration

FinErva is designed to be extensible.

If you would like to explore the FinErva framework with **your own dataset**, you can do so by replacing the original dataset with your customized data. In particular:

- You need to **encode visual features** for your dataset using the same vision encoder (e.g., ViT) adopted in FinErva.
- The extracted visual features should follow the same format and dimensionality as those used in FinErva-Pact or FinErva-Price.
- Once the visual features are prepared, simply replace the dataset directory (`data-finerva/`) with your own dataset directory.

### Visual Feature Encoding

FinErva provides a standalone script for visual feature extraction:

```text
vision_encoding.py
```

You can run this script to encode visual features for your own dataset before training.

**Run command:**

```bash
python vision_encoding.py \
    --data_root images-finerva-price \
    --output_dir vision_features-fin-price \
    --img_type vit
```

- `--data_root` specifies the directory containing your raw images.
- `--output_dir` specifies where the encoded visual features will be saved.
- `--img_type` selects the vision encoder type (currently supports `vit` and `detr`).

You may use https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth to download model at ./checkpoints/ .

After feature encoding, replace the original visual feature directory with your newly generated one, and proceed with the standard FinErva training pipeline.

This design allows researchers to flexibly adapt FinErva to new financial scenarios, alternative visual modalities, or institution-specific proprietary data, while reusing the same interpretable multimodal reasoning pipeline.

---

## 📚 Citation

If you use FinErva, please cite the following:

```bibtex
@article{FinErva2026,
  title={Interpretable Multimodal Reasoning for Robo-Advisory: The FinErva Framework},
  author={Chi, J.},
  year={2026},
  journal={Frontiers in Artificial Intelligence}
}
```

---

## 📜 License

This project is released under the **Apache License 2.0**.\
You are free to use, modify, and distribute this work in compliance with the terms of the license.

---

## 🤝 Acknowledgements

We gratefully acknowledge the scholars and professionals who provided guidance and annotation support.

---

## 💬 Final Note

FinErva is not just another dataset — it is a blueprint for **trustworthy financial AI**:

- multimodal
- interpretable
- cost-efficient
- analyst-aligned
- audit-ready

Whether you're building robo-advisors, conducting financial research, or exploring multimodal reasoning, FinErva provides the foundation for the *next generation of intelligent, explainable financial systems.*

