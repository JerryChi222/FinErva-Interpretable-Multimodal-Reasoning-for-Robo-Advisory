<h1 align="center">🌌 FinErva: Interpretable Multimodal Reasoning for Robo-Advisory</h1>

<p align="center">
  <em>A dataset & lightweight training framework that teaches small models to think like financial analysts.</em>
</p>

<p align="center">
  <img src="CoT-pipeline.png" width="80%" />
</p>

<p align="center">
  🔗 <strong>Code Repository:</strong>
  <a href="https://github.com/JerryChi222/FinErva-Interpretable-Multimodal-Reasoning-for-Robo-Advisory">
    GitHub – FinErva Framework
  </a>
</p>

---

FinErva — short for **FINancial-llm-with-minERVA-wisdom** — is a multimodal Chain-of-Thought (CoT) dataset designed explicitly for *financial* reasoning. It captures two of the most economically important tasks in investment decision-making:

- **Contract & disclosure understanding** (FinErva-Pact)  
- **Candlestick-chart technical analysis** (FinErva-Price)

And here’s the bigger reveal:  
> **FinErva enables models under 0.8B parameters to approach the reasoning ability of human finance professionals** — including step-by-step interpretability — while remaining cost-efficient and deployment-friendly.

This dataset aims to support research on **auditable, multimodal, interpretable, and financially compliant AI systems** for robo-advisory, risk management, and financial decision support.

---

## 🎯 Key Features

- 🧠 **Multimodal Chain-of-Thought (CoT)**  
  The **first** financial dataset combining contracts, real-world financial images, and candlestick charts with *human-verified reasoning chains*.  

- 📊 **Realistic Financial Context**  
  Includes authentic financial documents, disclosures, screenshots, and K-line charts — not synthetic toy data.  

- 🔍 **Explicit Interpretability**  
  Each sample provides step-by-step reasoning, enabling transparent and auditable financial inference.  

- 🪶 **Lightweight-Model Friendly**  
  Designed to support training and evaluation with sub-0.8B vision–language models.  

- 📈 **Expert-Level Reasoning Signals**  
  Human-curated rationales reflect professional financial analysis practices.

---

## 🗂️ Dataset Overview

FinErva contains **7,544** multimodal, manually verified samples across two subsets:

| Subset            | Samples | Description                             |
| ----------------- | ------- | --------------------------------------- |
| **FinErva-Pact**  | 5,488   | Contract & disclosure understanding     |
| **FinErva-Price** | 2,056   | Candlestick-chart technical analysis    |

Each data point includes:

- A real financial image (contracts, charts, screenshots, etc.)
- A finance-oriented question with distractors
- A **human-validated Chain-of-Thought rationale**
- A single correct answer

The dataset is split into **train / validation / test** sets for both subsets.

---

## 📚 Citation

If you use FinErva, please cite:

```bibtex
@article{FinErva2026,
  title={Interpretable Multimodal Reasoning for Robo-Advisory: The FinErva Framework},
  author={Chi, J.},
  year={2026},
  journal={Frontiers in Artificial Intelligence}
}
```

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