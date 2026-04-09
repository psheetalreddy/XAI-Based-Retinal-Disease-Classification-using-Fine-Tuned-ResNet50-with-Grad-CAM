# An Explainable Retinal Disease classification using ResNet50 and Grad-Cam with ClinicalBert Alignment
### ResNet50 · Grad-CAM · ClinicalBERT · Gradio Dashboard

> An end-to-end deep learning pipeline for multi-class retinal disease classification with systematic explainability analysis and clinical NLP alignment.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-green.svg)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  Overview

This project develops an interpretable diagnostic support system for classifying retinal fundus images into four categories — **Cataract**, **Diabetic Retinopathy**, **Glaucoma**, and **Normal** — achieving **89.61% test accuracy** on a 674-image held-out test set.

Beyond classification accuracy, the system quantitatively evaluates the *quality* of its own explanations through consistency, stability, and clinical alignment analysis — addressing a critical gap in clinical AI adoption.

---

## Project Structure

```
├── 1_Model_Development_ResNet50_FineTuning_GradCAM.ipynb   # Module 1: Training & Grad-CAM
├── 2_Explanation_Analysis.ipynb                             # Module 2: Explainability Analysis
├── 3_ClinicalBERT_Dashboard.ipynb                          # Module 3: NLP Alignment & Dashboard
├── eye-disease-dataset/
│   ├── train/   # cataract / diabetic_retinopathy / glaucoma / normal
│   ├── valid/
│   ├── test/
│   └── cat_to_name.json                                     # Class index → label mapping
├── Models/
    ├── phase1_best_model.keras                              # Phase 1 checkpoint
    └── phase2_best.keras                                    # Phase 2 (final) model
├── Outputs/
    ├── explanation_store.csv                                    # Grad-CAM + predictions for all test images
    ├── consistency_summary.csv
    └── stability_summary.csv
└── requirements.txt
```

---

## System Pipeline

```
Retinal Fundus Image
        ↓
  Preprocessing (224×224, ResNet50 normalization)
        ↓
  ResNet50 (fine-tuned, 2-phase transfer learning)
        ↓
  [Predicted Class · Confidence · Softmax Probabilities]
        ↓
  Grad-CAM (conv5_block3_out → 7×7 heatmap)
        ↓
  Severity Estimation (confidence + disease weight + attention concentration)
        ↓
  Textual Explanation Generation
        ↓
  ClinicalBERT Semantic Alignment Scoring
        ↓
  Gradio Diagnostic Dashboard
```

---

## Results

### Model Performance

| Phase | Test Accuracy |
|---|---|
| Phase 1 — Feature Extraction | 88.70% |
| Phase 2 — Fine-Tuning | **89.61%** |

### Per-Class Classification Report (Phase 2)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Cataract | 0.92 | 0.91 | 0.91 |
| Diabetic Retinopathy | 0.93 | 0.95 | **0.94** |
| Glaucoma | 0.85 | 0.79 | 0.82 |
| Normal | 0.90 | 0.93 | 0.91 |

### Explanation Consistency (Pairwise Cosine Similarity)

| Class | Mean | Std |
|---|---|---|
| Normal | **0.7016** | 0.1103 |
| Cataract | 0.5565 | 0.1995 |
| Glaucoma | 0.4606 | 0.2370 |
| Diabetic Retinopathy | 0.4349 | 0.2288 |

### Explanation Stability (Mean Cosine Similarity Under Gaussian Noise)

| Class | σ=0.01 | σ=0.03 | σ=0.05 | σ=0.10 |
|---|---|---|---|---|
| Cataract | 0.9992 | 0.9989 | 0.9984 | 0.9969 |
| Diabetic Retinopathy | 1.0000 | 0.9995 | 0.9992 | 0.9986 |
| Glaucoma | 0.9999 | 0.9996 | 0.9988 | 0.9970 |
| Normal | 0.9998 | 0.9996 | 0.9993 | 0.9990 |

> **Confidence–Consistency Correlation:** Pearson r = 0.2911, p < 0.0001

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/retinal-disease-explainability.git
cd retinal-disease-explainability
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset
Download the [Eye Disease Retinal Images dataset]([https://www.kaggle.com/](https://ieee-dataport.org/documents/four-public-datasets-explainable-medical-image-classifications)) and organize it as:
```
eye-disease-dataset/
├── train/
│   ├── 1/   # cataract
│   ├── 2/   # diabetic_retinopathy
│   ├── 3/   # glaucoma
│   └── 4/   # normal
├── valid/
└── test/
```

### 4. Run on Google Colab (Recommended)
Mount your Google Drive and update the path variables in each notebook:
```python
BASE_DIR = "/content/drive/MyDrive/retinal-project"
MODEL_PATH = f"{BASE_DIR}/phase2_best.keras"
test_dir   = f"{BASE_DIR}/eye-disease-dataset/test"
```

---

##  Usage

### Module 1 — Train the Model
Run `1_Model_Development_ResNet50_FineTuning_GradCAM.ipynb` end-to-end. This performs two-phase transfer learning and saves `phase2_best.keras`.

### Module 2 — Explainability Analysis
Run `2_Explanation_Analysis.ipynb`. This generates `explanation_store.csv` for all 674 test images and produces four analyses: consistency, stability, average attention maps, and confidence–consistency correlation.

### Module 3 — Dashboard
Run `3_ClinicalBERT_Dashboard.ipynb`. This launches the Gradio interface:
```python
demo.launch(share=True)  # Generates a public URL
```
Upload any retinal fundus image to receive a full diagnostic report.

---

## 🧩 Key Components

| Component | Details |
|---|---|
| **Backbone** | ResNet50, ImageNet pre-trained, `include_top=False` |
| **Custom Head** | GAP → BatchNorm → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax) |
| **Phase 1** | Frozen base, Adam lr=1e-4, 10 epochs |
| **Phase 2** | Last 20 layers unfrozen, Adam lr=1e-5, 15 epochs |
| **Grad-CAM Layer** | `conv5_block3_out` (7×7×2048) |
| **Explanation Store** | 674 rows × [image_id, true_class, predicted_class, confidence, probabilities, 49-value heatmap vector] |
| **ClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT`, [CLS] token cosine similarity |
| **Severity Score** | 0.5×confidence + 0.3×disease_weight + 0.2×attention_concentration |

---

##  Requirements

```
tensorflow==2.15.0
torch>=2.1.0
transformers>=4.38.0
gradio>=4.0.0
scikit-learn>=1.3.0
numpy>=1.26.0
pandas>=2.0.0
opencv-python>=4.9.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.11.0
tqdm>=4.66.0
```

---

## Output Files

| File | Description |
|---|---|
| `phase2_best.keras` | Final fine-tuned model |
| `explanation_store.csv` | Predictions + flattened Grad-CAM heatmaps for all test images |
| `consistency_summary.csv` | Per-class intra-class cosine similarity statistics |
| `stability_summary.csv` | Per-class stability scores across 4 noise levels |
| `consistency_analysis.png` | Bar chart + pairwise similarity heatmap |
| `stability_analysis.png` | Line plot + grouped bar chart |
| `average_attention_maps.png` | Correct vs. incorrect Grad-CAM averages per class |
| `confidence_vs_consistency.png` | Scatter + box plot with Pearson r annotation |

---

## Clinical Disclaimer

This system is developed as an academic research prototype for **decision-support purposes only**. All outputs — predictions, Grad-CAM heatmaps, severity scores, and clinical alignment scores — are experimental and **not a substitute for diagnosis by a qualified ophthalmologist**. Clinical validation through prospective patient trials has not been conducted.

---

## Authors

| Name | Register No. |
|---|---|
| P Sheetal Reddy | 22MIC0080 |
| Khushi Arora | 22MIC0031 |

**Supervisor:** Prof. Arivoli A, School of Computer Science and Engineering, VIT Vellore

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [ResNet50](https://arxiv.org/abs/1512.03385) — He et al., 2016
- [Grad-CAM](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017
- [ClinicalBERT](https://arxiv.org/abs/1904.03323) — Alsentzer et al., 2019
- [Gradio](https://gradio.app) for the interactive dashboard framework
- VIT Vellore, School of Computer Science and Engineering
