# Multimodal Multi-label Classification with ViT & TinyBERT

This project implements a multimodal learning pipeline for multi-label classification tasks, leveraging **ViT** for visual input and **TinyBERT** for textual input. It integrates data preprocessing, model definition, training, evaluation, and inference, optimized for compact models under 100MB when required.

---

## Project Structure

```
.
├── Data.py                   # Data loading & preprocessing
├── final_multimodality.pth  # Final trained multimodal model (145MB)
├── result.csv     # Submission CSV (test predictions)
├── Loss.py                  # Custom loss functions (e.g., asymmetric loss)
├── tinybert/                # Directory containing TinyBERT configs/checkpoints
```

>  **Note**: Additional core files like `Train.py`, `Multimodality.py`, `ViTTiny.py`, `TinyBert.py`, etc., are expected in the full repo.

---

## Model Overview

- **Vision Encoder**: ViT-Small (or ViT-Tiny)
- **Text Encoder**: TinyBERT
- **Fusion Strategy**: Concatenation of projected embeddings
- **Classifier**: MLP head
- **Loss**: BCEWithLogitsLoss or Asymmetric Loss

---

##  Dependencies

```bash
pip install torch torchvision transformers timm scikit-learn pandas matplotlib
```

---

##  Usage

### 1. Training

```bash
python Train.py
```

Configure:
- Dropout
- Hidden layer depth
- Optimizer (Adam / AdamW)
- Temperature

### 2. Inference

```bash
python Predict.py
```

Generates results in `hjia0784_yzha0544.csv`.

### 3. Visualization

```bash
python Plot.py
```

Uses `training_log.csv` to plot loss/F1 over epochs.

---

##  Folder: `tinybert/`

Contains:
- `config.json`
- `pytorch_model.bin`
- `vocab.txt`

Used to load and tokenize text inputs using HuggingFace `AutoModel` and `AutoTokenizer`.

---

##  Outputs

-  `final_multimodality.pth`: Final model (145.4MB)
-  `training_log.csv`: Training/Validation metrics
-  `hjia0784_yzha0544.csv`: Prediction results

---
