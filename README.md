# Hierarchical Plant Disease Detection
### YOLOv8 + DINOv2 Three-Stage Pipeline

A deep learning system for real-world plant disease detection using a hierarchical coarse-to-fine architecture. Designed to address the performance gap between lab-controlled models and real field deployment.

---

## Results

| Metric | Value |
|--------|-------|
| Species Classification Accuracy | 95.8% |
| End-to-End Disease Accuracy | 71.49% |
| Standalone Disease Classifier | 75.63% |
| Cross-Species Errors | 0 (0.0%) |
| Disease Classes | 30 |
| Plant Species | 13 |
| Dataset | PlantDoc v2 (2,569 images) |

---

## Pipeline Architecture

**Stage 1 → YOLOv8s** — Detects and crops the leaf region from input image

**Stage 2 → DINOv2 Species Classifier** — Identifies plant species from full image (95.8% accuracy)

**Stage 3 → DINOv2 Disease Classifier** — Diagnoses disease from cropped leaf, constrained by species-aware masking

> **Key Innovation:** Species-aware masking forces predictions to only biologically valid disease classes for the identified species — eliminating cross-species errors entirely with zero additional parameters.

---

## Tech Stack

- **Detection:** YOLOv8s (Ultralytics)
- **Classification:** DINOv2 Vision Transformer (facebook/dinov2-base)
- **Framework:** PyTorch
- **Dataset:** PlantDoc v2
- **Training:** Google Colab (Tesla T4)

---## Repository Structure

- `yolo_code.ipynb` — Stage 1: YOLOv8 training
- `dino_code.ipynb` — Stage 2 & 3: DINOv2 training  
- `Final_Model_code.ipynb` — End-to-end pipeline
- `confusion_matrix.png` — Evaluation results
- `training_curves.png` — Training dynamics
- Research paper (PDF) included

---

## Key Design Decisions

- **Progressive backbone unfreezing** — DINOv2 backbone frozen for first 7-8 epochs, then partially unfrozen with 10-20x lower learning rate to prevent catastrophic forgetting
- **Species-aware masking** — Hard biological constraint applied at inference, reducing effective hypothesis space per species
- **Full image for species, crop for disease** — Exploits complementary spatial information at different scales

---

## Research Paper

Full methodology, results, and failure analysis available in the included IEEE-format paper.

## Authors

**Vandana Kushwaha** — Netaji Subhas University of Technology, Delhi
**Srishti** — Netaji Subhas University of Technology, Delhi





