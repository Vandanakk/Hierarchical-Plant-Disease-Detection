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
Input Image
│
▼
[Stage 1] YOLOv8s ──────────── Leaf localization + crop
│
▼
[Stage 2] DINOv2-Species ───── Identifies plant species (95.8% acc)
│
▼
[Stage 3] DINOv2-Disease ───── Diagnoses disease with species-aware masking
│
▼
Output: Species + Disease Label

**Key Innovation — Species-Aware Masking:** Constrains disease prediction to only biologically valid classes for the identified species, eliminating cross-species errors entirely with zero additional parameters.

---

## Tech Stack

- **Detection:** YOLOv8s (Ultralytics)
- **Classification:** DINOv2 Vision Transformer (facebook/dinov2-base)
- **Framework:** PyTorch
- **Dataset:** PlantDoc v2
- **Training:** Google Colab (Tesla T4)

---

## Repository Structure
├── yolo_code.ipynb          # Stage 1: YOLOv8 leaf detection training
├── dino_code.ipynb          # Stage 2 & 3: DINOv2 classifier training
├── Final_Model_code.ipynb   # End-to-end pipeline evaluation
├── confusion_matrix.png     # Model evaluation results
├── training_curves.png      # Training dynamics
├── spot_check_test.png      # Sample pipeline outputs
└── IEEE_Conference_Template__1_ (2).pdf  # Full research paper

---

## Key Design Decisions

- **Progressive backbone unfreezing** — DINOv2 backbone frozen for first 7-8 epochs, then partially unfrozen with 10-20x lower learning rate to prevent catastrophic forgetting
- **Species-aware masking** — Hard biological constraint applied at inference, reducing effective hypothesis space per species
- **Full image for species, crop for disease** — Exploits complementary spatial information at different scales

---

## Research Paper

Full methodology, results, and failure analysis available in the included IEEE-format paper.
