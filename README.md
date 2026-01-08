# Pixel_Play_26
This repository contains all the code that i have tried for the pixel play 2026 hackathon along with the readme.md file and the outcome scores of the code
# Pixel Play 26: Video Anomaly Detection 🎥🚨

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![VLG](https://img.shields.io/badge/VLG-Recruitment_Challenge-red)

## 📜 Overview

This repository contains the source code and documentation for my submission to the **VLG Recruitment Challenge '26 (Pixel Play)**. The objective of this project was to develop a computer vision model capable of detecting anomalous events in surveillance footage (Avenue Dataset).

Our final solution utilizes a **Hybrid Ensemble Approach**, combining a Deep CNN classifier with statistical feature extraction (Variance/Edge Density) and temporal smoothing, achieving a final anomaly detection score of **~0.58**.

---

## 📂 Repository Structure

```text
Pixel_Play_26/
├── Code_1_Pseudo3D_CNN.py       # Iteration 1: Attempt at 3D Convolutions (Score: 0.25)
├── Code_2_Autoencoder.py        # Iteration 2: Unsupervised Reconstruction (Score: 0.39)
├── Code_3_Hybrid_Ensemble.py    # FINAL MODEL: CNN + Statistical Features (Score: 0.58)
├── submission.csv               # Final generated submission file
└── README.md                    # Project documentation
