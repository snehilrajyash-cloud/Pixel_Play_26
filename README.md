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

# 🧠 Code Breakdown & Methodology
1️⃣ Code 1: The Pseudo-3D CNN (Baseline)
File: Code_1_Pseudo3D_CNN.py

Architecture: We implemented a custom Simple3DCNN class utilizing nn.Conv3d layers. The network consists of three blocks, each containing a 3D Convolution, 3D Batch Normalization, and 3D Max Pooling.

Hypothesis: The goal was to capture spatiotemporal features (motion + appearance) simultaneously using 3D kernels.

Why it Failed (Score: 0.25): The model required a sequence of frames (depth > 1) to learn motion. However, due to data loading constraints, single frames were reshaped to (Batch, 3, 1, 224, 224). With a depth of 1, the 3D kernels could not extract temporal relationships, effectively acting as a computationally expensive 2D CNN with no motion context.

2️⃣ Code 2: Spatial Autoencoder (Reconstruction)
File: Code_2_Autoencoder.py

Architecture: A symmetric Encoder-Decoder network.

Encoder: Uses Conv2d layers to compress the 64x64 input image into a low-dimensional latent representation, stripping away noise.

Decoder: Uses ConvTranspose2d layers to reconstruct the original image from the latent vector.

Anomaly Detection Logic: The model is trained on normal footage to minimize Mean Squared Error (MSE). During inference:

Normal frames are reconstructed well (Low Error).

Anomalous frames (e.g., running, throwing objects) are reconstructed poorly (High Error).

Result (Score: 0.39): While this standard industry approach worked better than the baseline, it struggled with global lighting changes and camera noise, often flagging valid background motion as anomalies.

3️⃣ Code 3: Hybrid Ensemble & Temporal Smoothing (Final Solution) 🏆
File: Code_3_Hybrid_Ensemble.py

Architecture: A Weighted Ensemble combining two distinct scoring mechanisms:

Deep CNN (ImprovedCNN): A 5-block VGG-style network with BatchNorm and Dropout to extract high-level visual features.

Heuristic Feature Scorer: A custom function compute_feature_score that calculates:

Variance: High variance often indicates cluttered or chaotic movement.

Edge Density: Computed using gradients (np.diff) to detect sharp edges of foreign objects.

Color Deviation: Standard deviation of color channels.

Ensemble Logic: $$ \text{Final Score} = 0.6 \times \text{CNN Probability} + 0.4 \times \text{Heuristic Score} $$

Post-Processing (Temporal Smoothing): Real-world anomalies persist over time. We applied a Rolling Window Mean (window size = 5) to the final scores. This smoothed out single-frame noise spikes and reinforced consistent anomaly detections.

Result (Score: 0.58): This hybrid approach proved the most robust, leveraging the pattern recognition of deep learning with the stability of statistical features.

📊 Results Table
Model Iteration	Technique	Key Feature	Score
Iteration 1	Pseudo-3D CNN	Conv3d Layers	0.25
Iteration 2	Autoencoder	MSE Reconstruction Loss	0.39
Iteration 3	Hybrid Ensemble	Deep CNN + Edge/Var + Smoothing	0.58

Export to Sheets

🛠️ Installation & Usage
To run the final model locally, ensure you have the dataset downloaded and paths configured.

Prerequisites
Bash

pip install torch torchvision opencv-python pandas numpy pillow
Running the Inference
Clone the repository.

Open Code_3_Hybrid_Ensemble.py.

Update the DATA_DIR variable to point to your test dataset folder.

Run the script:

Bash

python Code_3_Hybrid_Ensemble.py
The script will generate a submission.csv file in the working directory.

📈 Key Learnings
Temporal Context is King: Analyzing frames in isolation is insufficient for video anomaly detection; smoothing across time steps significantly boosts performance.

Ensembling: Combining "Black Box" deep learning models with interpretable statistical features (like edge detection) creates a much more robust system than either approach alone.

Data Handling: Efficient data loading and handling of corrupt/missing frames is crucial for real-world datasets.

👤 Author
Snehil Rajyash

Institution: IIT Roorkee

GitHub: @snehilrajyash-cloud

Event: VLG Recruitment Challenge 2026
