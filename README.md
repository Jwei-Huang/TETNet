# TETNet : A Target-Enhanced Triple-Branch Network with Spectral-Spatial Attention for Hyperspectral Image Classification

This repository provides the official implementation of **TETNet**, a target-enhanced triple-branch architecture for hyperspectral image classification. The framework integrates:

- TECEM-based target-enhanced preprocessing
- A 2D spatial branch
- A 3D spectral–spatial branch
- An attention modulation (AM) branch

The repository includes preprocessing, training, inference, and deployment profiling scripts required to reproduce the experimental results.

---

## 1. Environment Setup

We recommend using a virtual environment.

Create and activate environment:

    python -m venv myEnv
    myEnv\Scripts\activate        # Windows
    # source myEnv/bin/activate   # Linux/macOS

Install dependencies:

    pip install -r requirements.txt

---

## 2. Data Preparation

Place the following files under:

    dataset/

Required files:

    dataset/
    ├── Salinas_corrected.mat
    └── Salinas_gt.mat

<!-- The dataset is not redistributed in this repository. -->

---

## 3. TECEM Preprocessing

Generate class-wise TECEM maps:

    python -m scripts.precompute_tecem --data_dir dataset --out_dir outputs/SA_TECEM_cls

This step produces the target-enhanced class maps used by the 2D branch.

Output directory:

    outputs/SA_TECEM_cls/

---

## 4. Training TETNet

Train the model:

    python scripts/main.py --data_dir dataset --tecem_dir outputs/SA_TECEM_cls

Default configuration:
- Test ratio: 0.98
- Batch size: 64
- Epochs: 500

The best model weights will be saved to:

    runs/SA/seed0/best.hdf5

---

## 5. Evaluation

After training completes, classification metrics are automatically computed and saved to:

    runs/SA/seed0/
    ├── metrics.json
    └── classification_report.txt

Metrics include:
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Cohen’s Kappa

---


## Repository Structure

    src/
    ├── models/        # TETNet architecture
    ├── tecem/         # TECEM implementation
    ├── data/          # Data preprocessing utilities
    └── utils/         # Utility functions

    scripts/
    ├── precompute_tecem_sa.py
    ├── train_sa.py
    └── profile_deployment_sa.py

---

## Citation

If you use this implementation in your research, please cite the corresponding paper.




