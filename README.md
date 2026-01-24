# Wave-to-Map: Knowledge Distillation for Wave Propagation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)

A deep learning research project investigating the efficiency of **Convolutional Neural Networks (CNN)** versus **Fourier Neural Operators (FNO)** in reconstructing 2D wave heatmaps from sparse 1D sensor data, utilizing **Knowledge Distillation**.

---

## 📖 Overview

The goal of this project is to predict a complete **2D spatial heatmap** of wave propagation using only **1D time-series data** collected from sensors at the boundaries.

We compare two student architectures:
1.  **CNN Student:** A standard Convolutional Network with `ConvTranspose2d` decoding.
2.  **FNO (Learned Basis) Student:** An MLP-based architecture that learns an optimal basis transformation (evolving beyond standard FFT).

Key to this experiment is the use of a **Teacher Model** (210M parameters) to distill knowledge into the smaller student models.

---

## 🏗️ Architecture

### 1. The Teacher
* **Size:** ~210 Million Parameters.
* **Role:** Acts as an "Oracle," providing smoothed supervision signals to stabilize student training.

### 2. The Students
| Model | Type | Architecture Details | Params |
| :--- | :--- | :--- | :--- |
| **CNN** | Convolutional | `Conv1d` (Time) -> `ConvTranspose2d` (Space) | ~3.1M |
| **FNO** | Learned Basis | Learnable Basis Transform -> Dense MLP -> Grid | ~2.6M |

---

## 🧪 Experiments & Results

We trained both models to map `(Batch, Time_Steps, Sensors)` $\to$ `(Batch, 32, 32)`.

### Key Observation: The Distillation Gap
In our specific test setup, the **CNN outperformed the FNO**. 

Upon deep analysis, this result highlights the impact of the training strategy rather than pure architectural superiority:
* **CNN Training:** Utilized **Knowledge Distillation** (Loss = Ground Truth + Teacher Guidance).
* **FNO Training:** Utilized **Standard Supervision** (Loss = Ground Truth only).

While the FNO (especially with a learned basis) typically captures global wave physics better, the CNN benefitted significantly from the Teacher's "smoothing" effect, allowing it to converge faster and reach a lower error rate (~2.1m vs ~3.1m) in this specific notebook configuration.

---

## 🚀 Usage

### Prerequisites
* Python 3.8+
* PyTorch
* NumPy, Matplotlib

### Running the Notebook
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/wave-distillation.git](https://github.com/yourusername/wave-distillation.git)
    cd wave-distillation
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open the analysis notebook:
    ```bash
    jupyter notebook Analysis_Notebook.ipynb
    ```

---

## 📊 Visualizations

The notebook includes comparisons of:
* **Ground Truth:** The actual simulation physics.
* **Teacher Prediction:** The high-fidelity oracle output.
* **Student Predictions:** Comparative heatmaps of the CNN vs. FNO.

---

## 📝 Citation
If you use this code for your research, please cite:
```text
[Your Name/Team Name]
Wave-to-Map Distillation Project
2024
