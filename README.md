# Inverse Wave Source Localization from Sparse Sensors: A Comparative Study of FNO and CNN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9%2B-ee4c2c.svg)](https://pytorch.org/)

A deep learning research project investigating the efficiency of **Convolutional Neural Networks (CNN)** versus **Fourier Neural Operators (FNO)** in reconstructing 2D wave heatmaps from sparse 1D sensor data, utilizing **Knowledge Distillation**.

Yotam Cohen - yotam.cohen@campus.technion.ac.il  
Idan Nissany - idan.nissany@campus.technion.ac.il

---

## Overview

The precise localization of wave sources from sparse sensor measurements is a fundamental inverse problem. While the "forward" problem of simulating wave propagation is well-posed and governed by known partial differential equations (PDEs), the "inverse" problem - determining the source location from a few scattered sensors- is often ill-posed and computationally expensive. Traditional numerical methods rely on iterative optimization techniques that suffer from high latency. We aim to replace these expensive solvers with a data-driven Deep Learning approach capable of mapping sparse time-series data directly to a spatial source distribution with significantly reduced inference latency.

In this study, we present a rigorous comparative analysis of two distinct architectures:
1.  **CNN Student:** A standard Convolutional Network with `ConvTranspose2d` decoding, trained using knowledge distillation.
2.  **FNO (Learned Basis) Student:** An MLP-based architecture that learns an optimal basis transformation (evolving beyond standard FFT).

We generated a synthetic dataset, trained each architecture independently, and conducted a series of comparative experiments to evaluate their final accuracy.

### Final Architectures
A visual comparison of the **CNN** (left) and **FNO** (right) architectures:
<p align="center">
  <img src="Assests/cnn_architecture.png" width="45%" />
  <img src="Assests/fno_architecture.png" width="45%" /> 
</p>

### Experimental Evaluation
To rigorously compare the architectures, we conducted four distinct stress tests focusing on real-world reliability and computational efficiency.

* **Convergence Speed:** We tracked the validation loss over training epochs to determine which model learns the wave physics faster.
* **Data Efficiency:** We retrained both models on increasingly smaller subsets of data (from 100% down to 10%) to test generalization capabilities in data-scarce environments.
* **Sensor Resilience:** We simulated hardware failure by "killing" one sensor (in probability p=0.5, outputting zeros or gaussian noise with std=0.1) during inference to see if the models could reconstruct the source using only partial data.
* **Noise Robustness:** We injected varying levels of Gaussian noise ($\sigma \in [0, 0.1]$) to evaluate performance stability in high-interference environments.

The quantitative results of these experiments are visualized below:

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="Assests/accuracy(epochs).jpg" width="100%" />
      <br />
      <b>(a) Convergence Speed</b>
    </td>
    <td align="center" width="50%">
      <img src="Assests/accuracy(sample_num).jpg" width="100%" />
      <br />
      <b>(b) Data Efficiency</b>
    </td>
  </tr>
</table>

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="Assests/sensor_mal.jpg" width="100%" />
      <br />
      <b>(c) Sensor Resilience</b>
    </td>
    <td align="center" width="50%">
      <img src="Assests/accuracy(noise).jpg" width="100%" />
      <br />
      <b>(d) Noise Robustness</b>
    </td>
  </tr>
</table>

### Key Findings
* **Superior Generalization:** The **FNO** converges significantly faster and maintains high accuracy even with 80% less training data compared to the CNN (see plots *a* and *b*).
* **Safety Critical Resilience:** In the event of a sensor failure, the **FNO** successfully reconstructs the wave source using global physical dependencies, whereas the CNN's error spikes (see plot *c*).
* **The Trade-off:** While the FNO dominates in clean and low-noise environments, the **CNN** exhibits slightly better stability when the signal-to-noise ratio is extremely low (high noise), as shown in plot *d*.

---
### Project Structure
```text
.
├── assets/                              # Images and plots used in README
│   ├── cnn_architecture.png
│   ├── fno_architecture.png
│   └── [experimental_plots]             # (Various result graphs)
├── Experimentation/
│   ├── CNN_experimentations.ipynb       # Experiments refining the CNN model
│   ├── FNO_experimentations.ipynb       # Experiments refining the FNO model
│   └── utilities.py                     # Helper functions for experimental runs
├── Models/
│   ├── CNN.py                           # CNN model class definition
│   └── FNO.py                           # FNO model class definition
├── Utilities/
│   ├── data_utilities.py                # Functions for data generation & augmentation
│   └── testing.py                       # Evaluation functions for final comparisons
├── CNN_vs_FNO_comparisons.ipynb         # Main comparison results (The "Final Report")
├── CNN_vs_FNO_controlled.ipynb          # Comparisons with controlled model sizes
├── environment.yml                      # Conda environment configuration
├── requirements.txt                     # Pip package dependencies
└── README.md                            # Project documentation
```

### 🏗️ Architecture

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
