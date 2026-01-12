# qPACT Assessment via Virtual Imaging Studies

<p align="center">
  <b>Code release accompanying the paper</b><br>
  <i>Application of a Virtual Imaging Framework for Investigating a Deep Learning–Based Reconstruction Method<br>
  for 3D Quantitative Photoacoustic Computed Tomography (qPACT)</i>
</p>

---

## 📌 Overview

This repository contains the source code used to conduct **virtual imaging studies (VIS)** for the evaluation of a **deep learning–based 3D quantitative photoacoustic computed tomography (qPACT)** reconstruction method.

The code supports the **full experimental pipeline**, including:

- preparation of network inputs via **acoustic time-reversal reconstruction**,
- deep learning–based **joint segmentation and quantitative estimation**,
- and **task-specific assessment** of segmentation accuracy and quantitative oxygen saturation (sO₂) metrics.

The implementation was developed to support the simulation-based studies reported in the accompanying paper and is intended to facilitate **reproducibility, transparency, and reuse** for related qPACT research.

---

## 🧠 Key Contributions Enabled by This Repository

- A **virtual imaging study** for evaluating qPACT reconstruction under controlled yet realistic conditions  
- End-to-end integration of:
  - numerical acoustic reconstruction,
  - learning-based inference,
  - and quantitative post-hoc assessment
- Study-specific evaluation protocols for:
  - vessel and tumor segmentation,
  - vessel-wise sO₂ estimation accuracy,
  - confusion-matrix–based performance metrics

---

## 📁 Repository Structure

```
qPACT_assessment_via_VI_studies/
│
├── dataloader/
│   └── PyTorch dataset definitions and data-loading utilities
│
├── model/
│   └── Deep learning architectures used for joint segmentation and regression
│
├── scripts/
│   └── Training and inference scripts for the learning-based reconstruction model
│
├── acoustic_reconstruction_time_reversal/
│   ├── time_reversal_recon2.m
│   └── README.md
│   └── MATLAB code to generate time-reversal reconstructions
│       used as network inputs (Study 2)
│
├── assessment/
│   ├── separate_vessels_and_tumors.m
│   ├── compute_confusion_matrix.m
│   ├── compute_average_vessel_so2.m
│   └── README.md
│   └── MATLAB scripts for quantitative and segmentation-based evaluation
│
└── README.md
```

---

## 🗂 Data Availability

The numerical breast phantom datasets used with this code are **publicly available** via the Illinois Data Bank:

- **Title:** *Anatomy-Matched Multi-Variant Stochastic Optoacoustic Numerical Breast Phantoms and Simulated OAT Measurement Data*  
- **DOI:** https://doi.org/10.13012/B2IDB-8164905_V1

This repository **does not redistribute the dataset**. Users should download the data directly from the official repository and configure local paths as needed for the provided scripts.


---

## 🔬 Acoustic Reconstruction (Time Reversal)

The folder [`acoustic_reconstruction_time_reversal/`](./acoustic_reconstruction_time_reversal) contains MATLAB code used to generate **3D time-reversal reconstructions** using the *k-Wave* toolbox.

These reconstructions serve as **inputs to the deep learning model** in **Study 2** of the paper.

**Main features:**
- Multi-wavelength time-reversal reconstruction
- Optional noise injection and frequency-domain filtering
- Consistent reconstruction protocol across all virtual subjects

See the folder-level README for detailed usage instructions and assumptions.

---

## 📊 Assessment and Evaluation

The [`assessment/`](./assessment) folder contains MATLAB scripts used to compute evaluation metrics reported in the paper.

### Included assessments:
- **Vessel / tumor separation**
- **Confusion-matrix–based segmentation metrics**
- **Average vessel sO₂ estimation**

Each script includes a clearly marked **USER CONFIG** section to specify:
- input/output paths,
- label conventions,
- and simple evaluation options,

while preserving the original numerical logic used in the study.

---

## 🚀 Typical Workflow

A typical evaluation pipeline proceeds as follows:

1. **Generate acoustic reconstructions**  
   → `acoustic_reconstruction_time_reversal/time_reversal_recon2.m`

2. **Train or run inference with the deep learning model**  
   → `scripts/`

3. **Post-hoc quantitative assessment**  
   → `assessment/`

This modular structure allows individual components (e.g., reconstruction or assessment) to be reused independently.

---

## ⚙️ Requirements

Depending on which parts of the pipeline are used:

### MATLAB
- MATLAB (R2020a or newer recommended)
- k-Wave Toolbox (https://www.k-wave.org/)

### Python
- Python ≥ 3.8
- PyTorch
- NumPy, SciPy, h5py
- Additional dependencies as noted in training scripts

---

## 📄 Citation

If you use this code, please cite the accompanying paper:

```
@article{CAM2025100792,
title = {Application of a virtual imaging framework for investigating a deep learning-based reconstruction method for 3D quantitative photoacoustic computed tomography},
journal = {Photoacoustics},
pages = {100792},
year = {2025},
issn = {2213-5979},
doi = {https://doi.org/10.1016/j.pacs.2025.100792},
url = {https://www.sciencedirect.com/science/article/pii/S2213597925001156},
author = {Refik Mert Cam and Seonyeong Park and Umberto Villa and Mark A. Anastasio},
keywords = {Quantitative photoacoustic computed tomography, Numerical breast phantoms, Breast imaging, Virtual imaging studies}
}
```

---

## 📜 License

The license for this repository will be specified upon publication of the paper.

---
