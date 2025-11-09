<h1 align="center">
Verbalized Probabilistic Graphical Modeling
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2406.05516"><img src="https://img.shields.io/badge/arXiv-2406.05516-b31b1b.svg" alt="arXiv"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10.11-blue.svg" alt="Python Version"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.7.0-red.svg" alt="PyTorch Version"></a>
  <a href="https://docs.vllm.ai/en/v0.9.1/"><img src="https://img.shields.io/badge/vLLM-0.9.1-54a0f8.svg" alt="vLLM Version"></a>
</p>

This repository contains the official implementation of the paper:
> __Verbalized Probabilistic Graphical Modeling__  
> [Hengguan Huang*](https://scholar.google.com/citations?hl=en&user=GQm1eZEAAAAJ), [Xing Shen*](https://scholar.google.com/citations?hl=en&user=U69NqfQAAAAJ), Guang-Yuan Hao, Songtao Wang, Lingfa Meng, Dianbo Liu, David Alejandro Duchene, Hao Wang, Samir Bhatt  
> _*Equal contribution_  
> _AAAI Conference on Artificial Intelligence, 2026_  
> __Paper ([arXiv preprint](https://arxiv.org/abs/2406.05516))__


## Overview

## 1. Preparation

### 1.1 Installation
It is recommended to use a virtual environment (e.g., `venv`) to avoid package conflicts. Here we assume you are using `venv` as your virtual environment. If you are using conda, please adjust the commands accordingly.
```bash
git clone https://github.com/xingbpshen/agentic-reasoning-vpgm.git
cd agentic-reasoning-vpgm/
pip install -r requirements.txt
```

### 1.2 Preparing the Dataset
The downloaded dataset should be structured in the following format, the `datasets/` directory should be placed at the root of the repository:
```
datasets/
└── my_scienceqa/
    ├── val_1005.json
    └── test_2563.json
```
This dataset is a subset, and a processed (all tools' responses are included) version of the original [ScienceQA](https://github.com/lupantech/ScienceQA) dataset. For convenience, we provide a Google Drive Link (coming soon) to download the processed dataset used in our experiments, note that the dataset is under CC BY-NC-SA 4.0 license.

## 2. Running Inference
Please run the following command to run inference:
```bash
bash auto_run.sh
```
The inference results file will be saved in the specified path `results/` under the project root.

## Acknowledgments

## Citation

## Contact
Please raise a GitHub issue or email us at <a href="mailto:xing.shen@mail.mcgill.ca">xing.shen@mail.mcgill.ca</a> (with the email subject starting with "[vPGM]") if you have any question or encounter any issue.