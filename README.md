# EMK-KEN: A High-Performance Approach for Assessing Knowledge Value in Citation Network

<img width="800" alt="emkken_plot" src="./assets/model.jpg">

## About

This is the github repo for the paper ["EMK-KEN: A High-Performance Approach for Assessing Knowledge Value in Citation Network"]().
By integrating Mamba and KAN architectures, the model addresses the inefficiency and poor generalization of traditional methods in large-scale networks. The framework consists of two key modules:

- Semantic Feature Extraction: MetaFP and Mamba process metadata and long-sequence text embeddings to learn contextual representations.
- Structural Information Capture: KAN leverages structural differences across domains to model complex dependencies in citation networks.

Extensive experiments on 10 cross-domain benchmark datasets demonstrate that EMK-KEN outperforms state-of-the-art models (e.g., GNNs, Transformers) in accuracy, F1-score, and AUC, with faster inference and stronger robustness. Ablation studies confirm the necessity of components (e.g., causal convolution, SSMs), while hyperparameter analysis highlights tunable sensitivity. This work provides a powerful tool for literature analysis and knowledge mining, with potential extensions to broader knowledge evaluation tasks.

## Installation
EMKKEN can be installed directly from GitHub.

**Pre-requisites:**

```
Python 3.8.18 or higher
```

**For developers**

```
git clone https://github.com/naipings/EMKKEN.git
cd EMKKEN
pip install -e .
```

**Installation via github**

```
pip install git+https://github.com/naipings/EMKKEN.git
```

Requirements

```python
# python==3.8.18
imbalanced_learn==0.12.3
imblearn==0.0
langdetect==1.0.9
mamba_ssm==2.2.2
matplotlib==3.7.5
networkx==3.1
nltk==3.8.1
numpy==1.24.3
optuna==3.6.1
pandas==2.0.3
scikit_learn==1.3.2
seaborn==0.13.2
torch==2.2.0
torch_geometric==2.6.1
transformers==4.43.3
```

After activating the virtual environment, you can install specific package requirements as follows:
```python
pip install -r requirements.txt
```

Other requirements:
- Linux
- NVIDIA GPU
- PyTorch 1.12+
- CUDA 11.6+

**Optional: Conda Environment Setup**
For those who prefer using Conda:
```
conda create --name emkken-env python=3.8.18
conda activate emkken-env
pip install git+https://github.com/naipings/EMKKEN.git  # For GitHub installation
```

