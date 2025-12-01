# DeepMIF

## DeepMIF: A Multi-view Interactive Fusion based Deep Learning Method for RNA-Small Molecule Binding Affinity Prediction

This repository contains the source code and datasets for the paper **"DeepMIF: A Multi-view Interactive Fusion based Deep Learning Method for RNA-Small Molecule Binding Affinity Prediction"**, submitted to the *Journal of Chemical Information and Modeling (JCIM)*.

### 1. Introduction
**DeepMIF** is a novel deep learning framework designed to accurately predict the binding affinity between RNA and small molecules. It leverages:
*   **Hybrid RNA Representation**: Combining RNA-FM embeddings with a Localized Enhanced Scalable k-mer (L-ESKmer) strategy.
*   **Multi-view Interactive Fusion**: A decoupled architecture that integrates sequence and graph views of both RNA and drugs via a context-aware cross-attention mechanism.

### 2. System Requirements
To reproduce the results, we recommend using a Linux or Windows environment with an NVIDIA GPU (CUDA supported).

#### Dependencies
environment.yaml

You can install the required packages using the following command:
```bash
pip install -r requirements.txt
