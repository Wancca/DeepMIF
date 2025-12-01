# DeepMIF

## DeepMIF: A Multi-view Interactive Fusion based Deep Learning Method for RNA-Small Molecule Binding Affinity Prediction

### 2. System Requirements
To reproduce the results, we recommend using a Linux or Windows environment with an NVIDIA GPU (CUDA supported).

#### Dependencies
environment.yaml


3. Data Preparation
The dataset used in this study (R-SIM) and the processed features should be placed in the data/ directory.
Raw Data: The original R-SIM dataset can be found at https://web.iitm.ac.in/bioinfo2/R_SIM/.
Processed Data: We have provided the processed sequence and graph features in this repository under data/.

To train the DeepMIF model and evaluate it using 5-fold cross-validation for the binding affinity regression task, please run the following command:
python finetune_rna_dta_Regression_cv.py
