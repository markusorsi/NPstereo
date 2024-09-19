# **NPstereo**

This repository contains the code and resources for predicting the absolute stereochemistry of natural products (NPs) using a transformer-based model. The models are trained to accurately predict stereocenters from the absolute SMILES representation of a chemical compound.

<img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square"/> <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>

## Theory

The objective of our model is to predict the absolute configuration of natural products based on their absolute SMILES representations. We employ a transformer model implemented using OpenNMT to translate absolute SMILES into isomeric SMILES. Our dataset is derived from the latest version (09-2024) of the COCONUT database, which is the largest repository of natural products. The most effective model achieves an accuracy exceeding 85% and a per-stereocenter accuracy above 95%. The repository includes comprehensive code for data extraction, preprocessing, model training, and evaluation. However, the repository will be subject to updates and improvements in the near future.

## Getting started

To replicate the results of our model, follow the instructions below.

#### 1. Clone the repository
```bash
git clone https://github.com/reymond-group/NPstereo.git
```

#### 2. Download the data/ directory from the Zenodo repository.

[Zenodo NPstereo repository](https://zenodo.org/records/13790363)

#### 3. Install the required conda environments using the following command:
```bash
conda env create -f npstereo.yml
conda env create -f npstereotmap.yml
```
Since the TMAP package does not support Python versions >3.7 we create a separate environment for notebooks that generate the TMAP plots.

#### 4. Run the code in the notebooks to reproduce the results.

## Notebooks

The notebooks are organized as follows:

1. **01_extract_coconut**: Contains the SQL query to extract the dataset from the PostgreSQL dump and the preprocessing steps to clean up the dataset. 
2. **02_augment_data**: Contains the code to augment the dataset via SMILES randomization.
3. **03_prepare_dataset**: Contains the code to prepare the dataset in the format required by OpenNMT for training the model.
4. **04_train**: Contains the code to train the transformer model using OpenNMT.
5. **05_predict**: Contains the code to run the predictions on the test set.
6. **06_evaluate**: Contains the code to evaluate the model's performance.
7. **07_analysis**: Contains the code to generate the TMAP plots and the in-depth analysis of the model's performance.
8. **08_validation**: Contains the code to run the predictions on a small set of manually curated compounds to validate the model's performance.

## License
[MIT](LICENSE)

## Contact

<img src="https://img.shields.io/twitter/follow/reymondgroup?style=social"/> 
<img src="https://img.shields.io/twitter/follow/markusorsi?style=social"/>