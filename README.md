# **NPstereo**

<img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square"/> <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>

**Assigning the Stereochemistry of Natural Products by Machine Learning**

NPstereo is a transformer-based language model that assigns stereochemistry (tetrahedral R/S, double bond E/Z) to natural products (NPs) from their SMILES representations. It was trained on referenced data from the open-access [COCONUT database](https://coconut.naturalproducts.net) and can complete missing stereocenters in partially assigned or fully unassigned NP structures.

This repository contains:
- Data extraction and augmentation scripts
- Model training and evaluation code
- Utilities for running predictions on new molecules

## Overview
Nature shows strong stereochemical regularities — L-amino acids in proteins, D-sugars in nucleotides. The idea of NPstereo is to investigate whether similar patterns exist in other NPs and whether they can be machine-learned.

**Key features:**
- Predicts stereochemistry directly from SMILES
- Supports **full assignment** and **completion of partially assigned stereochemistry**
- Achieves **80.2% per-stereocenter accuracy** for full assignment and **85.9%** for partial assignment on canonical SMILES
- Works across diverse NP classes (alkaloids, polyketides, lipids, terpenes, peptides, glycosides, etc.)

For details, see the manuscript!

## Installation

Clone the repository:
```bash
git clone https://github.com/reymond-group/NPstereo.git
cd NPstereo
```

Create and activate a conda environment:
```bash
conda create -n npstereo python=3.10
conda activate npstereo
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data
Training and evaluation data are derived from the **COCONUT** database, filtered for entries with at least one DOI reference.  
We include:
- Scripts to extract and process the dataset (`data/`)
- Data augmentation methods:
  - **SMILES randomization**
  - **Partial stereochemistry removal**
- Download links to processed datasets on [Zenodo](https://zenodo.org/records/13790363). This data is too heavy to include on GitHub, however following the code you should be able to reproduce the dataset from scratch.

## Model Training
Models are trained using [OpenNMT-py](https://opennmt.net/OpenNMT-py/):

Example training command:
```bash
onmt_build_vocab -config configs/npstereo.yaml -n_sample -1
onmt_train -config configs/npstereo.yaml
```

Key architectures:
- **C1** – baseline (canonical SMILES only)
- **A2–A50** – canonical + randomized SMILES
- **NPstereo** – partially assigned → canonical
- **M65** – combined augmentation (randomization + partial assignment)

## Running Predictions

To predict stereochemistry for new molecules:
```bash
python predict.py   --model seed-0/npstereo/npstereo_step_100000.pt   --src input_smiles.txt   --output predictions.txt
```

**Input:** SMILES without full stereochemistry (absolute or partially assigned).  
**Output:** Predicted isomeric SMILES with stereochemistry assigned.

For best performance:
- Use **canonical SMILES** as input for NPstereo
- For non-canonical SMILES, use models trained with randomization (A50 or M65)

## Evaluation

We provide evaluation scripts in `evaluation/` to reproduce the metrics reported in the manuscript:

- **SMILES validity**
- **Full-assignment accuracy**
- **Per-stereocenter accuracy**
- Class-wise breakdown by NP structural category

## License
[MIT](LICENSE)

## Contact

<img src="https://img.shields.io/twitter/follow/reymondgroup?style=social"/> 
<img src="https://img.shields.io/twitter/follow/markusorsi?style=social"/>
