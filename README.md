# Gamma–hadron separation for SWGO D8 using Machine Learning Tecniques

This repository contains the code for a gamma–hadron separation study using *machine learning* techniques for the Southern Wide-field Gamma-ray Observatory (SWGO). The goal is to improve or, if possible, optimize the current results reported in the SWGO science prospects paper (https://doi.org/10.48550/arXiv.2506.01786) by employing relatively simple boosting models and physics-motivated input variables inspired by other experiments (e.g. HAWC and LHAASO).

## Overview

We explore supervised classification models to distinguish air showers produced by gamma rays from those induced by hadrons (protons) at ground level. The feature set is inspired by classical gamma–hadron separation variables used in water-Cherenkov and ground-based arrays such as HAWC and LHAASO (for example, timing variables (time quantiles – DOI 10.1088/1748-0221/20/04/T04008), Compactness – arXiv:1508.04047, PINCness – https://arxiv.org/pdf/2506.18277, R_max, and Muon Count).

The main focus of the project is:

- **To evaluate boosting-based algorithms** (for example, CatBoost, XGBoost, LightGBM) for the SWGO D8 detector configuration.  
- **To compare their performance** with existing gamma–hadron separation strategies reported in the literature.  
- **To provide a transparent and reproducible pipeline** that can later be extended with more complex models.

## Methods

- **Data acquisition:** filtering and preprocessing of simulated shower-level (CORSIKA) and detector-level (Geant4-AERIE) variables.  
- **Feature engineering:** construction of candidate gamma–hadron separation variables inspired by HAWC/LHAASO analyses and standard air-shower observables (followed by quality cuts, normalization, and train/validation/test splits).  
- **Models:** boosting-based classifiers as the main models (CatBoost, XGBoost, LightGBM).  
- **Evaluation:** ROC curves, AUC, background rejection at fixed signal efficiency (Q-factor), and stability tests across different energy ranges (robustness and overfitting analysis).

## Repository structure (planned)

- `data/` – link to the SWGO repository where the downloadable database for project collaborators is hosted.  
- `notebooks/` – exploratory analysis, feature studies and construction (gamma–hadron separators, essentially from HAWC), and validation plots.  
- `src/` – core Python code for training, testing, and validation.  
- `configs/` – configuration files for experiments (features, hyperparameters, splits – to be implemented).  
- `results/` – metrics, tables, and figures for the different runs.

## Reference

The main physics motivation and baseline expectations come from the SWGO science prospects paper:

> SWGO Collaboration, *Science Prospects for the Southern Wide-field Gamma-ray Observatory: SWGO*, arXiv:2506.01786.

Please cite this work if you use the ideas, code, or results from this repository.
```

