# ML_GHSEP_SWGO – Gamma–hadron separation for SWGO using ML

This repository contains the code for a machine-learning study of gamma–hadron separation for the Southern Wide-field Gamma-ray Observatory (SWGO). The goal is to improve or optimize the current results reported in the SWGO prospects paper (https://doi.org/10.48550/arXiv.2506.01786) using relatively simple boosting models and physics-motivated input variables.

## Overview

We explore supervised classification models to distinguish gamma-induced from hadron-induced air showers at ground level. The feature set is inspired by gamma–hadron separation variables used in water-Cherenkov and ground-based experiments such as HAWC and LHAASO (e.g. timing, lateral distribution, charge-related and topology variables).

The focus of the project is:

- To benchmark boosting-based algorithms (e.g. gradient boosting, XGBoost, LightGBM) for SWGO-like detector configurations.
- To compare their performance with existing gamma–hadron separation strategies reported in the literature.
- To provide a transparent and reproducible pipeline that can be extended with more complex models.

## Methods

- **Data preparation:** loading and preprocessing of simulated shower and detector-level variables (quality cuts, normalisation, train/validation/test splits).
- **Feature engineering:** construction of gamma–hadron separator candidates inspired by HAWC/LHAASO analyses and by standard air-shower observables.
- **Models:** baseline boosting classifiers plus simple baselines (e.g. logistic regression, random forest) for comparison.
- **Evaluation:** ROC curves, AUC, background rejection at fixed signal efficiency, and stability checks across energy and zenith-angle bins.

## Repository structure (planned)

- `data/` – links or scripts to obtain and preprocess the input datasets.
- `notebooks/` – exploratory analysis, feature studies and validation plots.
- `src/` – core Python code for training, evaluation and utilities.
- `configs/` – configuration files for experiments (features, hyperparameters, splits).
- `results/` – metrics, tables and figures for the different runs.

## Reference

Main physics motivation and baseline expectations are taken from the SWGO science prospects paper:

> SWGO Collaboration, *Science Prospects for the Southern Wide-field Gamma-ray Observatory: SWGO*, arXiv:2506.01786.

Please cite this work if you use the ideas, code or results from this repository.
