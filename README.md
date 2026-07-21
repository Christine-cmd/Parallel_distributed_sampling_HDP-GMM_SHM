# Hierarchical Bayesian Nonparametric Structural Anomaly Detection

This repository contains the Python and Jupyter Notebook implementation associated with the manuscript:

> **Hierarchical Bayesian Nonparametric Modeling for Uncertainty-Aware and Bias-Mitigated Structural Anomaly Detection with a Parallel Inference Scheme**  
> Lin-Feng Mei¹ and Wang-Ji Yan¹˒²*  
>
> ¹ State Key Laboratory of Internet of Things for Smart City and Department of Civil and Environmental Engineering, University of Macau, China  
> ² Guangdong–Hong Kong–Macau Joint Laboratory for Smart Cities, China  

The manuscript has been submitted to *Mechanical Systems and Signal Processing* and is currently under review.

## Overview

This work proposes a data-driven structural anomaly-detection framework that integrates transmissibility functions (TFs) with a hierarchical Dirichlet process Gaussian mixture model (HDP-GMM).

The HDP-GMM models temporally grouped monitoring data using local DP-GMMs coupled through a global Dirichlet process. This hierarchical structure allows the model to:

- identify group-specific and recurring patterns of structural behavior;
- infer the number of mixture components without specifying it in advance;
- help reduce the influence of dominant components arising from the self-reinforcing property of conventional DP-GMMs;
- quantify uncertainty in clustering and component parameters.

A parallel distributed Gibbs sampler is developed to improve posterior-inference efficiency. Gibbs updates are performed across multiple processors and conditioned on aggregated component statistics maintained by a master node.

## Repository structure

```text
.
├── CRP_DPGMM/
│   └── Implementation of DP-GMM based on the Chinese restaurant process
│
├── Truncated_VI/
│   └── Truncated variational-inference implementation for the DP-GMM
│
├── Truncation_free_VI/
│   └── Truncation-free variational-inference implementation
│
├── CRF_hdp_gmm_5seeds.ipynb
│   └── Serial collapsed Gibbs sampling for the HDP-GMM
│
├── CRP-DPGMM_5seeds.ipynb
│   └── Collapsed Gibbs sampling for the conventional DP-GMM
│
├── distributed_hdp_gmm_ray_5seeds.ipynb
│   └── Parallel distributed Gibbs sampling for the HDP-GMM using Ray
│
├── TVI-DPGMM_5seeds.ipynb
│   └── Truncated variational-inference experiments
│
├── TFVI-DPGMM_revision_5seeds.ipynb
│   └── Truncation-free variational-inference experiments
│
├── Outlier_detectors_5seeds.ipynb
│   └── Comparative experiments using conventional outlier detectors
│
├── Uncertainty_calibration_scores.py
│   └── Computation of the Brier score and expected calibration error
│
├── hdpgmm.py
│   └── Core HDP-GMM functions
│
├── hdpgmm_syn.py
│   └── HDP-GMM implementation for the numerical case study
│
├── model_loglikelihood.py
│   └── Computation of model log marginal likelihood
│
└── [dataset file]
    └── Dataset used for the numerical case study
