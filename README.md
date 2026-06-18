# Practical Homework

A curated collection of solved practical assignments from **Machine Learning**, **Deep Learning**,
**Reinforcement Learning**, **Generative AI**, and **Matrix Analysis** courses at
**HSE** and the **Yandex School of Data Analysis (YSDA)**.

Each course lives in its own top-level directory named `Topic-Org`, and every assignment is a
self-contained folder following a uniform `hwN-topic` / `labN-topic` convention.

<p>
  <img alt="Python"   src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white">
  <img alt="PyTorch"  src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="NumPy"    src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="Jupyter"  src="https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white">
</p>

## Contents

- [Machine Learning (HSE)](#machine-learning-hse)
- [Deep Learning (HSE)](#deep-learning-hse)
- [Deep Learning in Natural Sciences (HSE)](#deep-learning-in-natural-sciences-hse)
- [Generative AI (YSDA)](#generative-ai-ysda)
- [Reinforcement Learning (YSDA)](#reinforcement-learning-ysda)
- [Matrix Analysis (HSE & YSDA)](#matrix-analysis-hse--ysda)

---

## Machine Learning (HSE)

| # | Assignment | Description |
|---|------------|-------------|
| 1 | [Linear Regression](ML-HSE/hw1-linear-regression) | Data analysis, feature engineering, visualization, and model fitting with `scikit-learn`. |
| 2 | [Gradient Descent](ML-HSE/hw2-gradient-descent) | From-scratch gradient descent variants (`descents.py`) and linear regression training (`linear_regression.py`). |
| 3 | [Linear Classification](ML-HSE/hw3-linear-classification) | SVM, logistic regression, probability calibration, feature transformation, and multi-class classification on a near real-world business case. |
| 4 | [Decision Trees](ML-HSE/hw4-decision-trees) | Classification trees, hyperparameter analysis, a custom tree implementation, and regression trees with linear models in the leaves. |
| 5 | [Boosting](ML-HSE/hw5-boosting) | Custom gradient boosting, optimization techniques, and experiments with CatBoost. |

## Deep Learning (HSE)

| # | Assignment | Description |
|---|------------|-------------|
| 1 | [Feed-Forward Neural Networks](DL-HSE/hw1-feedforward-nn) | Fully connected network with forward/backward propagation, hyperparameter tuning, and performance visualization. |
| 2 | [Convolutional Neural Networks](DL-HSE/hw2-cnn) | CNN for image classification with convolutional and pooling layers. |
| 3 | [Recurrent Neural Networks](DL-HSE/hw3-rnn) | LSTM-based sequence model with a custom dataset loader, model definitions, and training pipeline. |
| 4 | [CLIP](DL-HSE/hw4-clip) | Contrastive Language–Image Pretraining: text and image encoders with a projection head, trained on paired image–text data. |

## Deep Learning in Natural Sciences (HSE)

| # | Assignment | Description |
|---|------------|-------------|
| 1 | [Bioinformatics](DL-NS-HSE/hw1-bioinformatics) | Deep neural networks for genomic sequence analysis and functional annotation prediction. |
| 2 | [Materials Science I](DL-NS-HSE/hw2-materials-science-i) | Convolutional and graph-based networks for forecasting material properties from atomic structures. |
| 3 | [Materials Science II](DL-NS-HSE/hw3-materials-science-ii) | ML workflows for materials discovery with uncertainty quantification and robustness analysis. |
| 4 | [Physics-Informed Neural Networks](DL-NS-HSE/hw4-pinns) | Networks with physical laws embedded in the loss to solve partial differential equations. |

## Generative AI (YSDA)

| # | Assignment | Description |
|---|------------|-------------|
| 1 | [Flow Matching](GenAI-YSDA/hw1-flow-matching) | Flow Matching model training with JIT compilation and a REPA-based architecture. |
| 2 | [Flow Map Models](GenAI-YSDA/hw2-flow-map-models) | Exploration of flow map models for generative tasks. |
| 3 | [MMD Distillation](GenAI-YSDA/hw3-mmd-distillation) | Few-step generator distillation with an added Maximum Mean Discrepancy (MMD) objective. |
| 4 | [MAR with Flow Matching Head](GenAI-YSDA/hw4-mar-flow-matching-head) | Masked Autoregressive image generation with a per-token flow matching head, built on a VAE latent space. |

## Reinforcement Learning (YSDA)

| # | Assignment | Description |
|---|------------|-------------|
| 1 | [Cross-Entropy Method](RL-YSDA/hw1-crossentropy-method) | Deep cross-entropy method with neural network function approximation for control tasks. |
| 2 | [Dynamic Programming](RL-YSDA/hw2-dynamic-programming) | Value iteration and policy iteration for solving Markov Decision Processes. |
| 3 | [Model-Free RL](RL-YSDA/hw3-model-free-rl) | Monte Carlo, Temporal Difference, and on-/off-policy methods with sample-efficiency experiments. |
| 4 | [Deep Q-Networks (DQN)](RL-YSDA/hw4-dqn) | DQN with experience replay and target networks for discrete action spaces. |
| 5 | [Continuous Control (TD3 & SAC)](RL-YSDA/hw5-continuous-control) | Twin Delayed DDPG and Soft Actor-Critic for continuous action spaces. |

## Matrix Analysis (HSE & YSDA)

| # | Lab | Description |
|---|-----|-------------|
| 1 | [Image Search via SVD](MatrixAnalysis-HSE-YSDA/lab1-image-search-svd) | Singular value decomposition for eigenfaces and similar-image retrieval. |
| 2 | [Recommendations via ALS](MatrixAnalysis-HSE-YSDA/lab2-als-recommendation) | A recommender system based on low-rank approximation of sparse matrices. |
| 3 | [Tomography](MatrixAnalysis-HSE-YSDA/lab3-tomography) | Tomogram reconstruction from ray-intensity data. |
| 4 | [Resistor Network Voltages](MatrixAnalysis-HSE-YSDA/lab4-resistor-voltages) | Iterative methods for solving large sparse linear systems in a resistor network. |

---

## Notes

- Large datasets and downloadable archives (e.g. `ml-10m.zip`, `lfw-a.tgz`) and training
  artifacts (`catboost_info/`, `*.db`) are **not** tracked in git — see [`.gitignore`](.gitignore).
  Download the corresponding datasets to reproduce a notebook end-to-end.
</content>
</invoke>
