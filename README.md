# practical-hw

A repository of practical tasks from ML, DL, RL, and Matrix Analysis courses. All courses completed at HSE or YSDA. There may be some issues in the code.


## Machine Learning course-hse

### Overview of Project

#### **Linreg**

* Analysis of data and feature engineering.
* Data visualization and working with `scikit-learn`.
* Fitting and predicting with real datasets.

#### **Gradient Descent**

* Implementation of various gradient descent methods in `descents.py`.
* Linear regression training using these methods in `linear_regression.py`.
* Understanding optimization algorithms through hands-on implementation.

#### **Linear Classification**

* Classification metrics and model evaluation.
* Exploration of SVM and logistic regression.
* Probability calibration, feature transformation, and selection.
* Multi-class classification and a near real-world business case.

#### **Trees**

* Decision trees from `scikit-learn` for classification and hyperparameter analysis.
* Implementation of a custom decision tree for classification.
* A regression tree with linear models in the leaf nodes.

#### **Boosting**

* Custom implementation of gradient boosting.
* Exploration of optimization techniques for boosting.
* Experiments with the capabilities of boosting algorithms.

---

## Deep Learning


#### **Feed-Forward Neural Networks (FNN)**

* Implementation of a fully connected neural network with forward and backward propagation to perform classification on a sample dataset, including hyperparameter tuning and performance visualization.

#### **Convolutional Neural Networks (CNN)**

* Construction and training of a convolutional neural network for image classification, showcasing convolutional, pooling layers and evaluation metrics on validation data.

#### **Recurrent Neural Networks (RNN)**

* Development of a recurrent neural network model using LSTM cells for sequence prediction tasks, featuring sequence preprocessing, training loops, and loss tracking.

* Utility scripts for dataset loading (`dataset.py`), model definitions (`model_l.py`), and training workflow (`train.py`) to streamline experiments and reproducible model training.

#### **Contrastive Language–Image Pretraining (CLIP)**

* Integration of text (`TextEncoder.py`) and image (`ImageEncoder.py`) encoders with a projection head (`ProjectionHead.py`) and custom dataset loader (`CLIPDataset.py`) in a Jupyter notebook (`my_clip.ipynb`), training a contrastive model on paired image-text data.

## Deep Learning in Natural Sciences

#### **Bioinformatics**

* Implementation of deep neural networks for genomic sequence analysis to predict functional annotations, featuring sequence encoding strategies, cross-validation performance metrics, and filter visualization.

#### **Materials Science I**

* Application of convolutional and graph-based neural networks to forecast material properties from atomic structures, including preprocessing of crystallographic data and model evaluation against real-world measurements.

#### **Materials Science II**

* Extension of maching learning workflows for materials discovery using different customizations of classic ML model and uncertainty quantification, showcasing dataset augmentation and robustness analysis across multiple material classes.

#### **Physics-Informed Neural Networks (PINNs)**

* Development of physics-informed neural networks to solve partial differential equations in natural science applications, integrating physical laws into the loss function and tracking physics constraint errors during training.

## Reinforcement Learning

#### **Dynamic Programming (VI/PI)**

* Implementation of value iteration and policy iteration algorithms for solving Markov Decision Processes, with gridworld examples illustrating convergence and policy evaluation.

#### **Deep Q-Networks (DQN)**

* Development of DQN algorithm with experience replay and target networks to learn policies in discrete action spaces, including training on OpenAI Gym environments and evaluation of learning curves.

#### **Deep Cross-Entropy Method**

* Application of the cross-entropy method with neural network function approximation, featuring episode sampling, parameter updates, and performance tracking in control tasks.

#### **Continuous Control (TD3 & SAC)**

* Implementation of Twin Delayed DDPG and Soft Actor-Critic algorithms for continuous action spaces, including actor-critic network structures, entropy regularization, and benchmark comparisons.

#### **Model-Free RL**

* Exploration of model-free reinforcement learning strategies, including Monte Carlo, Temporal Difference learning, and on-policy/off-policy methods, with experiments demonstrating sample efficiency and stability.

## Application of Matrix Analysis and computational linear algebra

This repository contains practice homework projects related to applied aspects of working with matrices and implementing computational linear algebra algorithms in Python.

#### Overview of project

**lab1** - In this project, I worked with one possible application of singular value decomposition - finding a "good" basis (eigenfaces) in a set of images and using it to find similar images.

**lab2** - In this project, I built a recommendation model based on sparse rank approximations of sparse matrices.

**lab3** - In this project, I constructed a tomogram of some object using data on the intensity of rays that passed through it.

**lab4** - In this project, I was concerned with calculating voltages in a resistor system and iterative methods for solving linear systems, specifically comparing different iterative methods for solving systems, both modeled and real large and sparse.
