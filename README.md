# Room Occupancy Estimation from Multi-Sensor Data

![scikit-learn](https://img.shields.io/badge/scikit--learn-classical%20ML-f7931e)
![Keras](https://img.shields.io/badge/Keras-neural%20network-d00000)
![Dataset](https://img.shields.io/badge/dataset-UCI%20Room%20Occupancy%20Estimation-lightgrey)

This project estimates how many people are in a room (0–3) from non-intrusive environmental sensors, and studies how the choice of data representation affects classifiers on a heavily imbalanced dataset.

*Graduate team project, UC Riverside (2023). Each team member implemented one model family; notebook names carry the author's initials.*

## Problem

The [UCI Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation) dataset records temperature, light, sound, CO₂ (level and slope) and passive-infrared motion from several sensor nodes in a room. The task is to predict the occupant count. The classes are strongly imbalanced, with the empty-room class dominating, so plain accuracy is misleading and per-class F1 is reported.

## Approach

- **Exploratory analysis:** feature correlations and class balance.
- **Class imbalance:** majority-class undersampling.
- **Representations:** raw features vs. PCA, LDA, t-SNE, factor analysis, truncated SVD and ICA projections.
- **Models:** logistic regression, Naive Bayes, decision tree, SVM and a feed-forward neural network, each compared against off-the-shelf implementations.

## Results

- **Best representation:** LDA. With logistic regression on LDA features, the model reaches **0.99 accuracy / F1** on the held-out set (`faisal_LR-checkpoint.ipynb`).
- **Decision tree:** reaches 0.996 test accuracy (`Decision-Tree-KR-checkpoint.ipynb`).

## Notebooks

| Notebook | Content |
|---|---|
| `faisal_LR-checkpoint.ipynb` | Logistic regression + comparison of data representations |
| `Decision-Tree-KR-checkpoint.ipynb` | Decision tree |
| `Neural-Network-JI-checkpoint.ipynb` | Feed-forward neural network (Keras), with class balancing |
| `Naive-Bayes-NT-checkpoint.ipynb`, `Support-Vector-Machine-SM-checkpoint.ipynb`, `Logistic-Regression-FB-checkpoint.ipynb`, `Data_Analysis-checkpoint.ipynb` | *To be uploaded* |

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow tqdm
```

Download `Occupancy_Estimation.csv` from the UCI page and place it next to the notebooks.
