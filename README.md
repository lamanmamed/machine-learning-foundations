# Machine Learning Foundations

This repository brings together core machine learning methods across supervised learning and probabilistic modelling. Most of the main algorithms are implemented directly so the optimisation and inference steps are visible in the code rather than hidden behind high-level estimators.

The project covers linear and polynomial regression, logistic regression, neural networks, and Gaussian Mixture Models trained with Expectation-Maximization. The experiments use diabetes data, Iris classification, XOR, and vowel recognition from acoustic features.

## Supervised learning

### Linear regression

I implemented linear regression with gradient descent and tested how the learning rate affects convergence on the diabetes dataset.

The same model was trained for 100 iterations with four learning rates:

| Learning rate | Training MSE | Test MSE |
| ---: | ---: | ---: |
| 1.0 | 29029.56 before divergence | infinity |
| 0.1 | **2890.02** | **2886.15** |
| 0.01 | 3351.27 | 3425.44 |
| 0.001 | 19769.70 | 18346.24 |

A learning rate of `1.0` diverged. `0.001` was too slow for the fixed number of iterations. Of the values tested, `0.1` gave the lowest training and test error.

The implementation includes:

- feature standardisation
- an explicit bias feature
- mean squared error
- the analytical MSE gradient
- gradient-descent weight updates

### Polynomial regression and L2 regularisation

I also fitted a fifth-degree polynomial using the feature expansion

```text
[1, x, x², x³, x⁴, x⁵]
```

and compared different L2 penalties.

| λ | Recorded final cost |
| ---: | ---: |
| 0 | 0.0082 |
| 0.01 | 0.0089 |
| 1 | 0.0529 |
| 100 | 0.2563 |

The bias term is excluded from the penalty. As the regularisation strength increases, the higher-order coefficients are pushed closer to zero and the fitted curve becomes less flexible.

### Logistic regression on Iris

For classification, I implemented binary logistic regression with a sigmoid output, binary cross-entropy, and manual gradient updates.

For the three-class Iris dataset, I trained three one-vs-rest classifiers:

```text
Setosa vs. the rest
Versicolor vs. the rest
Virginica vs. the rest
```

Using the same 80/20 split with `random_state=42`, the model classified **27 of 30 test samples correctly**, giving **90% accuracy**. The cleaned implementation reproduces the same result.

### Neural networks: XOR and Iris

The first neural-network experiment is a small network built manually for XOR. It has two hidden sigmoid units and one sigmoid output, with the forward pass and backpropagation updates written directly.

After training:

```text
Targets:      [0, 1, 1, 0]
Predictions:  [0.0046, 0.9953, 0.9952, 0.0058]
Rounded:      [0, 1, 1, 0]
```

The cleaned implementation reproduces these predictions to the shown precision.

I then used PyTorch to train one-hidden-layer networks on Iris with different hidden sizes:

| Hidden neurons | Final training loss | Final test loss |
| ---: | ---: | ---: |
| 1 | 0.4689 | 0.4657 |
| 2 | 0.2678 | 0.2471 |
| 4 | 0.1619 | 0.1377 |
| 8 | 0.1548 | 0.1310 |
| 16 | 0.1267 | 0.1081 |
| 32 | 0.1112 | 0.0915 |

The XOR network keeps the learning equations visible. The Iris experiment uses the usual PyTorch training loop with `nn.Linear`, automatic differentiation, and SGD.

## Gaussian Mixture Models

The second project uses the Peterson and Barney vowel dataset. The main features are the first two formant frequencies, `F1` and `F2`.

I implemented a Gaussian Mixture Model trained with Expectation-Maximization. The implementation supports diagonal covariance matrices for the main experiments and full covariance matrices for the singularity experiment.

### EM implementation

For each iteration:

1. the E-step computes the responsibility of each Gaussian component for every observation
2. the M-step updates mixture weights, means, and covariance matrices
3. the log-likelihood is tracked until convergence

I also tested the sensitivity of the fitted components to random initialisation. For the three-component model of phoneme 1, repeated runs converged to very similar cluster locations, although the component labels could swap.

### Vowel classification

Two GMMs are fitted separately, one for each phoneme. A new sample is assigned to the model with the larger likelihood.

The submitted experiment reported:

| Mixture components | Accuracy | Misclassification error |
| ---: | ---: | ---: |
| K = 3 | **95.07%** | 4.93% |
| K = 6 | **95.72%** | 4.28% |

I reran both experiments with the cleaned implementation and the submitted `PB_data.npy`. With fixed initialisations, the results reproduce **95.07%** for `K=3` and **95.72%** for `K=6`.

The six-component model performs slightly better, but the difference is less than one percentage point.

### Covariance singularity

One experiment deliberately adds a dependent third feature:

```text
F3 = F1 + F2
```

so each sample becomes:

```text
[F1, F2, F1 + F2]
```

Because the third feature is an exact linear combination of the first two, the 3×3 covariance matrix has rank 2.

The cleaned code reproduces this:

```text
3D covariance rank: 2 for 3 features
```

A full-covariance Gaussian cannot invert that covariance matrix without regularisation. The implementation fixes this by adding a small value to the diagonal after each covariance update:

```text
Σ ← Σ + εI
```

with `ε = 0.001`.

## Repository structure

```text
machine-learning-foundations/
├── src/ml_foundations/
│   ├── datasets.py
│   ├── supervised/
│   │   ├── preprocessing.py
│   │   ├── regression.py
│   │   ├── classification.py
│   │   └── neural_networks.py
│   └── probabilistic/
│       ├── gaussian_mixture.py
│       └── phoneme_classifier.py
├── experiments/
│   ├── diabetes_regression.py
│   ├── iris_classification.py
│   ├── xor_network.py
│   └── vowel_gmm.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_regression.py
│   ├── test_classification.py
│   ├── test_neural_networks.py
│   └── test_gmm.py
├── data/
│   └── README.md
├── pyproject.toml
└── README.md
```

## Running the project

Install the package:

```bash
pip install -e .
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

The cleaned project has **11 tests** covering preprocessing, regression updates, polynomial features, logistic classification, the Iris result, XOR learning, GMM fitting, covariance rank, and covariance regularisation.

Run the experiments individually:

```bash
python experiments/iris_classification.py
python experiments/xor_network.py
python experiments/diabetes_regression.py
python experiments/vowel_gmm.py path/to/PB_data.npy
```

## Data

The Iris dataset is loaded from scikit-learn.

The diabetes experiment uses the same public raw diabetes table referenced in the original notebook and requires internet access when run through the experiment script.

`PB_data.npy` is not included in the repository because its redistribution terms were not provided with the dataset. The expected fields are documented in `data/README.md`.