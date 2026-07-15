# Spatial Active Learning for Marine Microplastics

A machine learning framework for evaluating active learning strategies that efficiently predict marine microplastic concentrations using spatially distributed environmental sampling data.

## Overview

Collecting marine microplastics data is expensive, time-consuming, and geographically sparse. This project investigates whether active learning can reduce the number of required labeled samples while maintaining predictive performance.

Using a Mediterranean Sea microplastics dataset, multiple active learning strategies are compared to determine which sampling approach produces the most accurate predictions under limited labeling budgets.

The framework emphasizes reproducibility through spatial validation, repeated experiments across multiple random seeds, statistical hypothesis testing, and publication-quality visualizations.

---

## Research Objectives

- Predict marine microplastic concentrations using machine learning.
- Compare multiple active learning sampling strategies.
- Evaluate how different sampling strategies affect prediction accuracy.
- Determine whether spatially informed sampling improves model performance over traditional methods.

---

## Active Learning Strategies

The following acquisition strategies are implemented:

- **Random Sampling** — Randomly selects unlabeled samples.
- **Uncertainty Sampling** — Selects samples with the highest Gaussian Process prediction uncertainty.
- **Geographic Sampling** — Prioritizes spatial coverage by selecting geographically distant samples.
- **Hybrid Sampling** — Combines uncertainty and geographic diversity using a weighted scoring function.

---

## Methodology

### Data Preprocessing

The preprocessing pipeline:

- Loads the raw Mediterranean microplastics dataset
- Validates required columns
- Converts dates and numeric features
- Removes duplicate observations
- Removes invalid values
- Creates temporal features
- Applies a log transformation to microplastic concentration

---

### Machine Learning Models

Three baseline regression models are evaluated:

- Linear Regression
- Random Forest Regressor
- Gaussian Process Regressor

Gaussian Process Regression serves as the primary model for active learning because it provides predictive uncertainty estimates required for uncertainty-based sampling.

---

### Spatial Validation

To prevent overly optimistic performance estimates, the dataset is divided into geographic blocks rather than using a purely random train-test split.

This better reflects deployment to previously unseen regions.

---

### Active Learning Evaluation

Each strategy is evaluated using repeated experiments across multiple random seeds.

Sampling budgets range from 30 to 200 labeled observations.

Performance metrics include:

- Log Mean Absolute Error (Log MAE)
- Log Root Mean Squared Error (Log RMSE)
- Log R²
- Original-scale MAE
- Original-scale RMSE
- Original-scale R²

---

### Statistical Analysis

Statistical significance is evaluated using:

- Wilcoxon Signed-Rank Test
- Paired t-test
- Holm multiple-comparison correction
- Cohen's *d*
- Rank-biserial correlation

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
│
├── figures/
│   ├── primary/
│   └── supplementary/
│
├── results/
│   ├── statistics/
│   └── *.csv
│
├── src/
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── active_learning.py
│   ├── visualization.py
│   ├── statistics.py
│   └── __init__.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/spatial-active-learning-microplastics.git
cd spatial-active-learning-microplastics
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

macOS / Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the complete pipeline:

```bash
python main.py
```

This will:

1. preprocess the dataset
2. train baseline models
3. run active learning experiments
4. generate visualizations
5. perform statistical analysis
6. save all figures and results

---

## Generated Outputs

### Figures

- Learning curves
- Strategy comparison plots
- Improvement over random sampling
- Supplementary performance figures

### Results

- Baseline model metrics
- Active learning experiment results
- Statistical hypothesis tests
- Final performance summaries

---

## Dataset

This project uses a Mediterranean Sea marine microplastics dataset.

The repository does not include the original raw dataset. Place the dataset in:

```text
data/raw/
```

before running the pipeline.

---

## Dependencies

- Python 3.11+
- NumPy
- pandas
- matplotlib
- scikit-learn
- SciPy

Install with:

```bash
pip install -r requirements.txt
```

---

## License

This repository is intended for academic and research purposes.
