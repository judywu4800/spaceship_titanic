# Project: Space Titanic  - Survival Prediction

This is the  final project for STAT GU4241 Statistical Machine Learning.

It has now been restructured and extended for **STAT 607** Project 1 to emphasize reproducibility practices.

Term: Spring 2022

+ Project summary: This project used the fictional dataset from Kaggle Competition [Spaceship Titanic
](https://www.kaggle.com/competitions/spaceship-titanic), which records the interstellar passenger's information on a spaceship that was traveling to three newly habitable exoplanets and had an accident. The project aims at constructing binary classification models to predict whether passengers on the spaceship  were transported to an alternate dimension, which would be beneficial for retrieving and rescuing process. In the project, EDA was conducted and algorithms including LDA, KNN. Naive Bayes, Logistic Regression, Random Forest and Gradient Boosting was used to build predictive models. The highest prediction accuracy is around 80 percent among these models.


This folder is organized as follows.

```
spaceship_titanic/
├── data/
│ ├── raw/ # immutable source (read-only)
│ └── processed/ # cleaned data
├── src/
│ ├── pipeline/ # populates processed/ and artifacts/
│ └── analysis/ # populates results/
├── artifacts/ # intermediate outputs
│ ├── sufficient-stats/
│ └── models/
├── results/
| ├── tables/
│ ├── figures/
│ └── report/
├── tests/
├── docs/
├── requirements.txt (or environment.yml)
├── run_analysis.py (or Makefile)
├── README.md
└── .gitignore
```

## Environment Setup

To reproduce results on a fresh system:

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/spaceship_titanic.git
   cd spaceship_titanic
   ```
   
2. **Create and activate virtual environment**
   ```bash
    python3 -m venv .venv
    source .venv/bin/activate     
   ```
   
3. **Install dependencies**
   ```bash
    pip install -r requirements.txt
   ```

## Usage
### Run the full pipeline
   ```bash
   python run_analysis.py
   ```
This script peforms:
1. Data processing → generates train_processed.csv and test_processed.csv
2. Model training → trains multiple classifiers, saves models and sufficient stats
3. Evaluation → evaluates models, saves metrics and confusion matrices

### Results
+ Trained models → artifacts/models/
+ Cross-validation stats → artifacts/sufficient-stats/
+ Evaluation metrics → results/tables/
+ Confusion matrices → results/figures/
+ 
Example expected output (from terminal):
```makefile
lda: 0.7625
qda: 0.7450
log_reg: 0.7750
knn: 0.7380
random_forest: 0.8015
gbdt: 0.7950
naive_bayes: 0.7205
```


