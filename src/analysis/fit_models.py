import os
import pickle
import json
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Default data and artifact directories
DEFAULT_INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "train_processed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
SUFF_DIR = os.path.join(BASE_DIR, "artifacts", "sufficient-stats")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUFF_DIR, exist_ok=True)

# Candidate models (LDA/QDA wrapped with scaler for stability)
MODELS = {
    "lda": Pipeline([("scaler", StandardScaler()), ("lda", LDA())]),
    "qda": Pipeline([("scaler", StandardScaler()), ("qda", QDA())]),
    "log_reg": LogisticRegression(max_iter=1000),
    "knn": KNeighborsClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier(),
    "gbdt": GradientBoostingClassifier(),
    "naive_bayes": GaussianNB()
}


def fit_models(input_path: str = DEFAULT_INPUT_PATH,
               output_dir: str = MODELS_DIR,
               suff_dir: str = SUFF_DIR) -> dict:
    """
    Train multiple classification models on the processed dataset and save them,
    along with cross-validation scores, feature importances/coefficients, and parameters.

    Parameters
    ----------
    input_path : str, optional
        Absolute path to the processed training dataset (CSV).
    output_dir : str, optional
        Directory to save the trained models.
    suff_dir : str, optional
        Directory to save sufficient statistics (CV scores, coefficients, params).

    Returns
    -------
    results : dict
        Dictionary mapping model names to their mean cross-validation accuracy.
    """
    # 1. Load dataset
    df = pd.read_csv(input_path)
    X = df.drop("Transported", axis=1)
    y = df["Transported"]

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Cross-validation setup
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)

    results = {}

    # 4. Train, evaluate, save
    for name, model in MODELS.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
            mean_score = scores.mean()
            results[name] = mean_score
            print(f"{name}: {mean_score:.4f}")

            # Save CV scores
            cv_scores_df = pd.DataFrame({"cv_scores": scores})
            cv_scores_df.to_csv(os.path.join(suff_dir, f"{name}_cv_scores.csv"), index=False)

            # Fit final model
            model.fit(X_train, y_train)

            # Save model
            model_path = os.path.join(output_dir, f"{name}.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save coefficients / feature importances if available
            if hasattr(model, "coef_"):
                coef_df = pd.DataFrame(model.coef_, columns=X_train.columns)
                coef_df.to_csv(os.path.join(suff_dir, f"{name}_coefficients.csv"), index=False)

            if hasattr(model, "feature_importances_"):
                fi_df = pd.DataFrame({
                    "feature": X_train.columns,
                    "importance": model.feature_importances_
                }).sort_values(by="importance", ascending=False)
                fi_df.to_csv(os.path.join(suff_dir, f"{name}_feature_importances.csv"), index=False)

            # Save parameters
            with open(os.path.join(suff_dir, f"{name}_params.json"), "w") as f:
                json.dump(model.get_params(), f, indent=4)

        except Exception as e:
            print(f"{name} failed: {e}")
            results[name] = None
            continue

    print(f"Models and sufficient stats saved to {output_dir} and {suff_dir}")
    return results


if __name__ == "__main__":
    fit_models()
