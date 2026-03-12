from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def build_model(task: str, name: str, params: dict):
    if task == "classification":
        if name == "logreg":
            return LogisticRegression(max_iter=params.get("max_iter", 200))
        if name == "rf":
            return RandomForestClassifier(
                n_estimators=params.get("n_estimators", 200),
                max_depth=params.get("max_depth", None),
                random_state=params.get("seed", 42),
                n_jobs=-1
            )
    if task == "regression":
        if name == "linreg":
            return LinearRegression()
        if name == "rf":
            return RandomForestRegressor(
                n_estimators=params.get("n_estimators", 300),
                max_depth=params.get("max_depth", None),
                random_state=params.get("seed", 42),
                n_jobs=-1
            )
    raise ValueError("Unknown task/model")
