import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def train_model(data_path: str, model_path: str):
    """Function for Model Training

    Args:
        data_path (str): DataFrame Path
        model_path (str): Model Path

    Returns:
        _type_: Returns Pipeline, X_test, y_test
    """
    df = pd.read_csv(data_path)

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )

    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, model_path)

    return pipeline, X_test, y_test