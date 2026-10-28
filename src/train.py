"""Train a classifier and save it as an MLflow model.

Runs the same way locally and as an Azure ML command job.
"""
import argparse
import os
import shutil

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="CSV file or folder containing one CSV. Defaults to the built-in breast cancer dataset.")
    p.add_argument("--target", default="target", help="Name of the label column in the CSV.")
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--model_output", default="outputs/model", help="Where to save the MLflow model.")
    return p.parse_args()


def load_data(path, target):
    if path is None:
        df = load_breast_cancer(as_frame=True).frame
        target = "target"
    else:
        import glob
        import os

        if os.path.isdir(path):
            path = glob.glob(os.path.join(path, "*.csv"))[0]
        df = pd.read_csv(path)
    return df.drop(columns=[target]), df[target]


def main():
    args = parse_args()
    X, y = load_data(args.data, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    with mlflow.start_run():
        mlflow.log_params(vars(args))

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
            )),
        ])
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="weighted"),
        }
        if y.nunique() == 2:
            metrics["roc_auc"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics(metrics)
        print("Metrics:", metrics)

        signature = mlflow.models.infer_signature(X_test, preds)
        # MLflow refuses a non-empty folder; clear the previous local model.
        # (In Azure ML the output folder is always empty, so this is skipped.)
        if os.path.isdir(args.model_output) and os.listdir(args.model_output):
            shutil.rmtree(args.model_output)
        mlflow.sklearn.save_model(
            model,
            args.model_output,
            signature=signature,
            # Azure ML's auto-generated MLflow scoring script imports this package,
            # so the serving container crashes on startup without it.
            extra_pip_requirements=["azureml-ai-monitoring"],
        )
        print(f"Model saved to {args.model_output}")


if __name__ == "__main__":
    main()
