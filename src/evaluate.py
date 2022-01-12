from joblib import load
import json
from pathlib import Path
from sklearn.metrics import accuracy_score
import pandas as pd

def main(repo_path):

    test_csv_path = repo_path / "data/prepared/val_set.csv"
    val_set = pd.read_csv(test_csv_path)
    val_set = val_set.fillna(0)
    x_val = val_set[['Pclass','Age','SibSp']]
    y_val = val_set['Survived']

    model = load(repo_path / "models/model.joblib")
    predictions = model.predict(x_val)

    accuracy = accuracy_score(y_val, predictions)
    metrics = {"accuracy": accuracy}
    accuracy_path = repo_path / "metrics/accuracy.json"
    accuracy_path.write_text(json.dumps(metrics))


if __name__ == "__main__":

    repo_path = Path(__file__).parent.parent

    main(repo_path)
