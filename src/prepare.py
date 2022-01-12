from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def get_data(df, typ):
    if typ.lower() == 'test':
        return df[['Pclass', 'Age', 'SibSp']]
    return df[['Pclass', 'Age', 'SibSp', 'Survived']]


if __name__ == "__main__":

    repo_path = Path(__file__).parent.parent

    training_set = pd.read_csv(repo_path / 'data/raw/titanic_train.csv')
    training_set, val_set = train_test_split(training_set, test_size=0.2, random_state=42)
    test_set = pd.read_csv(repo_path / 'data/raw/titanic_test.csv')

    get_data(training_set, 'train').to_csv(repo_path / 'data/prepared/training_set.csv', index=0)
    get_data(val_set, 'val').to_csv(repo_path / 'data/prepared/val_set.csv', index=0)
    get_data(test_set, 'test').to_csv(repo_path / 'data/prepared/test_set.csv', index=0)

