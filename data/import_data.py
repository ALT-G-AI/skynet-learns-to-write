import pandas as pd
from sklearn.model_selection import train_test_split

TRAINING_PATH = './data/train.csv'


def import_data(training_path=TRAINING_PATH):
    frame = pd.read_csv(training_path)
    train_set, test_set = train_test_split(
        frame,
        test_size=0.2,
        random_state=0)

    return train_set, test_set
