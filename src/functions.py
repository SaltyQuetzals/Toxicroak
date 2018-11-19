from constants import DATA_PATH, LABEL_COLS

import pandas as pd
import re
import string
from sklearn import metrics


def load_bad_words():
    """
    Loads a list of obscene words.

    Returns:
    - A list of strings.
    """
    with open(DATA_PATH + 'bad_words.txt', 'r') as f:
        return f.read().split('\n')
    
def load_ethnic_slurs():
    """
    Loads a list (from Wikipedia) of ethnic slurs.
    
    Returns:
    - A list of strings.
    """
    with open(DATA_PATH + 'slurs.txt', 'r') as f:
        return f.read().split('\n')


def build_data_path(filename, use_preprocessed=False):
    """
    Returns the path to the desired file. Depends on whether use_preprocessed is True or False.
    
    Args:
    - filename: The name of the file to build a path to. Should be in the data/ directory.
    - use_preprocessed: Whether or not to use preprocessed data. Defaults to False.
    Returns:
    - A path string.
    """
    training_data_path = DATA_PATH
    if use_preprocessed:
        training_data_path += '/preprocessed/'
    else:
        training_data_path += '/raw/'
    training_data_path += filename

    return training_data_path


def print_report(y_truth, y_predictions, data_type='VALIDATION'):
    print(f'{data_type} RESULTS:')
    print()
    print(metrics.classification_report(
        y_truth, y_predictions, target_names=LABEL_COLS))
    print('Class-wise AUC-ROC (Kaggle)', metrics.roc_auc_score(y_truth, y_predictions, average=None))
    print('Overall AUC-ROC (Kaggle)', metrics.roc_auc_score(y_truth, y_predictions, average='macro'))


def run_on_test_data(model):
    test_data = build_data_path('test.csv')

    data_df = pd.read_csv(test_data)

    test_labels = build_data_path('test_labels.csv')
    label_df = pd.read_csv(test_labels)

    test_df = data_df.set_index('id').join(label_df.set_index('id'))
    CONDITIONS = [f'{label} != -1' for label in LABEL_COLS]
    QUERY_STRING = ' & '.join(CONDITIONS)
    test_df = test_df.query(QUERY_STRING)
    X_test = test_df['comment_text']
    y_test = test_df[LABEL_COLS]

    y_predictions = model.predict(X_test)

    print_report(y_test, y_predictions, data_type='TESTING')