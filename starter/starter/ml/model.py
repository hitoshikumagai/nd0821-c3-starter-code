import yaml
import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier as Model

with open('config.yaml','rb') as yml:
    config = yaml.safe_load(yml)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    model : Model defined in train_model.py
        Machine Learning Model
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = Model(config['model']['random_state'])
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def compute_metrics_sliced_category(model, X, y, category_lst:list):
    for idx, category in enumerate(category_lst):
        u, indices = np.unique(X[idx], return_inverse =True)
        for indice in  indices:
            # todo udacityの教材をもう一度見てみる
#            print(category, X[:,indice])
#            preds = inference(model, X[:, indice])
#            return compute_model_metrics(y[indice], preds)

