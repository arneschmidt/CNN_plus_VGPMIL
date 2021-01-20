import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score


def calc_metrics(predictions: np.array, instance_labels: np.array, model_name: str):
    confusion_mat = confusion_matrix(instance_labels, predictions)
    metrics = pd.DataFrame(index=['recall', 'precision', 'accuracy', 'f1_score', 'cohens_kappa'], columns=[model_name])

    metrics.loc['recall', model_name] = round(confusion_mat[0][0] / (confusion_mat[0][0] + confusion_mat[1][0]), 3)
    metrics.loc['precision', model_name] = round(confusion_mat[0][0] / (confusion_mat[0][0] + confusion_mat[0][1]), 3)
    metrics.loc['accuracy', model_name] = round((confusion_mat[0][0] + confusion_mat[1][1]) / (
                confusion_mat[0][0] + confusion_mat[1][0] + confusion_mat[0][1] +
                confusion_mat[1][1]), 3)
    metrics.loc['f1_score', model_name] = round(f1_score(instance_labels, predictions), 3)
    metrics.loc['cohens_kappa', model_name] = round(cohen_kappa_score(instance_labels, predictions), 3)

    return metrics