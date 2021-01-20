import pandas as pd
import numpy as np


def convert_to_vgpmil_input(df: pd.DataFrame, train_with_instance_labels: bool=False):
    col_feature_prefix = 'feature'
    col_bag_label = 'Scan_label (bag)'
    col_bag_name = 'Scan'
    col_instance_label = 'groundtruth (instance)'

    # find all feature columns
    col_features = []
    for col in df.columns:
        if col_feature_prefix in col:
            col_features.append(col)
    bag_labels_per_instance = df[col_bag_label].to_numpy().astype('int')
    bag_names_per_instance = df[col_bag_name].to_numpy().astype('str')

    features = df[col_features].to_numpy().astype('float32')
    instance_labels = None
    pi = None
    mask = None
    Z = None

    if col_instance_label in df.columns:
        instance_labels = (df['groundtruth (instance)'].to_numpy().astype("int"))  # instance_label column
        if train_with_instance_labels:
            pi = np.random.uniform(0, 0.1, size=len(df))  # -1 for untagged
            pi = np.where((0 == instance_labels), 0, pi)
            pi = np.where((0 < instance_labels), 1, pi)

            mask = np.where(instance_labels > -1, False, True)
    return features, bag_labels_per_instance, bag_names_per_instance, Z, pi, mask, instance_labels

