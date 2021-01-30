from typing import Dict
import pandas as pd
import numpy as np


def convert_to_vgpmil_input(df: pd.DataFrame, config: Dict, train_with_instance_labels: bool=False):
    # we try to automatically derive the column names
    col_feature_prefix = config['col_feature_prefix']
    col_bag_label = config['col_bag_label']
    col_bag_name = config['col_bag_name']
    col_instance_label = config['col_instance_label']

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
        instance_labels = (df[col_instance_label].to_numpy().astype("int"))  # instance_label column
        if train_with_instance_labels:
            pi = np.random.uniform(0, 0.1, size=len(df))  # -1 for untagged
            pi = np.where((0 == instance_labels), 0, pi)
            pi = np.where((0 < instance_labels), 1, pi)

            mask = np.where(instance_labels > -1, False, True)

    return features, bag_labels_per_instance, bag_names_per_instance, Z, pi, mask, instance_labels

