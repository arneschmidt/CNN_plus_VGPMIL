from __future__ import print_function
import yaml
import os
import numpy as np
import pandas as pd
from vgpmil.helperfunctions import RBF
from vgpmil.vgpmil import vgpmil
from typing import Dict

from loading import convert_to_vgpmil_input
from metrics import calc_metrics


def train(vgpmil_model: vgpmil, config: Dict):
    train_df = pd.read_csv(config['path_train_df'])
    features, bag_labels_per_instance, bag_names_per_instance, Z, pi, mask, _ = convert_to_vgpmil_input(train_df)
    vgpmil_model.train(features, bag_labels_per_instance, bag_names_per_instance, Z, pi, mask)

def test(vgpmil_model: vgpmil, config: Dict):
    test_df = pd.read_csv(config['path_test_df'])
    features, _, _, _, _, _, instance_labels = convert_to_vgpmil_input(test_df)
    predictions = vgpmil_model.predict(features)
    predictions = np.where(predictions >= 0.5, 1, 0).astype("float32")
    metrics = calc_metrics(predictions, instance_labels, 'vgpmil')
    out_name = os.path.basename(config['path_test_df']).split('.')[0]

    col_cnn_prediction = 'prediction (instance)'
    if col_cnn_prediction in test_df.columns:
        cnn_predictions = test_df[col_cnn_prediction].to_numpy().astype("float32")
        cnn_metrics = calc_metrics(cnn_predictions, instance_labels, 'cnn')
        metrics = pd.concat([metrics, cnn_metrics], axis=1)
    metrics.to_csv(os.path.join(config['output_path'], 'metrics_' + out_name + '.csv'))


def main():
    with open('config.yaml') as file:
        config = yaml.full_load(file)
    kernel = RBF()
    vgppmil_model = vgpmil(kernel=kernel,
                           num_inducing=int(config['inducing_points']),
                           max_iter=int(config['iterations']),
                           normalize=bool(config['normalize']),
                           verbose=bool(config['verbose']))
    train(vgppmil_model, config)
    test(vgppmil_model, config)


if __name__ == "__main__":
    main()
