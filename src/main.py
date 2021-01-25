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
    print('Training..')
    train_df = pd.read_csv(config['path_train_df'])
    print('Loaded training dataframe. Number of instances: ' + str(len(train_df)))
    features, bag_labels_per_instance, bag_names_per_instance, Z, pi, mask, _ = convert_to_vgpmil_input(train_df, config)
    vgpmil_model.train(features, bag_labels_per_instance, bag_names_per_instance, Z, pi, mask)

def test(vgpmil_model: vgpmil, config: Dict):
    print('Testing..')
    test_df = pd.read_csv(config['path_test_df'])
    print('Loaded test dataframe. Number of instances: ' + str(len(test_df)))
    features, bag_labels_per_instance, bag_names_per_instance, _, _, _, instance_labels = convert_to_vgpmil_input(test_df, config)
    predictions = vgpmil_model.predict(features)
    predictions = np.where(predictions >= 0.5, 1, 0).astype("float32")
    metrics = calc_metrics(predictions, instance_labels, bag_labels_per_instance, bag_names_per_instance, 'vgpmil')
    out_name = os.path.basename(config['path_test_df']).split('.')[0]

    col_cnn_prediction = config['col_cnn_prediction']
    if col_cnn_prediction in test_df.columns:
        print('Found CNN predictions. Comparing..')
        cnn_predictions = test_df[col_cnn_prediction].to_numpy().astype("float32")
        col_bag_cnn_predictions = config['col_bag_cnn_predictions']
        bag_cnn_predictions = None
        if col_bag_cnn_predictions in test_df.columns:
            print('Using CNN bag predictions')
            bag_cnn_predictions = (test_df[col_bag_cnn_predictions].to_numpy().astype("int"))
        cnn_metrics = calc_metrics(cnn_predictions, instance_labels, bag_labels_per_instance, bag_names_per_instance, 'cnn', bag_cnn_predictions)
        metrics = pd.concat([metrics, cnn_metrics], axis=1)

    out_file = os.path.join(config['output_path'], 'metrics_' + out_name + '.csv')
    print('Save output to ' + out_file)
    metrics.to_csv(out_file)


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
