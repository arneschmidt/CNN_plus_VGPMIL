from __future__ import print_function
import yaml
import os
import numpy as np
import pandas as pd
from vgpmil.helperfunctions import RBF
from vgpmil.vgpmil import vgpmil
from typing import Dict

from loading import convert_to_vgpmil_input, load_cnn_predictions
from metrics import calc_instance_level_metrics, calc_bag_level_metrics


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

    # calculate the metrics for vgpmil
    vgpmil_probabilities = vgpmil_model.predict(features)
    predictions = np.where(vgpmil_probabilities >= 0.5, 1, 0).astype("float32")
    vgpmil_instance_metrics = calc_instance_level_metrics(predictions, instance_labels, 'vgpmil')
    vgpmil_bag_metrics = calc_bag_level_metrics(predictions, bag_labels_per_instance, bag_names_per_instance, 'vgpmil')
    vgpmil_metrics = pd.concat([vgpmil_instance_metrics, vgpmil_bag_metrics], axis=0)

    # calculate the metrics for the CNN
    cnn_predictions, bag_cnn_predictions, bag_cnn_probabilities = load_cnn_predictions(test_df, config)
    cnn_instance_metrics = calc_instance_level_metrics(cnn_predictions, instance_labels, 'cnn')
    cnn_bag_metrics = calc_bag_level_metrics(bag_cnn_predictions, bag_labels_per_instance, bag_names_per_instance, bag_cnn_probabilities, 'cnn')

    cnn_metrics = pd.concat([cnn_instance_metrics, cnn_bag_metrics], axis=0)

    metrics = pd.concat([vgpmil_metrics, cnn_metrics], axis=1)

    out_name = os.path.basename(config['path_test_df']).split('.')[0]
    out_file = os.path.join(config['output_path'], 'metrics_' + out_name + '.csv')
    os.makedirs(config['output_path'], exist_ok=True)

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
