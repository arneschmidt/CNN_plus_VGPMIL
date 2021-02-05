from __future__ import print_function
import yaml
import os
import numpy as np
import pandas as pd
from vgpmil.helperfunctions import RBF
from vgpmil.vgpmil import vgpmil
from typing import Dict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from loading import load_dataframe, load_cnn_predictions, get_bag_level_information
from metrics import Metrics

def initialize_models(config):
    vgpmil_model = None
    random_forest_model = None
    svm_model = None

    if config['use_models']['vgpmil'] == True:
        vgpmil_config = config['vgpmil']
        kernel = RBF()
        vgpmil_model = vgpmil(kernel=kernel,
                               num_inducing=int(vgpmil_config['inducing_points']),
                               max_iter=int(vgpmil_config['iterations']),
                               normalize=bool(vgpmil_config['normalize']),
                               verbose=bool(vgpmil_config['verbose']))
    if config['use_models']['random_forest'] == True:
        random_forest_model = RandomForestClassifier()
    if config['use_models']['svm'] == True:
        svm_model = SVC()

    return vgpmil_model, random_forest_model, svm_model


def train(config: Dict, vgpmil_model: vgpmil = None, rf_model: RandomForestClassifier = None, svm_model: SVC = None):
    print('Training..')
    train_df = pd.read_csv(config['path_train_df'])
    print('Loaded training dataframe. Number of instances: ' + str(len(train_df)))
    features, bag_labels_per_instance, bag_names_per_instance, _ = load_dataframe(train_df, config)
    bag_features, bag_labels, bag_names = get_bag_level_information(features, bag_labels_per_instance, bag_names_per_instance)
    if vgpmil_model is not None:
        print('Train VGPMIL')
        vgpmil_model.train(features, bag_labels_per_instance, bag_names_per_instance, Z=None, pi=None, mask=None)
    if rf_model is not None:
        rf_model.fit(X=bag_features, y=bag_labels)
    if svm_model is not None:
        svm_model.fit(X=bag_features, y=bag_labels)

def test(config: Dict, vgpmil_model: vgpmil = None, rf_model: RandomForestClassifier = None, svm_model: SVC = None):
    print('Testing..')
    test_df = pd.read_csv(config['path_test_df'])
    print('Loaded test dataframe. Number of instances: ' + str(len(test_df)))
    features, bag_labels_per_instance, bag_names_per_instance, instance_labels = load_dataframe(test_df, config)
    bag_features, bag_labels, bag_names = get_bag_level_information(features, bag_labels_per_instance, bag_names_per_instance)


    # calculate the metrics for vgpmil
    predictions = vgpmil_model.predict(features)
    predictions = np.where(predictions >= 0.5, 1, 0).astype("float32")
    vgpmil_instance_metrics = calc_instance_level_metrics(predictions, instance_labels, 'vgpmil')
    vgpmil_bag_metrics = calc_bag_level_metrics(predictions, bag_labels_per_instance, bag_names_per_instance, 'vgpmil')
    vgpmil_metrics = pd.concat([vgpmil_instance_metrics, vgpmil_bag_metrics], axis=0)

    # calculate the metrics for the CNN
    cnn_predictions, bag_cnn_predictions = load_cnn_predictions(test_df, config)
    cnn_instance_metrics = calc_instance_level_metrics(cnn_predictions, instance_labels, 'cnn')
    cnn_bag_metrics = calc_bag_level_metrics(bag_cnn_predictions, bag_labels_per_instance, bag_names_per_instance, 'cnn')

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
    vgpmil_model, random_forest_model, svm_model = initialize_models(config)
    train(config, vgpmil_model, random_forest_model, svm_model)
    test(config, vgpmil_model, random_forest_model, svm_model)


if __name__ == "__main__":
    main()
