import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, cohen_kappa_score, accuracy_score, recall_score, precision_score

class Metrics():
    def __init__(self, instance_labels, bag_labels, bag_names, bag_names_per_instance):
        self.instance_labels = instance_labels
        self.bag_names = bag_names
        self.bag_names_per_instance = bag_names_per_instance
        self.bag_labels = bag_labels
        self.metrics_df = pd.DataFrame(index=['recall', 'precision', 'accuracy', 'f1_score', 'cohens_kappa',
                                              'bag_accuracy', 'bag_f1_score', 'bag_cohens_kappa'])

    def calc_metrics(self, instance_predictions: np.array, bag_predictions: np.array, model_name: str):
        self.calc_instance_level_metrics(instance_predictions, model_name)
        self.calc_bag_level_metrics(bag_predictions, model_name)

    def calc_instance_level_metrics(self, predictions: np.array, model_name: str):
        # To calculate the instance metrics we need gt and predictions of the same size.
        # If the instance labels are not available in the table, an empty array is passed
        if self.instance_labels.size == predictions.size != 0:
            print('Calculate instance metrics of ' + model_name)

            #confusion matrix
            #confusion_mat = confusion_matrix(instance_labels, predictions)

            self.metrics_df.loc['recall', model_name] = round(recall_score(self.instance_labels, predictions), 3)
            self.metrics_df.loc['precision', model_name] = round(precision_score(self.instance_labels, predictions), 3)
            self.metrics_df.loc['accuracy', model_name] = round(accuracy_score(self.instance_labels, predictions), 3)
            self.metrics_df.loc['f1_score', model_name] = round(f1_score(self.instance_labels, predictions), 3)
            self.metrics_df.loc['cohens_kappa', model_name] = round(cohen_kappa_score(self.instance_labels, predictions), 3)
        else:
            print('The instance metrics of ' + model_name + ' have not been calculated.')

    def calc_bag_level_metrics(self, bag_predictions: np.array, model_name: str):
        # Case 1: the bag predictions are per instance
        if bag_predictions.size == self.instance_labels.size != 0:
            print('Calculate bag metrics of ' + model_name)

            bag_predictions_aggregated = []
            for bag_name in self.bag_names:
                bag_indices = (self.bag_names_per_instance == bag_name)
                bag_instance_predictions = bag_predictions[bag_indices]
                bag_predicted_label = int(np.any(bag_instance_predictions > 0.5)) # positive if one pred positive

                bag_predictions_aggregated.append(bag_predicted_label)
            bag_predictions = np.array(bag_predictions_aggregated)
        # Case 2: The bag predictions are per bag
        elif bag_predictions.size == self.bag_labels.size != 0:
            bag_predictions = np.array(bag_predictions)
        # Case 3: No bag predictions or wrong length
        else:
            print('The bag metrics of ' + model_name + ' have not been calculated.')
            return

        bag_gt = self.bag_labels

        bag_recall = recall_score(bag_gt, bag_predictions)
        bag_precision = precision_score(bag_gt, bag_predictions)
        bag_accuracy = accuracy_score(bag_gt, bag_predictions)
        bag_f1_score = f1_score(bag_gt, bag_predictions)
        bag_cohens_kappa = cohen_kappa_score(bag_gt, bag_predictions)

        self.metrics_df.loc['bag_recall', model_name] = round(bag_recall, 3)
        self.metrics_df.loc['bag_precision', model_name] = round(bag_precision, 3)
        self.metrics_df.loc['bag_accuracy', model_name] = round(bag_accuracy, 3)
        self.metrics_df.loc['bag_f1_score', model_name] = round(bag_f1_score, 3)
        self.metrics_df.loc['bag_cohens_kappa', model_name] = round(bag_cohens_kappa, 3)

    def write_to_file(self, config):
        out_name = os.path.basename(config['path_test_df']).split('.')[0]
        out_file = os.path.join(config['output_path'], 'metrics_' + out_name + '.csv')
        os.makedirs(config['output_path'], exist_ok=True)

        print('Save output to ' + out_file)
        self.metrics_df.to_csv(out_file)


