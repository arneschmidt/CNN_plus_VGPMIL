import pandas as pd
import numpy as np

train_feat_file = 'May_MIL.csv'
train_labels_file = 'May_MIL_labels2.csv'
test_feat_file = 'TEST_MIL.csv'
test_labels_file = 'TEST_MIL_labels2.csv'

input_dir = '/home/arne/datasets/UGR16anomaly/'
output_dir = '/home/arne/datasets/UGR16anomaly/formatted_vgpmil/'

def format_df(feat_file, label_file):
    feat_df = pd.read_csv(input_dir + feat_file)
    label_df = pd.read_csv(input_dir + label_file)

    n_feat = len(feat_df.columns)

    feat_col_names = []
    for i in range(n_feat):
        feat_col_names.append('feature_' + str(i))
    feat_df.columns = feat_col_names

    label_df.rename(columns={'binary_label': 'instance_label',
                             'hour_code': 'bag_name'}, inplace=True)

    bag_names_per_instance = np.array(label_df['bag_name'])
    bag_names = np.unique(bag_names_per_instance)
    label_df['bag_label'] = 0

    for bag_name in bag_names:
        bag_indices = (bag_names_per_instance == bag_name)
        bag_label = np.max(label_df['instance_label'].loc[bag_indices])
        label_df['bag_label'].loc[bag_indices] = bag_label

    out_df = pd.concat([feat_df, label_df], axis=1)
    out_df.to_csv(output_dir + feat_file)

format_df(train_feat_file, train_labels_file)
format_df(test_feat_file, test_labels_file)
