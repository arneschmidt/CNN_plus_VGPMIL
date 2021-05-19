import yaml
import pandas as pd
import numpy as np


def main():
    with open('config.yaml') as file:
        config = yaml.full_load(file)
    train_df = pd.read_csv(config['path_train_df'])
    test_df = pd.read_csv(config['path_test_df'])
    dict = {'Train': train_df, 'Test': test_df}
    for name, df in dict.items():
        print(name)
        if config['col_instance_label'] in df:
            print('Instance stats:')
            print('total: ', len(df))
            pos_inst = np.sum(df[config['col_instance_label']] == 1)
            print('positive: ', pos_inst)
            print('negative: ', len(df) - pos_inst)

        print('Bag stats:')
        bags = np.unique(df[config['col_bag_name']])
        print('total: ', bags.size)
        pos_bags = 0

        for bag in bags:
            label = np.unique(df[config['col_bag_label']].loc[df[config['col_bag_name']] == bag])
            if label == 1:
                pos_bags = pos_bags + 1
        print('positive: ', pos_bags)
        print('negative: ', bags.size - pos_bags)
        print('\n')







if __name__ == "__main__":
    main()