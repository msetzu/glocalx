import sys
sys.path.append('/home/riccardo/Documenti/PhD/LOREM/code/')

import json
import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from datamanager import prepare_adult_dataset
from datamanager import prepare_german_dataset
from datamanager import prepare_compass_dataset
from datamanager import prepare_churn_dataset
from datamanager import prepare_wine_dataset
from datamanager import prepare_dataset

from config import *


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def main():

    dataset = sys.argv[1]

    # for dataset in dataset_list:

    print(datetime.datetime.now(), dataset)

    if dataset == 'adult':
        df, class_name = prepare_adult_dataset(path_data + 'adult.csv')
    elif dataset == 'german':
        df, class_name = prepare_german_dataset(path_data + 'german_credit.csv')
    elif dataset == 'compas':
        df, class_name = prepare_compass_dataset(path_data + 'compas-scores-two-years.csv', binary=True)
    elif dataset == 'compasm':
        df, class_name = prepare_compass_dataset(path_data + 'compas-scores-two-years.csv', binary=False)
    elif dataset == 'whitewine':
        df, class_name = prepare_wine_dataset(path_data + 'winequality-white.csv')
    elif dataset == 'redwine':
        df, class_name = prepare_wine_dataset(path_data + 'winequality-red.csv')
    elif dataset == 'churn':
        df, class_name = prepare_churn_dataset(path_data + 'churn.csv')
    else:
        print('unknown dataset %s' % dataset)
        raise Exception

    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
        df, class_name)

    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_name].values,
                                                        test_size=test_size, random_state=random_state,
                                                        stratify=df[class_name].values)

    _, K, _, _ = train_test_split(rdf[real_feature_names].values, rdf[class_name].values, test_size=test_size,
                                  random_state=random_state, stratify=rdf[class_name].values)

    pd.DataFrame(X_train).to_csv(path_partitions + '%s_X_bb.csv' % dataset, index=None, header=None)
    pd.DataFrame(Y_train).to_csv(path_partitions + '%s_Y_bb.csv' % dataset, index=None, header=None)
    pd.DataFrame(X_test).to_csv(path_partitions + '%s_X_cb.csv' % dataset, index=None, header=None)
    pd.DataFrame(Y_test).to_csv(path_partitions + '%s_Y_cb.csv' % dataset, index=None, header=None)
    kjson_obj = {
        'class_name': class_name,
        'feature_names': feature_names,
        'class_values': class_values,
        'numeric_columns': numeric_columns,
        'real_feature_names': real_feature_names,
        'features_map': features_map,
        'K': K
    }
    kjson_file = open(path_data + '%s_K_cb.json' % dataset, 'w')
    json.dump(kjson_obj, kjson_file, cls=NumpyEncoder)
    kjson_file.close()


if __name__ == "__main__":
    main()
