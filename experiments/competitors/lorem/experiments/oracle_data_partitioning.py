import sys

import json
import pickle
import datetime
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from tensorflow.keras.models import load_model

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
        return json.JSONEncoder.default(self, obj)


def main():

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    #
    # if dataset not in dataset_list:
    #     print('unknown dataset %s' % dataset)
    #     return -1
    #
    # if black_box not in black_box_list:
    #     print('unknown black box %s' % black_box)
    #     return -1

    for dataset in ['adult', 'compas', 'churn', 'german']:
        for black_box in ['DT', 'RF', 'SVM', 'NN', 'DNN']:

            print(datetime.datetime.now(), dataset, black_box)

            X_test = pd.read_csv(path_partitions + '%s_X_cb.csv' % dataset, header=None).values
            Y_test = pd.read_csv(path_partitions + '%s_Y_cb.csv' % dataset, header=None).values.ravel()

            kjson_file = open(path_data + '%s_K_cb.json' % dataset, 'r')
            kjson_obj = json.load(kjson_file)
            kjson_file.close()
            class_name = kjson_obj['class_name']
            feature_names = kjson_obj['feature_names']
            class_values = kjson_obj['class_values']
            numeric_columns = kjson_obj['numeric_columns']
            real_feature_names = kjson_obj['real_feature_names']
            features_map = kjson_obj['features_map']
            K = np.array(kjson_obj['K'])

            if black_box in ['RF', 'SVM', 'NN', 'DT']:
                bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
            elif black_box in ['DNN']:
                bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
            else:
                print('unknown black box %s' % black_box)
                raise Exception

            Y_pred = bb.predict(X_test).ravel()
            if black_box in ['DNN', 'DNN0']:
                Y_pred = np.round(Y_pred).astype(int)

            X_2e, X_ts, Y_2e, Y_ts = train_test_split(X_test, Y_pred, test_size=test_size, random_state=random_state,
                                                      stratify=Y_pred)

            K_2e, _, _, _ = train_test_split(K, Y_pred, test_size=test_size, random_state=random_state, stratify=Y_pred)

            pd.DataFrame(X_2e).to_csv(path_partitions + '%s_%s_X_2e.csv' % (dataset, black_box), index=None, header=None)
            pd.DataFrame(Y_2e).to_csv(path_partitions + '%s_%s_Y_2e.csv' % (dataset, black_box), index=None, header=None)
            pd.DataFrame(X_ts).to_csv(path_partitions + '%s_%s_X_ts.csv' % (dataset, black_box), index=None, header=None)
            pd.DataFrame(Y_ts).to_csv(path_partitions + '%s_%s_Y_ts.csv' % (dataset, black_box), index=None, header=None)
            kjson_obj = {
                'class_name': class_name,
                'feature_names': feature_names,
                'class_values': class_values,
                'numeric_columns': numeric_columns,
                'real_feature_names': real_feature_names,
                'features_map': features_map,
                'K': K_2e
            }
            kjson_file = open(path_partitions + '%s_%s_K_2e.json' % (dataset, black_box), 'w')
            json.dump(kjson_obj, kjson_file, cls=NumpyEncoder)
            kjson_file.close()

            X_2e, X_ts, Y_2e_real, Y_ts_real = train_test_split(X_test, Y_test, test_size=test_size, random_state=random_state,
                                                                stratify=Y_pred)

            pd.DataFrame(Y_2e_real).to_csv(path_data + '%s_Y_2e_real.csv' % dataset, index=None, header=None)
            pd.DataFrame(Y_ts_real).to_csv(path_data + '%s_Y_ts_real.csv' % dataset, index=None, header=None)

            Y_2e_pred = bb.predict(X_2e).ravel()
            if black_box in ['DNN', 'DNN0']:
                Y_2e_pred = np.round(Y_2e_pred).astype(int)

            Y_ts_pred = bb.predict(X_ts).ravel()
            if black_box in ['DNN', 'DNN0']:
                Y_ts_pred = np.round(Y_ts_pred).astype(int)

            classification = {
                'dataset': dataset,
                'black_box': black_box,
                'train': {
                    'accuracy': accuracy_score(Y_2e_real, Y_2e_pred),
                    'report': classification_report(Y_2e_real, Y_2e_pred, output_dict=True)
                },
                'test': {
                    'accuracy': accuracy_score(Y_ts_real, Y_ts_pred),
                    'report': classification_report(Y_ts_real, Y_ts_pred, output_dict=True)
                }
            }

            print(datetime.datetime.now(), 'accuracy', classification['test']['accuracy'])

            classification_file = open(path_results + '%s_%s_2e.json' % (dataset, black_box), 'w')
            json.dump(classification, classification_file)
            classification_file.close()


if __name__ == "__main__":
    main()
