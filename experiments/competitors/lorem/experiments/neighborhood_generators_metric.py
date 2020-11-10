import sys
sys.path.append('/home/riccardo/Documenti/PhD/LOREM/code/')

import json
import gzip
import pickle
import datetime

import numpy as np
import pandas as pd

from keras.models import load_model

from adversary.lorem import LOREM
from config import *

from util import neuclidean, nmeandev
from scipy.spatial.distance import cosine


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

    num_samples = 1000

    dataset = sys.argv[1]
    black_box = sys.argv[2]

    # neigh_type = sys.argv[3]

    neigh_type = 'rndgen'

    for metric_name, metric in zip([neuclidean, nmeandev, cosine], ['neuclidean', 'nmeandev', 'cosine']):

        if dataset not in dataset_list:
            print('unknown dataset %s' % dataset)
            return -1

        if black_box not in black_box_list:
            print('unknown black box %s' % black_box)
            return -1

        if neigh_type not in neigh_list:
            print('unknown neigh type %s' % neigh_type)
            return -1

        print(datetime.datetime.now(), dataset, black_box, neigh_type)

        X_2e = pd.read_csv(path_partitions + '%s_%s_X_2e.csv' % (dataset, black_box), header=None).values
        kjson_file = open(path_partitions + '%s_%s_K_2e.json' % (dataset, black_box), 'r')
        kjson_obj = json.load(kjson_file)
        kjson_file.close()
        class_name = kjson_obj['class_name']
        feature_names = kjson_obj['feature_names']
        class_values = kjson_obj['class_values']
        numeric_columns = kjson_obj['numeric_columns']
        # real_feature_names = kjson_obj['real_feature_names']
        features_map = {int(k): v for k, v in kjson_obj['features_map'].items()}
        K = np.array(kjson_obj['K'])

        if black_box in ['DT', 'RF', 'SVM', 'NN']:
            bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
        elif black_box in ['DNN', 'DNN0']:
            bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
        else:
            print('unknown black box %s' % black_box)
            raise Exception

        def bb_predict(X):
            return bb.predict(X)

        if black_box in ['DNN', 'DNN0']:
            def bb_predict(X):
                return np.round(bb.predict(X).ravel()).astype(int)

        print(datetime.datetime.now(), 'building LOREM explainer')
        explainer = LOREM(K, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map,
                          neigh_type=neigh_type, categorical_use_prob=True, continuous_fun_estimation=True, size=1000,
                          ocr=0.1, multi_label=False, one_vs_rest=False, random_state=random_state, verbose=False,
                          ngen=10, Kc=X_2e, metric=metric)

        filename_neigh = path_neighbors + '%s_%s_%s_%s.json.gz' % (dataset, black_box, neigh_type, metric_name)

        for i2e, x in enumerate(X_2e):
            if i2e % 10 == 0:
                print(datetime.datetime.now(), dataset, black_box, neigh_type, '%.2f' % (100 * i2e / len(X_2e)),
                      '[%s/%s]' % (i2e, len(X_2e)))

            start_time = datetime.datetime.now()
            Z = explainer.neighgen_fn(x, num_samples)
            run_time = (datetime.datetime.now() - start_time).total_seconds()
            jrow = {'i2e': i2e, 'Z': Z, 'time': run_time}

            json_str = ('%s\n' % json.dumps(jrow, cls=NumpyEncoder)).encode('utf-8')

            with gzip.GzipFile(filename_neigh, 'a') as fout:
                fout.write(json_str)


if __name__ == "__main__":
    main()
