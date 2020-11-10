import sys

import json
import pickle
import datetime
import numpy as np
import pandas as pd

from scipy.stats import uniform

from sklearn.svm import SVC
from sklearn.utils import class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop, SGD

# from imblearn.over_sampling import RandomOverSampler

from config import *

# from keras.callbacks import Callback
#
#
# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
#         val_targ = self.model.validation_data[1]
#         _val_f1 = f1_score(val_targ, val_predict)
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print(" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
#         return


params = {
    'DT': {
        'min_samples_split': [0.002, 0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 2, 4, 6, 8, 10, 12, 16],
        'class_weight': ['balanced'],
        'random_state': [0],
    },
    'RF': {
        'n_estimators': [8, 16, 32, 64, 128, 256, 512, 1024],
        'min_samples_split': [2, 0.002, 0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [1, 0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 2, 4, 6, 8, 10, 12, 16],
        'class_weight': [None, 'balanced'],
        'random_state': [0],
    },
    'NN': {
        'hidden_layer_sizes': [(4,), (8,), (16,), (32,), (64,), (64, 16,), (128, 64, 8,)],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha': uniform(0.001, 0.1),
        'learning_rate': ['constant'],
        'learning_rate_init': uniform(0.001, 0.1),
        'max_iter': [10000],
        # 'class_weight': [None, 'balanced'],
        'random_state': [0],
    },
    'SVM': {
        'C': uniform(0.01, 1.0),
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'gamma': uniform(0.01, 0.1),
        'coef0': uniform(0.01, 0.1),
        'class_weight': [None, 'balanced'],
        'max_iter': [1000],
        'random_state': [0],
    },
    'DNN': {
        'activation_0': ['sigmoid', 'tanh', 'relu'],
        'activation_1': ['sigmoid', 'tanh', 'relu'],
        'activation_2': ['sigmoid', 'tanh', 'relu'],
        'dim_1': [1024, 512, 256, 128, 64, 32, 16, 8, 4],
        'dim_2': [1024, 512, 256, 128, 64, 32, 16, 8, 4],
        'dropout_0': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'dropout_1': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'dropout_2': [None, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        'optimizer': ['adam', 'rmsprop', 'sgd'],  # RMSprop(lr=.001, decay=.0001), SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)]
    },
    # 'DNN': {
    #     'init': ['glorot_uniform', 'normal', 'uniform'],
    #     'optimizer': ['rmsprop', 'adam'],
    # }
}


logboard = TensorBoard(log_dir='.logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch')


def build_dnn(dim_0, dim_1, dim_2, activation_0, activation_1, activation_2, dropout_0, dropout_1, dropout_2,
              optimizer, loss):
    model = Sequential()

    model.add(Dense(dim_0, activation=activation_0))
    if dropout_0 is not None:
        model.add(Dropout(dropout_0))

    model.add(Dense(dim_1, activation=activation_1))
    if dropout_1 is not None:
        model.add(Dropout(dropout_1))

    model.add(Dense(dim_2, activation=activation_2))
    if dropout_2 is not None:
        model.add(Dropout(dropout_2))

    model.add(Dense(1, activation='sigmoid'))

    # metrics = Metrics()
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model


# def build_dnn(dim_0, init, optimizer):
#
#     # create model
#     model = Sequential()
#     model.add(Dense(dim_0, kernel_initializer=init, activation='relu'))
#     model.add(Dense(8, kernel_initializer=init, activation='relu'))
#     model.add(Dense(1, kernel_initializer=init, activation='sigmoid'))
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model


class ParamsEncoder(json.JSONEncoder):
    """ Special json encoder for parameters types """
    def default(self, obj):
        if isinstance(obj, RMSprop):
            return 'RMSprop'
        elif isinstance(obj, SGD):
            return 'SGD'
        return json.JSONEncoder.default(self, obj)


def main():

    n_iter = 100
    dataset = sys.argv[1]
    black_box = sys.argv[2]

    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in black_box_list:
        print('unknown black box %s' % black_box)
        return -1

    print(datetime.datetime.now(), dataset, black_box)

    X_train = pd.read_csv(path_partitions + '%s_X_bb.csv' % dataset, header=None).values
    Y_train = pd.read_csv(path_partitions + '%s_Y_bb.csv' % dataset, header=None).values.ravel()
    X_test = pd.read_csv(path_partitions + '%s_X_cb.csv' % dataset, header=None).values
    Y_test = pd.read_csv(path_partitions + '%s_Y_cb.csv' % dataset, header=None).values.ravel()

    if black_box == 'DT':
        bb = DecisionTreeClassifier()
    elif black_box == 'RF':
        bb = RandomForestClassifier()
    elif black_box == 'SVM':
        bb = SVC(probability=True)
    elif black_box == 'NN':
        bb = MLPClassifier()
    elif black_box == 'DNN':
        def build_fn(dim_1, dim_2, activation_0, activation_1, activation_2, dropout_0, dropout_1, dropout_2,
                     optimizer):
            loss = 'binary_crossentropy' if len(np.unique(Y_train)) == 2 else 'categorical_crossentropy'
            return build_dnn(X_train.shape[1], dim_1, dim_2, activation_0, activation_1, activation_2, dropout_0,
                             dropout_1, dropout_2, optimizer, loss)
        # def build_fn(init, optimizer):
        #     return build_dnn(X_train.shape[1], init, optimizer)

        bb = KerasClassifier(build_fn=build_fn, epochs=1000, verbose=0)

    else:
        print('unknown black box %s' % black_box)
        raise Exception

    if black_box != 'DNN':
        n_jobs = 5
        rs = RandomizedSearchCV(bb, param_distributions=params[black_box], n_iter=n_iter, cv=5, scoring='f1_macro',
                                iid=False, n_jobs=n_jobs, verbose=1)
        rs.fit(X_train, Y_train)
        bb = rs.best_estimator_
    else:
        def build_fn():
            dim_0 = X_train.shape[1]
            loss = 'binary_crossentropy' if len(np.unique(Y_train)) == 2 else 'categorical_crossentropy'
            return build_dnn(dim_0=dim_0, dim_1=512, dim_2=128,
                             activation_0='relu', activation_1='relu', activation_2='relu',
                             dropout_0=0.5, dropout_1=0.1, dropout_2=0.01, optimizer='adam', loss=loss)

        cw = class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

        bb = KerasClassifier(build_fn=build_fn, epochs=1000, verbose=1)
        # ros = RandomOverSampler(random_state=random_state)
        # X_train, Y_train = ros.fit_sample(X_train, Y_train)
        bb.fit(X_train, Y_train, class_weight=cw)

    if black_box == 'DNN':
        bb.model.save(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        pickle_file = open(path_models + '%s_%s.pickle' % (dataset, black_box), 'wb')
        pickle.dump(bb, pickle_file)
        pickle_file.close()

    # params_file = open(path_models + 'params/%s_%s.json' % (dataset, black_box), 'w')
    # json.dump(rs.best_params_, params_file, cls=ParamsEncoder)
    # params_file.close()

    Y_pred_train = bb.predict(X_train).ravel()
    Y_pred_test = bb.predict(X_test).ravel()

    classification = {
        'dataset': dataset,
        'black_box': black_box,
        'train': {
            'accuracy': accuracy_score(Y_train, Y_pred_train),
            'report': classification_report(Y_train, Y_pred_train, output_dict=True)
        },
        'test': {
            'accuracy': accuracy_score(Y_test, Y_pred_test),
            'report': classification_report(Y_test, Y_pred_test, output_dict=True)
        }
    }

    print(datetime.datetime.now(), 'accuracy', classification['train']['accuracy'])
    print(datetime.datetime.now(), 'accuracy', classification['test']['accuracy'])

    classification_file = open(path_results + '%s_%s.json' % (dataset, black_box), 'w')
    json.dump(classification, classification_file)
    classification_file.close()


if __name__ == "__main__":
    main()
