import datetime
import json
import os
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from scipy.stats import uniform
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier


hyperparams = {
    'RF': {
        'n_estimators': [8, 16, 32, 64, 100, 128, 200, 256, 512, 1024],
        'min_samples_split': [100, 10, 2, 0.002, 0.01, 0.05, 0.1, 0.2],
        'min_samples_leaf': [100, 50, 10, 2, 1, 0.001, 0.01, 0.05, 0.1, 0.2],
        'max_depth': [None, 2, 4, 6, 8, 10, 12, 16, 32, 64, 128, 256],
        'class_weight': [None, 'balanced'],
        'random_state': [0],
    },
    'SVM': {
        'C': uniform(0.0, 1.0),
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
        'gamma': uniform(0.01, 0.1),
        'coef0': uniform(0.01, 0.1),
        'class_weight': [None, 'balanced'],
        'max_iter': [1000],
        'random_state': [0],
    },
    'DNN': {
        'activation_0': ['relu'],
        'activation_1': ['relu'],
        'activation_2': ['relu'],
        'dim_0': [1024, 512, 256, 128],
        'dim_1': [128, 64],
        'dim_2': [128, 64],  # [512, 256, 128, 64, 32, 16, 8, 4],
        'dropout_0': [None],  # , 0.25, 0.1, 0.05],
        'dropout_1': [None],  # , 0.25, 0.1, 0.05],
        'dropout_2': [None],  # 0.25, 0.1, 0.05],
        'optimizer': ['adam']
    }
}


def dnn(dim_0, dim_1, dim_2, activation_0, activation_1, activation_2, dropout_0, dropout_1, dropout_2, optimizer):
    model = Sequential()
    model.add(Dense(dim_0, activation=activation_0))
    if dropout_0 is not None:
        model.add(Dropout(dropout_0))

    model.add(Dense(dim_1, activation=activation_1))
    model.add(Dense(dim_1, activation=activation_1))
    if dropout_1 is not None:
        model.add(Dropout(dropout_1))

    model.add(Dense(dim_2, activation=activation_2))
    model.add(Dense(dim_2, activation=activation_2))
    if dropout_2 is not None:
        model.add(Dropout(dropout_2))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

    return model


def train_dnn(tr, vl, params=None, cv=3, n_iter=128, name=None):
    """
    Train the black box on the given training and validation sets.
    Args:
        tr (ndarray): Training set.
        vl (ndarray): Validation set.
        params (dict): Hyperparameter space.
        cv (int): CV folds.
        n_iter (int): Number of hyperparameter samples.
        name (str): Output file for the model. Defaults to current time if none.
    Returns:
        (KerasClassifier): Trained model.
    """
    output_file = name if name is not None else str(datetime.datetime.now()) + '.h5'
    x, y = tr[:, :-1], tr[:, -1].astype(int)
    x_vl = vl[:, :-1]
    model = KerasClassifier(build_fn=dnn, epochs=1000, batch_size=128, verbose=3)

    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=cv, n_jobs=13, verbose=3)
    random_search.fit(x, y.astype(int))
    model = random_search.best_estimator_

    # Validation
    y_tr = model.predict(x).round().astype(int)
    y_vl = model.predict(x_vl).round().astype(int)
    report = {
        'dataset': name,
        'black_box': 'dnn',
        'train': {
            'accuracy': accuracy_score(y, y_tr),
            'report': classification_report(y, y_tr, output_dict=True)
        },
        'test': {
            'accuracy': accuracy_score(vl[:, -1].astype(int), y_vl),
            'report': classification_report(vl[:, -1].astype(int), y_vl, output_dict=True)
        }
    }

    # Store model
    model.model.save(output_file + '.h5')
    # Store report
    with open(name + '.report.json', 'w') as log:
        json.dump(report, log)

    return model


def train_rf(tr, vl, params=None, cv=3, n_iter=128, name=None):
    """
    Train the black box on the given training and validation sets.
    Args:
        tr (ndarray): Training set.
        vl (ndarray): Validation set.
        params (dict): Hyperparameter space.
        cv (int): CV folds.
        n_iter (int): Number of hyperparameter samples.
        name (str): Output file for the model. Defaults to current time if none.
    Returns:
        (RandomForestClassifier): A random forest classifier.
    """
    output_file = name if name is not None else str(datetime.datetime.now()) + '.h5'
    x, y = tr[:, :-1], tr[:, -1].astype(int)
    x_vl = vl[:, :-1]
    model = RandomForestClassifier()

    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=cv, n_jobs=16, verbose=3)
    random_search.fit(x, y)
    model = random_search.best_estimator_

    # Validation
    y_tr = model.predict(x).round().astype(int)
    y_vl = model.predict(x_vl).round().astype(int)
    report = {
        'dataset': name,
        'black_box': 'dnn',
        'train': {
            'accuracy': accuracy_score(y, y_tr),
            'report': classification_report(y, y_tr, output_dict=True)
        },
        'test': {
            'accuracy': accuracy_score(vl[:, -1].astype(int), y_vl),
            'report': classification_report(vl[:, -1].astype(int), y_vl, output_dict=True)
        }
    }

    # Store model
    with open(output_file + '.rf.pickle', 'wb') as log:
        pickle.dump(model, log)
    # Store report
    with open(name + '.rf.report.json', 'w') as log:
        json.dump(report, log)

    return model


def train_svm(tr, vl, params=None, cv=3, n_iter=128, name=None):
    """
    Train the black box on the given training and validation sets.
    Args:
        tr (ndarray): Training set.
        vl (ndarray): Validation set.
        params (dict): Hyperparameter space.
        cv (int): CV folds.
        n_iter (int): Number of hyperparameter samples.
        name (str): Output file for the model. Defaults to current time if none.
    Returns:
        (RandomForestClassifier): A random forest classifier.
    """
    output_file = name if name is not None else str(datetime.datetime.now()) + '.h5'
    x, y = tr[:, :-1], tr[:, -1].astype(int)
    x_vl = vl[:, :-1]
    model = SVC()

    random_search = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=cv, n_jobs=16, verbose=3)
    random_search.fit(x, y.astype(int))
    model = random_search.best_estimator_

    # Validation
    y_tr = model.predict(x).round().astype(int)
    y_vl = model.predict(x_vl).round().astype(int)
    report = {
        'dataset': name,
        'black_box': 'dnn',
        'train': {
            'accuracy': accuracy_score(y, y_tr),
            'report': classification_report(y, y_tr, output_dict=True)
        },
        'test': {
            'accuracy': accuracy_score(vl[:, -1].astype(int), y_vl),
            'report': classification_report(vl[:, -1].astype(int), y_vl, output_dict=True)
        }
    }

    # Store model
    with open(output_file + '.svm.pickle', 'wb') as log:
        pickle.dump(model, log)
    # Store report
    with open(name + '.svm.report.json', 'w') as log:
        json.dump(report, log)

    return model
