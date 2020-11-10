import os
import pickle
import datetime

from adversary.lorem import LOREM
from datamanager import *
from util import record2str, neuclidean, multilabel2str

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


def main():

    random_state = 0
    path = '/Users/riccardo/Documents/PhD/LOREM/'
    path_data = path + 'dataset/'

    dataset = 'adult'
    blackbox = 'rf'

    if dataset == 'adult':
        df, class_name = prepare_adult_dataset(path_data + 'adult.csv')
    elif dataset == 'german':
        df, class_name = prepare_german_dataset(path_data + 'german_credit.csv')
    elif dataset == 'compas':
        df, class_name = prepare_compass_dataset(path_data + 'compas-scores-two-years.csv', binary=True)
    elif dataset == 'compasm':
        df, class_name = prepare_compass_dataset(path_data + 'compas-scores-two-years.csv', binary=False)
    elif dataset == 'iris':
        df, class_name = prepare_iris_dataset(path_data + 'iris.csv')
    elif dataset == 'winered':
        df, class_name = prepare_wine_dataset(path_data + 'winequality-red.csv')
    elif dataset == 'winewhite':
        df, class_name = prepare_wine_dataset(path_data + 'winequality-white.csv')
    elif dataset == 'yeast':
        df, class_name = prepare_yeast_dataset(path_data + 'yeast.arff')
    elif dataset == 'medical':
        df, class_name = prepare_medical_dataset(path_data + 'medical.arff')
    else:
        print('unknown dataset %s' % dataset)
        raise Exception

    df, feature_names, class_values, numeric_columns, rdf, real_feature_names, features_map = prepare_dataset(
        df, class_name)

    # print(df.head())
    # print(feature_names)
    # print(class_values)
    # print(numeric_columns)
    # print(rdf.head())
    # print(real_feature_names)
    # print(features_map)

    stratify = None if isinstance(class_name, list) else df[class_name].values
    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_name].values, test_size=0.30,
                                                        random_state=random_state, stratify=stratify)

    stratify = None if isinstance(class_name, list) else rdf[class_name].values
    _, K, _, _ = train_test_split(rdf[real_feature_names].values, rdf[class_name].values, test_size=0.30,
                                  random_state=random_state, stratify=stratify)

    if blackbox in ['rf', 'random_forest']:
        if os.path.isfile('adult_rf.pickle'):
            bb = pickle.load(open('adult_rf.pickle', 'rb'))
        else:
            bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
            pickle.dump(bb, open('adult_rf.pickle', 'wb'))

    elif blackbox in ['svm', 'support_vector_machines']:
        bb = SVC(random_state=random_state)
    elif blackbox in ['nn', 'mlp', 'multilayer perceptron']:
        bb = MLPClassifier(random_state=random_state)
    else:
        print('unknown black box %s' % blackbox)
        raise Exception

    def bb_predict(X):
        return bb.predict(X)

    bb.fit(X_train, Y_train)
    Y_pred = bb_predict(X_train)
    print('Performance Train', accuracy_score(Y_train, Y_pred), f1_score(Y_train, Y_pred))
    Y_pred = bb_predict(X_test)
    print('Performance Test', accuracy_score(Y_test, Y_pred), f1_score(Y_test, Y_pred))

    explainer = LOREM(K, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map,
                      neigh_type='rndgen', categorical_use_prob=True,
                      continuous_fun_estimation=True, size=1000, ocr=0.1, multi_label=False, one_vs_rest=False,
                      random_state=0, verbose=False, ngen=10)

    for i2e, x in enumerate(X_test):
        print(datetime.datetime.now(), i2e)

        print('x = %s' % record2str(x, feature_names, numeric_columns))
        print('')

        bb_outcome = bb_predict(x.reshape(1, -1))[0]
        bb_outcome_str = class_values[bb_outcome] if isinstance(class_name, str) else multilabel2str(bb_outcome,
                                                                                                     class_values)

        exp = explainer.explain_instance(x, samples=1000, use_weights=True, metric=neuclidean)

        print('bb(x) = { %s }' % bb_outcome_str)
        print('real(x) = { %s }' % class_values[Y_test[i2e]])

        print('e = {\n\tr = %s\n\tc = %s    \n}' % (exp.rstr(), exp.cstr()))
        print(exp.bb_pred, exp.dt_pred, exp.fidelity)

        print('-------\n')

    # xc = apply_counterfactual(x, exp.deltas[0], feature_names, features_map, numeric_columns)
    # print('xc = %s' % record2str(xc, feature_names, numeric_columns))
    # print('')
    #
    # bb_outcomec = bb_predict(xc.reshape(1, -1))[0]
    # bb_outcomec_str = class_values[bb_outcomec] if isinstance(class_name, str) else multilabel2str(bb_outcomec,
    #                                                                                              class_values)
    # print('bb(xc) = { %s }' % bb_outcomec_str)
    # print('')


if __name__ == "__main__":
    main()
