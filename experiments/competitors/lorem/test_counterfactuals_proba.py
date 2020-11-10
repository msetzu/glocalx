import datetime

from adversary.lorem import LOREM
from datamanager import *
from rule import apply_counterfactual
from util import neuclidean, multilabel2str

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


def main():

    random_state = 0
    path = '/Users/riccardo/Documents/PhD/LOREM/'
    path_data = path + 'dataset/'

    dataset = 'compas'
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

    stratify = None if isinstance(class_name, list) else df[class_name].values
    X_train, X_test, Y_train, Y_test = train_test_split(df[feature_names].values, df[class_name].values, test_size=0.30,
                                                        random_state=random_state, stratify=stratify)

    stratify = None if isinstance(class_name, list) else rdf[class_name].values
    _, K, _, _ = train_test_split(rdf[real_feature_names].values, rdf[class_name].values, test_size=0.30,
                                  random_state=random_state, stratify=stratify)

    X2e = X_test[:10]

    if blackbox in ['rf', 'random_forest']:
        bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif blackbox in ['svm', 'support_vector_machines']:
        bb = SVC(random_state=random_state)
    elif blackbox in ['nn', 'mlp', 'multilayer perceptron']:
        bb = MLPClassifier(random_state=random_state)
    else:
        print('unknown black box %s' % blackbox)
        raise Exception

    bb.fit(X_train, Y_train)

    def bb_predict(X):
        return bb.predict(X)

    def bb_predict_proba(X):
        return bb.predict_proba(X)

    dt_pred = list()
    bb_pred = list()
    nbr_crules = list()
    nbr_good_crules = 0

    explainer = LOREM(K, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map,
                      neigh_type='rndgenp', categorical_use_prob=True, continuous_fun_estimation=True, size=1000,
                      ocr=0.1, multi_label=False, one_vs_rest=False, filter_crules=True,
                      random_state=0, verbose=True, ngen=10, bb_predict_proba=bb_predict_proba)

    for i2e in range(len(X2e)):
        if i2e % 10 == 0:
            print(datetime.datetime.now(), '%.2f' % (i2e / len(X2e) * 100))
        x = X_test[i2e]
        exp = explainer.explain_instance(x, samples=2000, use_weights=True, metric=neuclidean)
        has_good_crules = False

        for delta, crule in zip(exp.deltas, exp.crules):
            xc = apply_counterfactual(x, delta, feature_names, features_map, numeric_columns)
            bb_outcomec = bb_predict(xc.reshape(1, -1))[0]
            bb_outcomec = class_values[bb_outcomec] if isinstance(class_name, str) else multilabel2str(bb_outcomec,
                                                                                                       class_values)
            dt_outcomec = crule.cons
            dt_pred.append(dt_outcomec)
            bb_pred.append(bb_outcomec)
            if dt_outcomec == bb_outcomec and not has_good_crules:
                nbr_good_crules += 1
                has_good_crules = True

        nbr_crules.append(len(exp.deltas))

    print(datetime.datetime.now(), '%.2f' % 100)

    print('')
    print('dataset: %s ' % dataset)
    print('black box: %s ' % blackbox)
    print('c-fielity: %.6f' % accuracy_score(bb_pred, dt_pred))
    print('nbr crules: %.2f pm %.2f' % (np.mean(nbr_crules), np.std(nbr_crules)))
    print('nbr good crules: %.2f' % (nbr_good_crules / len(X2e)))


if __name__ == "__main__":
    main()
