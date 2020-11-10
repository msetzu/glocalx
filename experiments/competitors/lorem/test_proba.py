from adversary.lorem import LOREM
from datamanager import *
from rule import apply_counterfactual
from util import record2str, neuclidean, multilabel2str

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():

    random_state = 0
    # path = '/home/riccardo/Documenti/PhD/LOREM/'
    path = '/Users/riccardo/Documents/PhD/LOREM/'
    path_data = path + 'dataset/'

    dataset = 'iris'
    blackbox = 'rf'

    if dataset == 'adult':
        df, class_name = prepare_adult_dataset(path_data + 'adult.csv')
    elif dataset == 'german':
        df, class_name = prepare_german_dataset(path_data + 'german_credit.csv')
    elif dataset == 'compas':
        df, class_name = prepare_compass_dataset(path_data + 'compas-scores-two-years.csv', binary=True)
    elif dataset == 'compasm':
        df, class_name = prepare_compass_dataset(path_data + 'compas-scores-two-years.csv', binary=False)
    elif dataset == 'churn':
        df, class_name = prepare_churn_dataset(path_data + 'churn.csv')
    elif dataset == 'bank':
        df, class_name = prepare_bank_dataset(path_data + 'bank.csv')
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
        bb = RandomForestClassifier(n_estimators=100, random_state=random_state)
    elif blackbox in ['svm', 'support_vector_machines']:
        bb = SVC(random_state=random_state)
    elif blackbox in ['nn', 'mlp', 'multilayer perceptron']:
        bb = MLPClassifier(random_state=random_state)
    else:
        print('unknown black box %s' % blackbox)
        raise Exception

    bb.fit(X_train, Y_train)

    i2e = 1
    x = X_test[i2e]

    print('x = %s' % record2str(x, feature_names, numeric_columns))
    print('')

    def bb_predict(X):
        return bb.predict(X)

    def bb_predict_proba(X):
        return bb.predict_proba(X)

    bb_outcome = bb_predict(x.reshape(1, -1))[0]
    bb_outcome_str = class_values[bb_outcome] if isinstance(class_name, str) else multilabel2str(bb_outcome, class_values)
    print('bb(x) = { %s } (%s)' % (bb_outcome_str, bb_predict_proba(x.reshape(1, -1))[0]))
    print('')

    explainer = LOREM(K, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map,
                      neigh_type='random', categorical_use_prob=True,
                      continuous_fun_estimation=False, size=1000, ocr=0.1, multi_label=False, one_vs_rest=False,
                      filter_crules=True, random_state=0, verbose=True, ngen=10, bb_predict_proba=bb_predict_proba)
    exp = explainer.explain_instance(x, samples=2000, use_weights=True, metric=neuclidean)

    print('e = {\n\tr = %s\n\tc = %s    \n}' % (exp.rstr(), exp.cstr()))
    print(exp.bb_pred, exp.dt_pred, exp.fidelity)

    xc = apply_counterfactual(x, exp.deltas[0], feature_names, features_map, numeric_columns)
    print('xc = %s' % record2str(xc, feature_names, numeric_columns))
    print('')

    bb_outcomec = bb_predict(xc.reshape(1, -1))[0]
    bb_outcomec_str = class_values[bb_outcomec] if isinstance(class_name, str) else multilabel2str(bb_outcomec,
                                                                                                   class_values)
    print('bb(xc) = { %s } (%s)' % (bb_outcomec_str, bb_predict_proba(xc.reshape(1, -1))[0]))
    print('')


if __name__ == "__main__":
    main()
