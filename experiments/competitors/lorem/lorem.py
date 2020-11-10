import numpy as np

import itertools
from functools import partial

from scipy.spatial.distance import cdist

from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

# from rule import Rule, compact_premises
#
# from explanation import Explanation, MultilabelExplanation
# from decision_tree import learn_local_decision_tree
# from neighgen import RandomGenerator, GeneticGenerator, RandomGeneticGenerator, ClosestInstancesGenerator
# from neighgen import GeneticProbaGenerator, RandomGeneticProbaGenerator
# from rule import get_rule, get_counterfactual_rules
# from util import calculate_feature_values, neuclidean, multilabel2str, multi_dt_predict
from adversary.lorem.decision_tree import learn_local_decision_tree
from adversary.lorem.explanation import Explanation, MultilabelExplanation
from adversary.lorem.neighgen import GeneticGenerator, RandomGeneticGenerator, GeneticProbaGenerator, RandomGeneticProbaGenerator, \
    RandomGenerator, ClosestInstancesGenerator
from adversary.lorem.rule import get_counterfactual_rules, get_rule, compact_premises, Rule
from adversary.lorem.util import multi_dt_predict, multilabel2str, calculate_feature_values, neuclidean

from logzero import logger

def default_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


# LOcal Rule-based Explanation Method
class LOREM(object):

    def __init__(self, K, bb_predict, feature_names, class_name, class_values, numeric_columns, features_map,
                 neigh_type='genetic', categorical_use_prob=True, continuous_fun_estimation=False,
                 size=1000, ocr=0.1, multi_label=False, one_vs_rest=False, filter_crules=True,
                 kernel_width=None, kernel=None, random_state=None, verbose=False, **kwargs):

        self.random_state = random_state
        self.bb_predict = bb_predict
        self.K = K
        self.class_name = class_name
        self.feature_names = feature_names
        self.class_values = class_values
        self.numeric_columns = numeric_columns
        self.features_map = features_map
        self.neigh_type = neigh_type
        self.multi_label = multi_label
        self.one_vs_rest = one_vs_rest
        self.filter_crules = self.bb_predict if filter_crules else None
        self.verbose = verbose

        kernel_width = np.sqrt(len(self.feature_names)) * .75 if kernel_width is None else kernel_width
        self.kernel_width = float(kernel_width)

        kernel = default_kernel if kernel is None else kernel
        self.kernel = partial(kernel, kernel_width=kernel_width)

        np.random.seed(self.random_state)

        self.__init_neighbor_fn(ocr, categorical_use_prob, continuous_fun_estimation, size, kwargs)

    def explain_instance(self, x, samples=1000, use_weights=True, metric='euclidean'):

        if isinstance(samples, int):
            if self.verbose:
                print('generating neighborhood - %s' % self.neigh_type)
            logger.info('LOREM | Generating neigh...')
            Z = self.neighgen_fn(x, samples)
            logger.info('LOREM | Neigh generated...')
        else:
            Z = samples

        Yb = self.bb_predict(Z).round().astype(int)
        if self.multi_label:
            Z = np.array([z for z, y in zip(Z, Yb) if np.sum(y) > 0])
            Yb = self.bb_predict(Z)

        if self.verbose:
            if not self.multi_label:
                neigh_class, neigh_counts = np.unique(Yb, return_counts=True)
                neigh_class_counts = {self.class_values[k]: v for k, v in zip(neigh_class, neigh_counts)}
            else:
                neigh_counts = np.sum(Yb, axis=0)
                neigh_class_counts = {self.class_values[k]: v for k, v in enumerate(neigh_counts)}

            # print('synthetic neighborhood class counts %s' % neigh_class_counts)

        weights = None if not use_weights else self.__calculate_weights__(Z, metric)

        if self.one_vs_rest and self.multi_label:
            exp = self.__explain_tabular_instance_multiple_tree(x, Z, Yb, weights)
        else:  # binary, multiclass, multilabel all together
            logger.info('LOREM | Explaining instance...')
            exp = self.__explain_tabular_instance_single_tree(x, Z, Yb, weights)

        logger.info('LOREM | Instance explained...')
        return exp

    def __calculate_weights__(self, Z, metric):
        if np.max(Z) != 1 and np.min(Z) != 0:
            Zn = (Z - np.min(Z)) / (np.max(Z) - np.min(Z))
            distances = cdist(Zn, Zn[0].reshape(1, -1), metric=metric).ravel()
        else:
            distances = cdist(Z, Z[0].reshape(1, -1), metric=metric).ravel()
        weights = self.kernel(distances)
        return weights

    def __explain_tabular_instance_single_tree(self, x, Z, Yb, weights):

        if self.verbose:
            print('learning local decision tree')

        logger.info('LOREM | Learning tree...')
        dt = learn_local_decision_tree(Z, Yb, weights, self.class_values, self.multi_label, self.one_vs_rest,
                                       prune_tree=True)
        Yc = dt.predict(Z).round().astype(int)

        if self.verbose:
            print('retrieving explanation')

        rule = get_rule(x, dt, self.feature_names, self.class_name, self.class_values, self.numeric_columns,
                        self.multi_label)
        # crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, self.feature_names, self.class_name,
        #                                           self.class_values, self.numeric_columns, self.features_map,
        #                                           self.filter_crules, self.multi_label)

        exp = Explanation()
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        # exp.crules = crules
        # exp.deltas = deltas
        exp.dt = dt
        # exp.fidelity = fidelity

        return exp

    def __explain_tabular_instance_multiple_tree(self, x, Z, Yb, weights):

        dt_list = list()
        premises = list()
        rule_list = list()
        crules_list = list()
        deltas_list = list()
        nbr_labels = len(self.class_name)

        if self.verbose:
            print('learning %s local decision trees' % nbr_labels)

        for l in range(nbr_labels):
            if np.sum(Yb[:, l]) == 0 or np.sum(Yb[:, l]) == len(Yb):
                outcome = 0 if np.sum(Yb[:, l]) == 0 else 1
                rule = Rule([], outcome, [0, 1])
                crules, deltas = list(), list()
                dt = DummyClassifier()
                dt.fit(np.zeros(Z.shape[1]).reshape(1, -1), np.array([outcome]))
            else:
                dt = learn_local_decision_tree(Z, Yb[:, l], weights, self.class_values, self.multi_label,
                                               self.one_vs_rest, prune_tree=True)
                Yc = dt.predict(Z)
                class_values = [0, 1]
                rule = get_rule(x, dt, self.feature_names, self.class_name[l], class_values, self.numeric_columns,
                                multi_label=False)
                crules, deltas = get_counterfactual_rules(x, Yc[0], dt, Z, Yc, self.feature_names,
                                                          self.class_name[l], class_values, self.numeric_columns,
                                                          self.features_map, self.filter_crules, multi_label=False)

            dt_list.append(dt)
            rule_list.append(rule)
            premises.extend(rule.premises)
            crules_list.append(crules)
            deltas_list.append(deltas)

        if self.verbose:
            print('retrieving explanation')

        Yc = multi_dt_predict(Z, dt_list)
        fidelity = accuracy_score(Yb, Yc, sample_weight=weights)

        premises = compact_premises(premises)
        dt_outcome = multi_dt_predict(x.reshape(1, -1), dt_list)[0]
        cons = multilabel2str(dt_outcome, self.class_values)
        rule = Rule(premises, cons, self.class_name)

        exp = MultilabelExplanation()
        exp.bb_pred = Yb[0]
        exp.dt_pred = Yc[0]
        exp.rule = rule
        exp.crules = list(itertools.chain.from_iterable(crules_list))
        exp.deltas = list(itertools.chain.from_iterable(deltas_list))
        exp.dt = dt_list
        exp.fidelity = fidelity

        exp.rule_list = rule_list
        exp.crules_list = crules_list
        exp.deltas_list = deltas_list

        return exp

    def __init_neighbor_fn(self, ocr, categorical_use_prob, continuous_fun_estimation, size, kwargs):

        neighgen = None
        numeric_columns_index = [i for i, c in enumerate(self.feature_names) if c in self.numeric_columns]

        self.feature_values = None
        if self.neigh_type in ['random', 'genetic', 'rndgen', 'geneticp', 'rndgenp']:
            if self.verbose:
                print('calculating feature values')

            self.feature_values = calculate_feature_values(self.K, numeric_columns_index,
                                                           categorical_use_prob=categorical_use_prob,
                                                           continuous_fun_estimation=continuous_fun_estimation,
                                                           size=size)

        nbr_features = len(self.feature_names)
        nbr_real_features = self.K.shape[1]

        if self.neigh_type in ['genetic', 'rndgen', 'geneticp', 'rndgenp']:
            alpha1 = kwargs.get('alpha1', 0.5)
            alpha2 = kwargs.get('alpha2', 0.5)
            metric = kwargs.get('metric', 'euclidean')
            ngen = kwargs.get('ngen', 10)
            mutpb = kwargs.get('mutpb', 0.5)
            cxpb = kwargs.get('cxpb', 0.7)
            tournsize = kwargs.get('tournsize', 3)
            halloffame_ratio = kwargs.get('halloffame_ratio', 0.1)
            random_seed = self.random_state

            if self.neigh_type == 'genetic':
                neighgen = GeneticGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                            nbr_real_features, numeric_columns_index, ocr=ocr, alpha1=alpha1,
                                            alpha2=alpha2, metric=metric, ngen=ngen,
                                            mutpb=mutpb, cxpb=cxpb, tournsize=tournsize,
                                            halloffame_ratio=halloffame_ratio, random_seed=random_seed,
                                            verbose=self.verbose)
            elif self.neigh_type == 'rndgen':
                neighgen = RandomGeneticGenerator(self.bb_predict, self.feature_values, self.features_map,
                                                  nbr_features, nbr_real_features, numeric_columns_index,
                                                  ocr=ocr, alpha1=alpha1, alpha2=alpha2,
                                                  metric=metric, ngen=ngen, mutpb=mutpb, cxpb=cxpb,
                                                  tournsize=tournsize, halloffame_ratio=halloffame_ratio,
                                                  random_seed=random_seed, verbose=self.verbose)
            elif self.neigh_type == 'geneticp':
                bb_predict_proba = kwargs.get('bb_predict_proba', None)
                neighgen = GeneticProbaGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                                 nbr_real_features, numeric_columns_index, ocr=ocr, alpha1=alpha1,
                                                 alpha2=alpha2, metric=metric, ngen=ngen,
                                                 mutpb=mutpb, cxpb=cxpb, tournsize=tournsize,
                                                 halloffame_ratio=halloffame_ratio,
                                                 bb_predict_proba=bb_predict_proba,
                                                 random_seed=random_seed,
                                                 verbose=self.verbose)

            elif self.neigh_type == 'rndgenp':
                bb_predict_proba = kwargs.get('bb_predict_proba', None)
                neighgen = RandomGeneticProbaGenerator(self.bb_predict, self.feature_values, self.features_map,
                                                       nbr_features, nbr_real_features, numeric_columns_index,
                                                       ocr=ocr, alpha1=alpha1, alpha2=alpha2,
                                                       metric=metric, ngen=ngen, mutpb=mutpb, cxpb=cxpb,
                                                       tournsize=tournsize, halloffame_ratio=halloffame_ratio,
                                                       bb_predict_proba=bb_predict_proba,
                                                       random_seed=random_seed, verbose=self.verbose)

        elif self.neigh_type == 'random':
            neighgen = RandomGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                       nbr_real_features, numeric_columns_index, ocr=ocr)
        elif self.neigh_type == 'closest':
            Kc = kwargs.get('Kc', None)
            k = kwargs.get('k', None)
            type = kwargs.get('core_neigh_type', 'simple')
            alphaf = kwargs.get('alphaf', 0.5)
            alphal = kwargs.get('alphal', 0.5)
            metric_features = kwargs.get('metric_features', neuclidean)
            metric_labels = kwargs.get('metric_labels', neuclidean)
            neighgen = ClosestInstancesGenerator(self.bb_predict, self.feature_values, self.features_map, nbr_features,
                                                 nbr_real_features, numeric_columns_index, ocr=ocr,
                                                 K=Kc, rK=self.K, k=k, core_neigh_type=type, alphaf=alphaf,
                                                 alphal=alphal, metric_features=metric_features,
                                                 metric_labels=metric_labels, categorical_use_prob=categorical_use_prob,
                                                 continuous_fun_estimation=continuous_fun_estimation, size=size,
                                                 verbose=self.verbose)
        else:
            print('unknown neighborhood generator')
            raise Exception

        self.neighgen_fn = neighgen.generate
