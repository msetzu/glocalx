from abc import abstractmethod
from collections import defaultdict
from functools import reduce
from itertools import product
import os

import pickle

# Future warning silencing for train_test_split future warning
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from logzero import logger

from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from evaluators import MemEvaluator
from callbacks import final_rule_dump_cb as final_rule_dump_callback
from models import Rule


class Predictor:
    """Interface to be implemented by black boxes."""
    @abstractmethod
    def predict(self, x):
        """
        Predict instance(s) `x`

        Args:
            x (np.array): The instance(s) to predict
        Returns:
            numpy.array: Array of predictions
        """
        pass


def shut_up_tensorflow():
    """Silences tensorflow warnings."""
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(3)


class GLocalX:
    """
    GLocalX instance. Aggregates local explanations into global ones.

    Attributes:
        oracle (Predictor): The black box to explain
        evaluator (MemEvaluator): Evaluator used to evaluate merges and distances
        fine_boundary (set): Explanation boundary

    """

    oracle: Predictor
    evaluator: MemEvaluator
    fine_boundary: set

    def __init__(self, oracle=None):
        """"""
        self.oracle = oracle
        self.evaluator = MemEvaluator(oracle=self.oracle)

    @staticmethod
    def batch(y, sample_size=128):
        """
        Sample `sample_size` objects from `x`.
        Args:
            y (np.array): Labels.
            sample_size (int): Number of samples.
        Returns:
            numpy.np.array: Indices of the sampled data.
        """
        idx_train, *rest = train_test_split(range(y.size), shuffle=True, stratify=y, train_size=sample_size)

        return idx_train

    def partition(self, A, B, record=None, intersecting='coverage'):
        """
        Find the conflicting, non-conflicting and disjoint groups between ruleset `A` and `B`.
        Args:
            A (list): List of rules.
            B (list): List of rules.
            record (int): Id of the record, if not None.
            intersecting (str): If 'coverage', rules are going to overlap if they cover at least
                                one record in common.
                                If 'polyhedra', rules are going to overlap if their premises do.
        Returns:
            tuple: Conflicting groups, non-conflicting groups, disjoint groups.
        """
        conflicting_groups = list()
        non_conflicting_groups = list()
        disjoint_A, disjoint_B = {a for a in A}, {b for b in B}
        for i, a in enumerate(A):
            coverage_a = self.evaluator.coverages[a] if record is None\
                                                    else self.evaluator.coverages[a][record]
            conflicting_a = set()
            non_conflicting_a = set()

            for j, b in enumerate(B):
                coverage_b = self.evaluator.coverages[b] if record is None\
                                                        else self.evaluator.coverages[b][record]

                if (a, b) in self.evaluator.intersecting:
                    a_intersecting_b = self.evaluator.intersecting[(a, b)]
                elif (b, a) in self.evaluator.intersecting:
                    a_intersecting_b = self.evaluator.intersecting[(b, a)]
                elif not ((b, a) in self.evaluator.intersecting or (a, b) in self.evaluator.intersecting):
                    if intersecting == 'coverage':
                        a_intersecting_b = (np.logical_and(coverage_a, coverage_b)).any()
                    else:
                        a_intersecting_b = a & b
                    self.evaluator.intersecting[(a, b)] = a_intersecting_b
                    self.evaluator.intersecting[(b, a)] = a_intersecting_b
                else:
                    a_intersecting_b = False

                if a_intersecting_b:
                    # Different consequence: conflicting
                    if a.consequence != b.consequence:
                        conflicting_a.add(a)
                        conflicting_a.add(b)
                        disjoint_A = disjoint_A - {a}
                        disjoint_B = disjoint_B - {b}

                    # Same consequence: non-conflicting
                    elif a.consequence == b.consequence:
                        non_conflicting_a.add(a)
                        non_conflicting_a.add(b)
                        disjoint_A = disjoint_A - {a}
                        disjoint_B = disjoint_B - {b}

            conflicting_groups.append(conflicting_a)
            non_conflicting_groups.append(non_conflicting_a)

        disjoint = disjoint_A | disjoint_B

        return conflicting_groups, non_conflicting_groups, disjoint

    def accept_merge(self, union, merge, **kwargs):
        """
        Decide whether to accept or reject the merge `merge`
        Args:
            union (set): The explanations' union
            merge (set): The explanations' merge
            **kwargs: Additional keyword arguments.

        Returns:
            bool: True to accept, False otherwise
        """
        data = kwargs['data']
        fidelity_weight, complexity_weight = kwargs['fidelity_weight'], kwargs['complexity_weight']
        non_merging_boundary = kwargs['non_merging_boundary']

        # BIC computation
        bic_union = self.evaluator.bic(union, data,
                                       fidelity_weight=fidelity_weight, complexity_weight=complexity_weight)
        bic_merge = self.evaluator.bic(merge, data,
                                       fidelity_weight=fidelity_weight, complexity_weight=complexity_weight)
        bic_union_validation, bic_merge_validation = bic_union, bic_merge
        if kwargs['global_direction']:
            union_boundary = set(reduce(lambda b, a: a.union(b), [union] + non_merging_boundary, set()))
            merge_boundary = set(reduce(lambda b, a: a.union(b), [merge] + non_merging_boundary, set()))

            bic_union_global = self.evaluator.bic(union_boundary, data)
            bic_merge_global = self.evaluator.bic(merge_boundary, data)

            bic_union_validation, bic_merge_validation = bic_union_global, bic_merge_global

        return bic_merge_validation <= bic_union_validation

    def _cut(self, conflicting_group, x, y, strict_cut=True):
        """
        Cut the provided `conflicting_groups`. Each conflicting group is
        a list of conflicting rules holding a 'king rule' with dominance
        over the others. Cut is performed between the king rule and every other
        rule in the group. A non-king rule is cut each time is designed as such.
        Arguments:
            conflicting_group (iterable): Set of conflicting groups.
            x (np.array): Data.
            y (np.array): Labels.
            strict_cut (bool): If True, the dominant rule cuts the non-dominant rules on all features.
                                If False, the dominant rule cuts the non-dominant rules only on shared features.
                                Defaults to True.
        Returns:
            List: List of rules with minimized conflict.
        """
        conflicting_group_list = list(conflicting_group)
        if len(conflicting_group_list) == 0:
            return conflicting_group

        cut_rules = set()
        default = int(y.mean().round())
        # Set ids to None to measure global fidelity
        fidelities = np.array([self.evaluator.binary_fidelity(rule, x, y, default=default, ids=None)
                               for rule in conflicting_group_list])
        dominant_rule = conflicting_group_list[np.argmax(fidelities).item(0)]
        cut_rules.add(dominant_rule)

        for rule in conflicting_group - {dominant_rule}:
            dominant_features = dominant_rule.features
            cut_rule = rule - dominant_rule
            if strict_cut:
                for r in cut_rule:
                    for f in dominant_features - r.features:
                        if f in r.features:
                            r.features = r.features - {f}
                            del r[f]
            cut_rules |= cut_rule
        cut_rules.add(dominant_rule)

        return cut_rules

    def _join(self, rules, x, y, strict_join=True):
        """
        Join concordant rules.
        Arguments:
            rules (iterable): List of sets of conflicting groups.
            x (np.array): Data.
            y (np.array): Labels.
            strict_join (bool): If True, join is less stringent: only features on both rules
                                    are merged, others are removed
                                    If False, join is less stringent: features on both rules are merged.
                                    If a feature is present only in one rule, it will be present as-is
                                    in the resulting join.
                                    Defaults to True.
        Returns:
            set: List of rules with minimized conflict.
        """
        # On an empty A_ set or B_ set, return the best rule of the non empty set.
        rules_list = list(rules)
        nr_rules = len(rules_list)
        if nr_rules == 0:
            return rules

        # List of ranges on each feature
        ranges_per_feature = defaultdict(list)
        for rule in rules_list:
            for feature, values in rule:
                ranges_per_feature[feature].append(values)

        default = int(y.mean().round())
        # ids set to None to measure global fidelity_weight
        fidelities = np.array([self.evaluator.binary_fidelity(r, x, y, default=default, ids=None) for r in rules_list])
        best_rule = rules_list[np.argmax(fidelities).item(0)]

        # Features shared by all
        shared_features = {f: ranges_per_feature[f] for f in ranges_per_feature
                           if len(ranges_per_feature[f]) == nr_rules}
        # Features not shared by all and from the best rule
        non_shared_features = {k: v for k, v in best_rule if k not in shared_features}

        premises = {}
        consequence = best_rule.consequence
        for f, values in shared_features.items():
            lower_bound, upper_bound = min([lb for lb, _ in values]), max([ub for _, ub in values])
            premises[f] = (lower_bound, upper_bound)
        # A highly-concordant merge includes non-shared features, hence making the join more stringent
        if not strict_join:
            premises.update(non_shared_features)
        rule = Rule(premises=premises, consequence=consequence, names=rules_list[0].names)

        return {rule}

    def merge(self, A, B, x, y, ids=None, intersecting='coverage', strict_join=True, strict_cut=True):
        """
        Merge the two rulesets.
        Args:
            A (set): Set of rules.
            B (set): Set of rules.
            x (np.array): Data.
            y (np.array): Labels.
            ids (iterable): Ids of the records.
            intersecting (str): If 'coverage', rules are going to overlap if they cover at least
                                one record in common.
                                If 'polyhedra', rules are going to overlap if their premises do.
            strict_join (bool): If True, join is less stringent: only features on both rules
                                    are merged, others are removed
                                    If False, join is less stringent: features on both rules are merged.
                                    If a feature is present only in one rule, it will be present as-is
                                    in the resulting join.
                                    Defaults to True.
            strict_cut (bool): If True, the dominant rule cuts the non-dominant rules on all features.
                                If False, the dominant rule cuts the non-dominant rules only on shared features.
                                Defaults to True.
        Returns:
            set: Set of merged rules.
        """
        AB = set()
        A_, B_ = list(A), list(B)

        # Compute the disjoint group and add it to AB
        _, _, disjoint_group = self.partition(A_, B_, ids)
        for record in ids:
            conflicting_group, non_conflicting_group, _ = self.partition(A_, B_, record, intersecting)
            conflicting_group, non_conflicting_group = conflicting_group[0], non_conflicting_group[0]
            disjoint_group = disjoint_group - conflicting_group - non_conflicting_group

            cut_rules = self._cut(conflicting_group, x, y, strict_cut)
            joined_rules = self._join(non_conflicting_group, x, y, strict_join)
            AB |= joined_rules
            AB |= cut_rules

        AB |= disjoint_group

        return AB

    def fit(self, rules, tr_set, batch_size=128, global_direction=False,
            intersecting='coverage', strict_join=True, strict_cut=True,
            fidelity_weight=1., complexity_weight=1.,
            callbacks=None, callback_step=5,
            name=None, pickle_this=False):
        """
        Train GLocalX on the given `rules`.
        Args:
            rules (list): List of rules.
            tr_set (np.array): Training set (records).
            batch_size (int): Batch size. Defaults to 128.
            global_direction (bool): False to compute the BIC on the data batch,
                                    True to compute it on the whole validation set.
                                    Defaults to False.
            intersecting (str): If 'coverage', rules are going to overlap if they cover at least
                                one record in common.
                                If 'polyhedra', rules are going to overlap if their premises do.
            strict_join (bool): If True, join is less stringent: only features on both rules
                                    are merged, others are removed
                                    If False, join is less stringent: features on both rules are merged.
                                    If a feature is present only in one rule, it will be present as-is
                                    in the resulting join.
                                    Defaults to True.
            strict_cut (bool): If True, the dominant rule cuts the non-dominant rules on all features.
                                If False, the dominant rule cuts the non-dominant rules only on shared features.
                                Defaults to True.
            callbacks (list): List of callbacks to use. Defaults to the empty list.
            fidelity_weight (float): Weight to fidelity_weight (BIC-wise). Defaults to 1 (no weight).
            complexity_weight (float): Weight to complexity_weight (BIC-wise). Defaults to 1 (no weight).
            callback_step (Union(int, float)): Evoke the callbacks every `callback_step` iterations.
                                                Use float in [0, 1] to use percentage or an integer.
                                                Defaults to 5.
            name (str): Name of this run for logging purposes. Defaults to None.
            pickle_this (bool): Whether to dump a pickle for this instance as the training finishes.
                                Defaults to False.
        Returns:
            GLocalX: Returns this trained instance.
        """
        x, y = tr_set[:, :-1], tr_set[:, -1]
        if self.oracle is not None:
            y = self.oracle.predict(tr_set[:, :-1]).round().astype(int).reshape(1, -1)
            tr_set[:, -1] = y

        m = len(rules)
        default = int(y.mean().round())
        input_rules = [{rule} for rule in rules]
        boundary = input_rules
        # The boundary vector holds the current currently available theories
        self.boundary = boundary
        boundary_len = len(self.boundary)
        self.fine_boundary = set(rules)
        full_name = name if name is not None else 'Anonymous run'

        # Merge
        iteration = 0
        merged = False
        rejections_list = list()
        while len(self.boundary) > 2 and (merged or iteration == 0):
            logger.debug(full_name + ' | ************************ Iteration ' + str(iteration) + ' of ' + str(m))
            merged = False

            # Update distance matrix
            candidates_indices = [(i, j) for i, j in product(range(boundary_len), range(boundary_len))
                                  if j > i]
            logger.debug('Computing distances')
            distances = [(i, j, self.evaluator.distance(self.boundary[i], self.boundary[j], x))
                         for i, j in candidates_indices]
            logger.debug(full_name + '|  sorting candidates queue')
            candidates = sorted(distances, key=lambda c: c[2])
            # No available candidates, distances go from 0 to 1
            if len(candidates) == 0 or candidates[0][-1] == 1:
                break

            # Sample a data batch
            batch_ids = GLocalX.batch(y.squeeze(), sample_size=batch_size)
            # Explore candidates
            rejections = 0
            A, B, AB_union, AB_merge = None, None, None, None
            logger.debug(full_name + ' creating fine boundary')
            self.fine_boundary = set(reduce(lambda b, a: a.union(b), self.boundary, set()))
            for candidate_nr, (i, j, distance) in enumerate(candidates):
                A, B = self.boundary[i], self.boundary[j]
                AB_union = A | B
                AB_merge = self.merge(A, B, x, y, ids=batch_ids, intersecting=intersecting,
                                      strict_cut=strict_cut, strict_join=strict_join)
                logger.debug(full_name + ' merged candidate ' + str(rejections))
                # Boundary without the potentially merging theories
                non_merging_boundary = [self.boundary[k] for k in range(boundary_len) if k != i and k != j]

                if self.accept_merge(AB_union, AB_merge, data=tr_set, global_direction=global_direction,
                                     non_merging_boundary=non_merging_boundary, fidelity_weight=fidelity_weight,
                                     complexity_weight=complexity_weight):
                    merged = True
                    rejections_list.append(rejections)
                    logger.debug(full_name + ' Merged candidate ' + str(rejections))
                    # Boundary update: the merging theories are removed and the merged theory is inserted
                    self.boundary = [AB_merge] + non_merging_boundary
                    boundary_len -= 1
                    self.fine_boundary = set(reduce(lambda b, a: a.union(b), self.boundary, set()))

                    break
                else:
                    rejections += 1

            # Callbacks
            if (iteration + 1) % callback_step == 0 and merged and callbacks is not None:
                logger.debug(full_name + ' Callbacks... ')
                nr_rules_union, nr_rules_merge = len(AB_union), len(AB_merge)
                coverage_union = self.evaluator.coverage(AB_union, x, ids=batch_ids)
                coverage_merge = self.evaluator.coverage(AB_merge, x, ids=batch_ids)
                union_mean_rules_len = np.mean([len(r) for r in AB_union])
                union_std_rules_len = np.std([len(r) for r in AB_union])
                merge_mean_rules_len = np.mean([len(r) for r in AB_merge])
                merge_std_rules_len = np.std([len(r) for r in AB_merge])
                self.fine_boundary = set(reduce(lambda b, a: a.union(b), self.boundary, set()))
                fine_boundary_size = len(self.fine_boundary)

                for callback in callbacks:
                    # noinspection PyUnboundLocalVariable
                    callback(self, iteration=iteration, x=x, y=y, default=default,
                             callbacks_step=callback_step,
                             winner=(i, j),
                             nr_rules_union=nr_rules_union, nr_rules_merge=nr_rules_merge,
                             coverage_union=coverage_union, coverage_merge=coverage_merge,
                             fine_boundary=self.fine_boundary, m=m,
                             union_mean_rules_len=union_mean_rules_len, merge_mean_rules_len=merge_mean_rules_len,
                             union_std_rules_len=union_std_rules_len, merge_std_rules_len=merge_std_rules_len,
                             fine_boundary_size=fine_boundary_size, merged=merged,
                             name=full_name,
                             rejections=rejections)

            # Iteration update
            iteration += 1

        # Final rule dump
        logger.debug(full_name + ' Dumping ')
        final_rule_dump_callback(self, merged=False, name=full_name)

        # Pickle this instance
        if pickle_this:
            with open(full_name + '.glocalx.pickle', 'wb') as log:
                pickle.dump(self, log)

        return self

    def rules(self, alpha=0.5, data=None, evaluator=None, is_percentile=False, strategy='fidelity'):
        """
        Return the fine boundary of this instance, filtered by `alpha`.
        Args:
            alpha (Union(float | int)): Pruning factor, set to None for no pruning. Defaults to 0.5.
                                        For pruning use a float in [0, 1]. For np.percentile
                                        pruning use a float in [0, 1] and set `np.percentile` to True.
                                        For a number of rules, use a positive int.
            data (np.array): Data (labels included).
            evaluator (Evaluator): Evaluator to use to prune, if any. None otherwise. Defaults to None.
            is_percentile (bool): Whether alpha is a np.percentile or a fidelity/coverage value.
            strategy (str): Rule selection strategy, if any. Defaults to selecting rules by fidelity
                            (select='fidelity'). Can also use coverage (select='coverage')
        Returns:
            list: Fine boundary after a fit.
        """
        if evaluator is None:
            evaluator_ = self.evaluator
        else:
            evaluator_ = evaluator

        fine_boundary = self.fine_boundary

        if data is None:
            return fine_boundary
        elif alpha is not None and len(fine_boundary) > 0:
            x, y = data[:, :-1], data[:, -1].astype(int)
            default = int(y.mean().round())
            rules_0 = [r for r in fine_boundary if r.consequence == 0]
            rules_1 = [r for r in fine_boundary if r.consequence == 1]
            if strategy == 'fidelity':
                values_0 = [evaluator_.binary_fidelity(rule, x, y, default=default) for rule in rules_0]
                values_1 = [evaluator_.binary_fidelity(rule, x, y, default=default) for rule in rules_1]
            elif strategy == 'coverage':
                values_0 = [evaluator_.coverage(rule, x) for rule in rules_0]
                values_1 = [evaluator_.coverage(rule, x) for rule in rules_1]
            else:
                raise ValueError("Unknown strategy: " + str(strategy) + ". Use either 'fidelity' (default) or"
                                                                        "'coverage'.")
            if is_percentile:
                lower_bound_0 = np.percentile(list(set(values_0)), alpha) if is_percentile else alpha
                lower_bound_1 = np.percentile(list(set(values_1)), alpha) if is_percentile else alpha

                fine_boundary_0 = [rule for rule, val in zip(rules_0, values_0) if val >= lower_bound_0]
                fine_boundary_1 = [rule for rule, val in zip(rules_1, values_1) if val >= lower_bound_1]
            elif isinstance(alpha, int):
                fine_boundary_0 = sorted(zip(rules_0, values_0), key=lambda el: el[1])[-alpha // 2:]
                fine_boundary_1 = sorted(zip(rules_1, values_1), key=lambda el: el[1])[-alpha // 2:]
                fine_boundary_0 = [rule for rule, _ in fine_boundary_0]
                fine_boundary_1 = [rule for rule, _ in fine_boundary_1]
            else:
                fine_boundary_0 = [rule for rule, val in zip(rules_0, values_0) if val >= alpha]
                fine_boundary_1 = [rule for rule, val in zip(rules_1, values_1) if val >= alpha]

            return fine_boundary_0 + fine_boundary_1

        return None
