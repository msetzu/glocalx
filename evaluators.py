"""
Evaluation module providing basic metrics to run and analyze GLocalX's results.
Two evaluators are provided, DummyEvaluator, which does not optimize performance,
and MemEvaluator, which stores previously computed measures to speed-up performance.
"""
from abc import abstractmethod

import numpy as np
from scipy.spatial.distance import hamming

from logzero import logger

from models import Rule


def covers(rule, x):
    """Does `rule` cover c?

    Args:
        rule (Rule): The rule.
        x (numpy.np.array): The record.
    Returns:
        bool: True if this rule covers c, False otherwise.
    """
    return all([(x[feature] >= lower) & (x[feature] < upper)] for feature, (lower, upper) in rule)


def binary_fidelity(unit, x, y, evaluator=None, ids=None, default=np.nan):
    """Evaluate the goodness of unit.
    Args:
        unit (Unit): The unit to evaluate.
        x (numpy.array): The data.
        y (numpy.array): The labels.
        evaluator (Evaluator): Optional evaluator to speed-up computation.
        ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        default (int): Default prediction for records not covered by the unit.
    Returns:
          float: The unit's fidelity_weight
    """
    coverage = evaluator.coverage(unit, x, ids=ids).flatten()
    unit_predictions = np.array([unit.consequence
                                 for _ in range(x.shape[0] if ids is None else ids.shape[0])]).flatten()
    unit_predictions[~coverage] = default

    fidelity = 1 - hamming(unit_predictions, y[ids] if ids is not None else y) if len(y) > 0 else 0

    return fidelity


def coverage_size(rule, x):
    """Evaluate the cardinality of the coverage of unit on c.

    Args:
        rule (Rule): The rule.
        x (numpy.array): The validation set.

    Returns:
        (int: Number of records of X covered by rule.
    """
    return coverage_matrix([rule], x).sum().item(0)


def coverage_matrix(rules, patterns, targets=None, ids=None):
    """Compute the coverage of @rules over @patterns.
    Args:
        rules (Union(list, Rule)): List of rules (or single Rule) whose coverage to compute.
        patterns (numpy.array): The validation set.
        targets (numpy.array): The labels, if any. None otherwise. Defaults to None.
        ids (numpy.array): Unique identifiers to tell each element in `x` apart.
    Returns:
        numpy.array: The coverage matrix.
    """
    def premises_from(rule, pure=False):
        if not pure:
            premises = np.logical_and.reduce([[(patterns[:, feature] > lower) & (patterns[:, feature] <= upper)]
                                              for feature, (lower, upper) in rule]).squeeze()
        else:
            premises = np.logical_and.reduce([(patterns[:, feature] > lower) & (patterns[:, feature] <= upper)
                                              & (targets == rule.consequence)
                                              for feature, (lower, upper) in rule]).squeeze()

        premises = np.argwhere(premises).squeeze()

        return premises

    if isinstance(rules, list):
        coverage_matrix_ = np.full((len(rules), len(patterns)), False)
        hit_columns = [premises_from(rule, targets is not None) for rule in rules]

        for k, hits in zip(range(len(patterns)), hit_columns):
            coverage_matrix_[k, hits] = True
    else:
        coverage_matrix_ = np.full((len(patterns)), False)
        hit_columns = [premises_from(rules, targets is not None)]
        coverage_matrix_[hit_columns] = True

    coverage_matrix_ = coverage_matrix_[:, ids] if ids is not None else coverage_matrix_

    return coverage_matrix_


class Evaluator:
    """Evaluator interface. Evaluator objects provide coverage and fidelity_weight utilities."""

    @abstractmethod
    def coverage(self, rules, patterns, target=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (list) or (Rule):
            patterns (numpy.array): The validation set.
            target (numpy.array): The labels, if any. None otherwise. Defaults to None.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            numpy.array: The coverage matrix.
        """
        pass

    @abstractmethod
    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (numpy.array): The validation set.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            int: Number of records of X covered by rule.
        """
        pass

    @abstractmethod
    def binary_fidelity(self, unit, x, y, ids=None, default=np.nan):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
            default (int): Default prediction when no rule covers a record.
        Returns:
              float: The unit's fidelity_weight
        """
        pass

    @abstractmethod
    def binary_fidelity_model(self, units, x, y, k=1, default=None, ids=None):
        """Evaluate the goodness of the `units`.
        Args:
            units (Union(list, set)): The units to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            k (int): Number of rules to use in the Laplacian prediction schema.
            default (int): Default prediction for records not covered by the unit.
            ids (numpy.array): Unique identifiers to tell each element in @c apart.
        Returns:
            float: The units fidelity_weight.
        """
        pass

    @abstractmethod
    def covers(self, rule, x):
        """Does @rule cover c?

        Args:
            rule (Rule): The rule.
            x (numpy.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        pass

    @abstractmethod
    def bic(self, rules, vl, fidelity_weight=1., complexity_weight=1.):
        """
        Compute the Bayesian Information Criterion for the given `rules` set.
        Args:
            rules (set): Ruleset.
            vl (numpy.array): Validation set.
            fidelity_weight (float): Weight to fidelity_weight (BIC-wise).
            complexity_weight (float): Weight to complexity_weight (BIC-wise).
        Returns:
            tuple: Triple (BIC, log likelihood, complexity_weight).
        """
        pass


class DummyEvaluator(Evaluator):
    """Dummy evaluator with no memory: every result is computed at each call!"""

    def bic(self, rules, vl, fidelity_weight=1., complexity_weight=1.):
        """
        Compute the Bayesian Information Criterion for the given `rules` set.
        Args:
            rules (set): Ruleset.
            vl (numpy.array): Validation set.
            fidelity_weight (float): Weight to fidelity_weight (BIC-wise).
            complexity_weight (float): Weight to complexity_weight (BIC-wise).
        Returns:
            tuple: Triple (BIC, log likelihood, complexity_weight).
        """
        x, y = vl[:, :-1], vl[:, -1]
        n = x.shape[0]
        default = round(y.mean() + .5)
        log_likelihood = [binary_fidelity(rule, x, y, default=default, ids=None) for rule in rules]
        log_likelihood = np.mean(log_likelihood)

        model_complexity = len(rules)
        model_bic = - (fidelity_weight * log_likelihood - complexity_weight * model_complexity / n)

        return model_bic, log_likelihood, model_complexity

    def __init__(self, oracle):
        """Constructor."""
        self.oracle = oracle
        self.coverages = dict()
        self.binary_fidelities = dict()
        self.coverage_sizes = dict()

    def covers(self, rule, x):
        """Does `rule` cover `x`?

        Args:
            rule (Rule): The rule.
            x (numpy.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        return covers(rule, x)

    def coverage(self, rules, patterns, target=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (Union(Rule, list): Rule (or list of rules) whose coverage to compute.
            patterns (numpy.array): The validation set.
            target (numpy.array): The labels, if any. None otherwise. Defaults to None.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
            numpy.array: The coverage matrix.
        """
        rules_ = rules if isinstance(rules, list) else [rules]
        coverage_ = coverage_matrix(rules_, patterns, target, ids=ids)

        return coverage_

    def coverage_size(self, rule, x, ids=None):
        """Evaluate the cardinality of the coverage of unit on c.

        Args:
            rule (Rule): The rule.
            x (numpy.array): The validation set.
            ids (numpy.array): Unique identifiers to tell each element in `x` apart.
        Returns:
            numpy.array: Number of records of X covered by rule.
        """
        return coverage_size(rule, x)

    def binary_fidelity(self, unit, x, y, default=np.nan, ids=None):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
            default (int): Default prediction when no rule covers a record.
        Returns:
              (float): The unit's fidelity_weight
        """
        if self.oracle is not None or y is None:
            y = self.oracle.predict(x).round().squeeze()

        return binary_fidelity(unit, x, y, self, default=default, ids=ids)

    def binary_fidelity_model(self, units, x, y, k=1, default=None, ids=None):
        """Evaluate the goodness of unit.
        Args:
            units (Union(list, set): The units to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            k (int): Number of rules to use in the Laplacian prediction schema.
            default (int): Default prediction for records not covered by the unit.
            ids (numpy.array): Unique identifiers to tell each element in @patterns apart.
        Returns:
              numpy.array: The units fidelity_weight.
        """
        if self.oracle is not None:
            y = (self.oracle.predict(x).round().squeeze())

        scores = np.array([self.binary_fidelity(rule, x, y, default=default) for rule in units])
        coverage = self.coverage(units, x, y)

        predictions = []
        for record in range(len(x)):
            companions = scores[coverage[:, record]]
            companion_units = units[coverage[:, record]]
            top_companions = np.argsort(companions)[-k:]
            top_units = companion_units[top_companions]
            top_fidelities = companions[top_companions]
            top_fidelities_0 = [top_fidelity for top_fidelity, top_unit in zip(top_fidelities, top_units)
                                if top_unit.consequence == 0]
            top_fidelities_1 = [top_fidelity for top_fidelity, top_unit in zip(top_fidelities, top_units)
                                if top_unit.consequence == 1]

            if len(top_fidelities_0) == 0 and len(top_fidelities_1) > 0:
                prediction = 1
            elif len(top_fidelities_1) == 0 and len(top_fidelities_0) > 0:
                prediction = 0
            elif len(top_fidelities_1) == 0 and len(top_fidelities_0) == 0:
                prediction = default
            else:
                prediction = 0 if np.mean(top_fidelities_0) > np.mean(top_fidelities_1) else 1

            predictions.append(prediction)
        predictions = np.array(predictions)
        fidelity = 1 - hamming(predictions, y) if len(y) > 0 else 0

        return fidelity


class MemEvaluator(Evaluator):
    """Memoization-aware Evaluator to avoid evaluating the same measures over the same data."""

    def __init__(self, oracle):
        """Constructor."""
        self.oracle = oracle
        self.coverages = dict()
        self.perfect_coverages = dict()
        self.intersecting = dict()
        self.bics = dict()
        self.distances = dict()
        self.binary_fidelities = dict()
        self.coverage_sizes = dict()
        self.scores = dict()

    @abstractmethod
    def covers(self, rule, x):
        """Does @rule cover c?

        Args:
            rule (Rule): The rule.
            x (numpy.array): The record.
        Returns:
            bool: True if this rule covers c, False otherwise.
        """
        return covers(rule, x)

    def coverage(self, rules, patterns, targets=None, ids=None):
        """Compute the coverage of @rules over @patterns.
        Args:
            rules (Union(Rule, list): Rule (or list of rules) whose coverage to compute.
            patterns (numpy.array): The validation set.
            targets (numpy.array): The labels, if any. None otherwise. Defaults to None.
            ids (numpy.array): IDS of the given `patterns`, used to speed up evaluation.
        Returns:
            numpy.array: The coverage matrix.
        """
        rules_ = [rules] if not isinstance(rules, list) and not isinstance(rules, set) else rules

        if targets is None:
            for rule in rules_:
                if rule not in self.coverages:
                    self.coverages[rule] = coverage_matrix(rule, patterns, targets)
            cov = np.array([self.coverages[rule] for rule in rules_])
        else:
            for rule in rules_:
                if rule not in self.perfect_coverages:
                    self.perfect_coverages[rule] = coverage_matrix(rule, patterns, targets)
            cov = np.array([self.perfect_coverages[rule] for rule in rules_])

        cov = cov[:, ids] if ids is not None else cov

        return cov

    def distance(self, A, B, x, ids=None):
        """
        Compute the distance between ruleset `A` and ruleset `B`.
        Args:
            A (iterable): Ruleset.
            B (iterable): Ruleset.
            x (numpy.array): Data.
            ids (numpy.array): IDS of the given `x`, used to speed up evaluation.
        Returns:
            (float): The Jaccard distance between the two.
        """
        if tuple(A) in self.distances and tuple(B) in self.distances[tuple(A)]:
            diff = self.distances[tuple(A)][tuple(B)]
            return diff
        if tuple(B) in self.distances and tuple(A) in self.distances[tuple(B)]:
            diff = self.distances[tuple(B)][tuple(A)]
            return diff

        # New distance to compute
        coverage_A, coverage_B = self.coverage(A, x, ids=ids).sum(axis=0), self.coverage(B, x, ids=ids).sum(axis=0)
        diff = hamming(coverage_A, coverage_B)
        if tuple(A) in self.distances:
            self.distances[tuple(A)][tuple(B)] = diff
        if tuple(B) in self.distances:
            self.distances[tuple(B)][tuple(A)] = diff

        # First time for A
        if tuple(A) not in self.distances:
            self.distances[tuple(A)] = {tuple(B): diff}
        # First time for B
        if tuple(B) not in self.distances:
            self.distances[tuple(B)] = {tuple(A): diff}

        return diff

    def binary_fidelity(self, unit, x, y, default=np.nan, ids=None):
        """Evaluate the goodness of unit.
        Args:
            unit (Unit): The unit to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            default (int): Default prediction for records not covered by the unit.
            ids (numpy.array): IDS of the given `x`, used to speed up evaluation.
        Returns:
              float: The unit's fidelity_weight
        """
        if y is None:
            y = self.oracle.predict(x).round().squeeze()

        if ids is None:
            self.binary_fidelities[unit] = self.binary_fidelities.get(unit, binary_fidelity(unit, x, y, self,
                                                                                            default=default, ids=None))
            fidelity = self.binary_fidelities[unit]
        else:
            fidelity = binary_fidelity(unit, x, y, self, default=default, ids=ids)

        return fidelity

    def binary_fidelity_model(self, units, x, y, k=1, default=None, ids=None):
        """Evaluate the goodness of the `units`.
        Args:
            units (Union(list, set)): The units to evaluate.
            x (numpy.array): The data.
            y (numpy.array): The labels.
            k (int): Number of rules to use in the Laplacian prediction schema.
            default (int): Default prediction for records not covered by the unit.
            ids (numpy.array): Unique identifiers to tell each element in @c apart.
        Returns:
              float: The units fidelity_weight.
        """
        if y is None:
            y = self.oracle.predict(x).squeeze().round()

        scores = np.array([self.binary_fidelity(rule, x, y, default=default) for rule in units])
        coverage = self.coverage(units, x)

        if len(units) == 0:
            predictions = [default] * y.shape[0]
        else:
            rules_consequences = np.array([r.consequence for r in units])
            # Fast computation for k = 1
            if k == 1:
                weighted_coverage_scores = coverage * scores.reshape(-1, 1)  # Coverage matrix weighted by score
                # Best score per row (i.e., record)
                best_rule_per_record_idx = weighted_coverage_scores.argmax(axis=0).squeeze()
                predictions = rules_consequences[best_rule_per_record_idx]
                # Replace predictions of non-covered records w/ default prediction
                predictions[coverage.sum(axis=0) == 0] = default
            # Iterative computation
            else:
                predictions = []
                for record in range(len(x)):
                    record_coverage = np.argwhere(coverage[:, record]).ravel()
                    if len(record_coverage) == 0:
                        prediction = default
                    else:
                        companions_0 = record_coverage[rules_consequences[record_coverage] == 0]
                        companions_1 = record_coverage[rules_consequences[record_coverage] == 1]
                        scores_0 = scores[companions_0]
                        scores_1 = scores[companions_1]
                        np.argsort_scores_0 = np.flip(np.argsort(scores[companions_0])[-k:])
                        np.argsort_scores_1 = np.flip(np.argsort(scores[companions_1])[-k:])
                        top_scores_0 = scores_0[np.argsort_scores_0]
                        top_scores_1 = scores_1[np.argsort_scores_1]

                        if len(top_scores_0) == 0 and len(top_scores_1) > 0:
                            prediction = 1
                        elif len(top_scores_1) == 0 and len(top_scores_0) > 0:
                            prediction = 0
                        elif len(top_scores_1) == 0 and len(top_scores_0) == 0:
                            prediction = default
                        else:
                            prediction = 0 if np.mean(top_scores_0) > np.mean(top_scores_1) else 1

                    predictions.append(prediction)
                predictions = np.array(predictions)
        fidelity = 1 - hamming(predictions, y) if len(y) > 0 else 0

        return fidelity

    def bic(self, rules, vl, fidelity_weight=1., complexity_weight=1.):
        """
        Compute the Bayesian Information Criterion for the given `rules` set.
        Args:
            rules (set): Ruleset.
            vl (numpy.array): Validation set.
            fidelity_weight (float): Weight to fidelity_weight (BIC-wise).
            complexity_weight (float): Weight to complexity_weight (BIC-wise).
        Returns:
            float: Model BIC 
        """
        if tuple(rules) in self.bics:
            model_bic = self.bics[tuple(rules)]
        else:
            x, y = vl[:, :-1], vl[:, -1]
            n, m = x.shape
            default = int(y.mean().round())
            log_likelihood = self.binary_fidelity_model(rules, x, y, default=default)

            model_complexity = np.mean([len(r) / m for r in rules])
            model_bic = - (fidelity_weight * log_likelihood - complexity_weight * model_complexity / n)

            logger.debug('Log likelihood: ' + str(log_likelihood) + ' | Complexity: ' + str(model_complexity))

            self.bics[tuple(rules)] = model_bic

        return model_bic

    def forget(self, rules, A=None, B=None):
        """
        Remove rules from this Evaluator's memory. Return the updated evaluator.
        Args:
            rules (iterable): Rules to remove.
            A (set): Rules merged.
            B (set): Rules merged.
        Returns:
            MemEvaluator: This evaluator with no memory of `rules`.

        """
        for rule in rules:
            if rule in self.binary_fidelities:
                del self.binary_fidelities[rule]
            if rule in self.coverages:
                del self.coverages[rule]
            if rule in self.coverage_sizes:
                del self.coverage_sizes[rule]
            if rule in self.perfect_coverages:
                del self.perfect_coverages[rule]
            if rule in self.scores:
                del self.scores[rule]

        if A is not None and B is not None:
            # Delete the whole A, as it has been merged and does not exist anymore
            del self.distances[tuple(A)]
            # Delete the whole B, as it has been merged and does not exist anymore
            del self.distances[tuple(B)]
            # Delete every reference to any of them, as they have been merged and do not exist anymore
            for T in self.distances:
                if tuple(A) in self.distances[T]:
                    del self.distances[T][tuple(A)]
                if tuple(B) in self.distances[T]:
                    del self.distances[T][tuple(B)]

        return self


########################
# Framework validation #
########################
def validate(glocalx, oracle, vl, m=None, alpha=0.5, is_percentile=False):
    """Validate the given `glocalx` instance on the given `tr` dataset.
    Arguments:
        glocalx (Union(GLocalX, list)): GLocalX object or list of rules.
        oracle (Predictor): Oracle to validate against.
        vl (numpy.array): Validation set.
        m (int): Initial number of rules, if known, None otherwise. Defaults to None.
        alpha (Union(float, int, iterable)): Pruning hyperparameter, rules with score
                                            less than `alpha` are removed from the ruleset
                                            used to perform the validation. The score can be
                                            - rule fidelity (`alpha` float, `is_percentile`=False)
                                            - rule fidelity percentile (`alpha` float, `is_percentile`=True)
                                            - number of rules (`alpha` integer)
                                            Same applies if an iterable is provided.
                                            Defaults to '0.5'.
        is_percentile (bool): Whether the provided `alpha` is a percentile or not. Defaults to False.
    Returns:
        dict: Dictionary of validation measures.
    """
    def len_reduction(ruleset_a, ruleset_b):
        return ruleset_a / ruleset_b

    def coverage_pct(rules, x):
        coverage = coverage_matrix(rules, x)
        coverage_percentage = (coverage.sum(axis=0) > 0).sum() / x.shape[0]

        return coverage_percentage

    if oracle is None:
        x = vl[:, :-1]
        y = vl[:, -1]
        evaluator = MemEvaluator(oracle=None)
    else:
        evaluator = MemEvaluator(oracle=oracle)
        x = vl[:, :-1]
        y = oracle.predict(x).round().squeeze()
    majority_label = int(y.mean().round())

    if isinstance(alpha, float) or isinstance(alpha, int):
        alphas = [alpha]
    else:
        alphas = alpha

    results = {}
    for alpha in alphas:
        if isinstance(glocalx, list) or isinstance(glocalx, set):
            rules = glocalx
        else:
            if oracle is None:
                evaluator = MemEvaluator(oracle=None)
            rules = glocalx.rules(alpha=alpha, data=np.hstack((x, y.reshape(-1, 1))),
                                  evaluator=evaluator, is_percentile=is_percentile)
        rules = [r for r in rules if len(r) > 0 and isinstance(r, Rule)]

        if len(rules) == 0:
            results[alpha] = {
                'alpha': alpha,
                'fidelity': np.nan,
                'fidelity_weight': np.nan,
                'coverage': np.nan,
                'mean_length': np.nan,
                'std_length': np.nan,
                'rule_reduction': np.nan,
                'len_reduction': np.nan,
                'mean_prediction': np.nan,
                'std_prediction': np.nan,
                'size': 0
            }
            continue

        evaluator = MemEvaluator(oracle=oracle)
        validation = dict()
        validation['alpha'] = alpha
        validation['size'] = len(rules)
        validation['fidelity'] = evaluator.binary_fidelity_model(rules, x=x, y=y, default=majority_label, k=1)
        validation['coverage'] = coverage_pct(rules, x)
        validation['mean_length'] = np.mean([len(r) for r in rules])
        validation['std_length'] = np.std([len(r) for r in rules])
        validation['rule_reduction'] = 1 - len(rules) / m if m is not None else np.nan
        validation['len_reduction'] = len_reduction(validation['mean_length'], m) if m is not None else np.nan

        # Predictions
        validation['mean_prediction'] = np.mean([r.consequence for r in rules])
        validation['std_prediction'] = np.std([r.consequence for r in rules])

        results[alpha] = validation

    return results
