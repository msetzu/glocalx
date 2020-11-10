import json
from copy import deepcopy

from numpy import argwhere, sign, inf, array
from more_itertools import collapse, flatten
from scipy.spatial.distance import euclidean

from adversary.trepan.trepan import SPLITS


class Rule:
    """ A_ logical rule in CNF form. """

    def __init__(self, premises=None, consequence=None, invert=None, distance=euclidean):
        """ Default rule with the given premises and consequence.
        Args:
            premises (dict): Dictionary {feature -> premise} holding the premise for @feature. Defaults to the empty
                                dictionary.
            consequence (int): Outcome of the rule. Defaults to None.
            invert (function): Function consequence -> consequence to invert the rule's consequence.
                                Defaults to (consequence + 1) % 2
            distance (function): Function Rule x Record -> float to evaluate the centrality of a record
                                w.r.t. the rule. Defaults to Euclidean distance.
        """
        self.features = set(premises.keys())
        self.premises = premises if premises is not None else dict()
        self.consequence = consequence
        self._invert_consequence = invert if invert is not None else lambda c: (c + 1) % 2
        self.distance = distance

    @classmethod
    def fromarrays(cls, features=None, thresholds=None, consequence=None, path=None):
        """
        Arguments:
            features (list): The list of features.
            thresholds (list): The list of thresholds per node in the path.
            consequence (int): Rule consequence.
            path (list): Path followed by the rules.
        """
        if thresholds is None:
            raise ValueError('Missing thresholds')
        if consequence is None:
            raise ValueError('Missing consequence')
        if path is None:
            raise ValueError('Missing path')
        if features is None:
            raise ValueError('Missing features')

        __premises = {}
        __features = features
        __consequence = int(consequence)

        thresholds_ = thresholds[:-1]
        indices_per_feature = {feature: argwhere(__features == feature).flatten()
                               for feature in __features}
        directions_per_feature = {f: [sign(path[k + 1]) for k in indices_per_feature[f] if k < len(path) - 1]
                                  for f in __features[:-1]}

        for feature in __features:
            if len(indices_per_feature[feature]) == 1:
                threshold = thresholds_[indices_per_feature[feature][0]]
                __premises[feature] = (-inf, threshold) if directions_per_feature[feature][0] < 0 else (threshold, inf)
            else:
                lower_bounds_idx = [index for index, direction in zip(indices_per_feature[feature],
                                                                      directions_per_feature[feature])
                                    if direction > 0]
                upper_bounds_idx = [index for index, direction in zip(indices_per_feature[feature],
                                                                      directions_per_feature[feature])
                                    if direction < 0]
                lower_bounds, upper_bounds = (array([thresholds[lower_idx] for lower_idx in lower_bounds_idx]),
                                              array([thresholds[upper_idx] for upper_idx in upper_bounds_idx]))

                if lower_bounds.shape[0] > 0 and upper_bounds.shape[0] > 0:
                    __premises[feature] = (max(lower_bounds), min(upper_bounds))
                elif lower_bounds.shape[0] == 0:
                    __premises[feature] = (-inf, min(upper_bounds))
                elif upper_bounds.shape[0] == 0:
                    __premises[feature] = (max(lower_bounds), +inf)

        return cls(__premises, __consequence)

    def __len__(self):
        return len(self.premises)

    def __repr__(self):
        return str(self.premises) + '\n' + str(self.consequence)

    def __str__(self):
        str_ = '{\n'
        for k in sorted(self.features):
            str_ = str_ + '\t' + str(k) + ': ' + str(self.premises[k]) + '\n'
        str_ += str(self.consequence)
        str_ += '}\n'

        return str_

    def __eq__(self, other):
        """ True iff the @other has the same ranges of this. """
        return self.premises == other.premises and self.consequence == other.consequence

    def __hash__(self):
        return hash(frozenset(self.premises.items())) ^ hash(self.consequence)

    def __contains__(self, item):
        return item in self.premises

    def __getitem__(self, item):
        if item not in self.premises:
            raise KeyError(str(item) + ' not in rule')
        return self.premises[item]

    def __setitem__(self, key, value):
        self.premises[key] = value
        return self

    def __iter__(self):
        for el in self.premises.items():
            yield el

    def __delitem__(self, key):
        del self.premises[key]
        return self

    def __copy__(self):
        cop_rule = Rule(self.premises, self.consequence)
        return cop_rule

    def __deepcopy__(self, memodict=None):
        cop_rule = Rule(deepcopy(self.premises), deepcopy(self.consequence))

        return cop_rule

    def json(self):
        """
        Encode this object in a JSON string. Note that this string is equivalent
        to str(self).
        Returns:
            (dict): The JSON dictionary representation of this object.
        """
        json_dict = deepcopy(self.premises)
        json_dict = {str(k): list(v) for k, v in json_dict.items()}
        json_dict['label'] = int(self.consequence)

        return json_dict

    @classmethod
    def from_json(cls, json_file):
        """
        Read rules from \*json_file.
        Args:
            json_file (str): Path of the json file.
        Returns:
            (list): List of rules in \*json_file.
        """
        with open(json_file, 'r') as log:
            jsonized_rules = json.load(log)
            rules = [{int(k): v for k, v in dic} for dic in jsonized_rules]

        return rules

    def to_dict(self):
        """
        Compute the python dictionary associated with this rule.
        Returns:
            (dict): Python dictionary.
        """
        this_copy = deepcopy(self)
        this_copy.premises = {str(k): list(v) for k, v in this_copy.premises.items()}
        this_copy.premises['consequence'] = self.consequence

        return this_copy.premises

    def __invert__(self):
        """ Negate rule by swapping its consequence. Defaults to (consequence + 1) % 2 if
        @invert was not provided at construction time.

        Returns:
            (Rule): Rule with the same premises and inverted consequence.
        """
        neg_rule = self.__copy__()
        neg_rule.consequence = self._invert_consequence(self.consequence)
        return neg_rule

    def __add__(self, other):
        """Sum to rule according to the quasi-polyhedra union.

        Args:
            other (Rule): The rule to add.

        Returns:
            (Rule): New rule with united premises and same consequence.
                    Throws ValueError when discordant consequences are found.
        """
        if self.consequence != other.consequence:
            raise ValueError('Rules should have the same consequence')

        sum_rule = Rule({}, self.consequence, invert=self._invert_consequence)
        premises_in_common = {feature for feature in self.premises if feature in other.premises}
        premises_exclusive = {feature for feature in self.premises if feature not in other.premises}

        for f in premises_in_common:
            sum_rule[f] = (min(self[f][0], other[f][0]), max(self[f][1], other[f][1]))
        for f in premises_exclusive:
            sum_rule[f] = self[f]
        sum_rule.features = set(sum_rule.premises.keys())

        return sum_rule

    def covers(self, x):
        """Does this rule cover x?

        Args:
            x (array): The record.
        Returns:
            (boolean): True if this rule covers x, False otherwise.
        """
        return all([[(x[feature] >= lower) & (x[feature] < upper)] for feature, (lower, upper) in self])


def trepan_to_rules(trepan):
    """Transform a Trepan instance into a list of `Rule`.
    Args:
        trepan (Trepan): A_ trained Trepan instance.
    Returns:
        (list): List of `Rule`.
    """
    rules = []
    populations = trepan.populations
    types = trepan.types
    for trepan_rule_ in trepan.constraints:
        treepan_rule = list(collapse(trepan_rule_, tuple))
        if len(treepan_rule) == 1:
            continue

        ranges = {}
        outcome = treepan_rule[-1][3]
        features = []
        for premise in treepan_rule:
            feature, foo, delta, outcome = premise
            features.append(feature)

            if foo == SPLITS.ANY:
                continue
            elif foo == SPLITS.EQ:
                ranges[feature] = [(delta, delta)]
            elif foo == SPLITS.NEQ:
                # First NEQ, add every
                if feature not in ranges:
                    ranges[feature] = [(delta_, delta_) for delta_ in set(populations[feature]) - {delta}]
                else:
                    ranges[feature] = [(delta_, delta_) for delta_ in set(populations[feature]) - {delta}]\
                                      + ranges[feature]
            elif foo == SPLITS.LEQ:
                ranges[feature] = [(-inf, delta)]
            elif foo == SPLITS.GT:
                ranges[feature] = [(delta, +inf)]

        # Need a list of rules to accommodate negation,
        # in which case more rules with 'equals' premises
        # are created, one per non-negated value
        rule_ranges = [{}]
        for feature in features:
            if types[feature] != object:
                lower_bounds = list(map(lambda r: r[0], ranges[feature]))
                upper_bounds = list(map(lambda r: r[1], ranges[feature]))
                for rule_range in rule_ranges:
                    rule_range[feature] = (max(lower_bounds), min(upper_bounds))
            else:
                # EQ
                if len(ranges[feature]) == 1:
                    for rule_range in rule_ranges:
                        rule_range[feature] = ranges[feature][0]
                # NEQ: generate one rule for each negation to preserve CNF
                else:
                    f = len(set(ranges[feature]))
                    nr_rules = len(rule_ranges)
                    equality_vals = list(set(ranges[feature]))
                    eq_ranges = list(flatten([[equality_val] * nr_rules for equality_val in set(equality_vals)]))
                    rule_ranges = list(flatten([deepcopy(rule_ranges) for _ in range(f)]))

                    for i, equality_val in enumerate(eq_ranges):
                        rule_ranges[i].update({feature: equality_val})

        local_rules = [Rule(premises={}, consequence=outcome) for _ in rule_ranges]
        for rule, ranges in zip(local_rules, rule_ranges):
            rule.premises = ranges
            rule.features = set(ranges.keys())
        rules += local_rules

    return rules
