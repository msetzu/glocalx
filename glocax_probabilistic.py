import numpy as np

from glocalx import GLocalX


class ProbabilisticCutGLocalX(GLocalX):

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
        n = len(conflicting_group_list)
        if n == 0:
            return conflicting_group

        cut_rules = set()
        default = int(y.mean().round())
        # Set ids to None to measure global fidelity_weight
        fidelities = np.array([self.evaluator.binary_fidelity(rule, x, y, default=default, ids=None)
                               for rule in conflicting_group_list])
        # maximum fidelities across fidelities array *except* at position i
        masked_max_fidelities = np.array([np.delete(fidelities, i).max() for i in range(n)])
        dominance_probabilities = np.random.rand(n)
        # dominance scores as a
        dominance_scores = (fidelities - dominance_probabilities) / (fidelities + masked_max_fidelities)
        dominant_rule = conflicting_group_list[np.argmax(dominance_scores).item(0)]
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
