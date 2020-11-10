from more_itertools import collapse
from pandas import DataFrame

# Numerical analysis
from numpy import inf, int64, bincount, log2
from numpy import hstack
from scipy.spatial.distance import hamming

# Stats
from numpy.random import sample, choice
from scipy.stats import itemfreq
from scipy.stats import rv_discrete
from sklearn.neighbors import KernelDensity

from enum import Enum


class SPLITS(Enum):
    """Splits at a node."""

    LEQ = 0
    GT = 1
    EQ = 2
    NEQ = 3
    ANY = 4


class TREPAN:
    """Trepan."""
    def __init__(self, oracle):
        """Build a Trepan instance with the given `oracle`.
        Args:
            oracle (Predictor): The oracle used to fit this Trepan instance.
        """
        self.oracle = oracle

    def fit(self, records, max_nodes=10, min_samples=100, size=1000, eps=.1, delta=.75):
        """
            Extract a tree from oracle with the TREEPAN algorithm.

            Arguments:
                records (ndarray): Records to analyze.
                oracle (Estimator): Oracle with 'predict' method.
                max_nodes (int): Maximum tree's nodes. Defaults to 100.
                min_samples (int): Minimum samples on leaves. Defaults to 100.

            Returns:
                (networkx.DirectedGraph): A_ decision tree induced from oracle
                                            with the TREEPAN algorithm.
        """
        nr_nodes = 0
        # constraints = [(None, SPLITS.ANY, None, None)]
        constraints = []
        distributions, populations = self.__compute_distributions(records[list(range(records.shape[1] - 1))])
        types = {t: str(records.dtypes[t]) for t in range(records.shape[1] - 1)}
        self.populations, self.types = populations, types
        records = records.astype(types)

        while nr_nodes <= max_nodes:
            records_ = records.iloc[self.__filter(records, constraints[:1])]
            splits = self.__find_splits(records_, distributions, populations, self.oracle, constraints[:1],
                                        size, min_samples, eps, delta)

            # Internal nodes
            if splits is not None:
                nr_nodes += 2
                parent_constraint = constraints[:1]
                constraints.append(parent_constraint + splits[:1])
                constraints.append(parent_constraint + splits[1:])
            else:
                break

            # Update constraints
            constraints = constraints[1:]

        # Keep only the paths ending in leaves
        constraints = constraints[-max_nodes:] if nr_nodes >= max_nodes else constraints
        constraints = list(map(lambda x: list(collapse(x, tuple)), constraints))
        # constraints = list(filter(lambda c: c[1] != SPLITS.ANY, constraints))
        self.constraints = constraints

        return self

    def __preprocess(self, records):
        """Preprocess the provided records."""
        _, n = records.shape
        records[n - 1] = records[n - 1].astype(int64)

        return records

    def __find_splits(self, records, distributions, populations, oracle, constraints, N, size=10000, eps=.1, delta=.75):
        """
            Find the best split on @records. Perform a first oracle-guided split that locally maximizes
            fidelity, then an m-of-n split that maximizes Gini Index.

            Arguments:
                records (DataFrame): Records to analyze.
                distributions (list): Features distribution.
                populations (list): List of populations. Discrete features
                                        require a sorted list of entries.
                oracle (Estimator): Oracle with 'predict' method.
                constraints (list): Set of constraints to limit the neighborhood.
                tree_paths (list): List of the tree's paths.
                N (int): Number of total instances.
                size (int): The neighborhood size. Defaults to 1000.

            Returns:
                (tuple): A_ tuple (feature, split_type, threshold, outcome) holding the best split.
                            None if no split is found.
        """
        split = None
        best_score = -inf
        m, n = records.shape
        types = {t: records.dtypes[t] for t in range(n)}
        # Local expansion
        neighborhood = self.__generate_neighborhood(types, distributions, populations, oracle, constraints, size, eps, delta)
        expanded_records = records.append(neighborhood, ignore_index=True)
        expanded_records = expanded_records.iloc[self.__filter(expanded_records, constraints)]
        expanded_records = expanded_records.astype(types)
        m_, _ = expanded_records.shape
        if m_ == 0:
            return split

        # Oracle-guided split
        for f, distribution, population in zip(range(n - 1), distributions, populations):
            for threshold in population[:-1]:
                filtered_records = expanded_records.copy()
                if types[f] != object:
                    filtered_records_idx_left = filtered_records[filtered_records[f] <= threshold].index.values
                    filtered_records_idx_right = filtered_records[filtered_records[f] > threshold].index.values
                else:
                    filtered_records_idx_left = filtered_records[filtered_records[f] == threshold].index.values
                    filtered_records_idx_right = filtered_records[filtered_records[f] != threshold].index.values

                # Assign split outcome
                target_left, target_right = self.__compute_target(filtered_records,
                                                             filtered_records_idx_left,
                                                             filtered_records_idx_right)
                tree_predictions = filtered_records[n - 1].copy()
                tree_predictions[filtered_records_idx_left] = target_left
                tree_predictions[filtered_records_idx_right] = target_right
                score = (filtered_records.shape[0] / N) * (1 - hamming(filtered_records[n - 1], tree_predictions))

                # Update best split
                if score > best_score:
                    best_score = score
                    if types[f] != object:
                        split = [(f, SPLITS.LEQ, threshold, None), (f, SPLITS.GT, threshold, None)]
                    else:
                        split = [(f, SPLITS.EQ, threshold, None), (f, SPLITS.NEQ, threshold, None)]

        # Best m-of-n split
        best_gini_score = +inf
        splits = None
        for plus_1_feature, population in zip(range(n - 1), populations):
            # Skip selected feature
            if plus_1_feature == split[0][0]:
                continue

            for plus_1_threshold in population[:-1]:
                if types[plus_1_feature] != object:
                    plus_1_split_left = [(plus_1_feature, SPLITS.LEQ, plus_1_threshold, None)]
                    plus_1_split_right = [(plus_1_feature, SPLITS.GT, plus_1_threshold, None)]
                else:
                    plus_1_split_left = [(plus_1_feature, SPLITS.NEQ, plus_1_threshold, None)]
                    plus_1_split_right = [(plus_1_feature, SPLITS.EQ, plus_1_threshold, None)]

                # Select the best m-of-n for the left constraint, left-left (c_ll) and left-right (c_lr) and
                # for the right constraint, right-left (c_lr) and right-right (c_rr).
                m_of_n_indices = self.__filter(expanded_records,
                                            constraints=constraints + [[split[0], split[1]]] +
                                                         [[plus_1_split_left, plus_1_split_right]],
                                           m_of_n=True)

                for c_ll, c_lr, c_rl, c_rr, indices_c_ll, indices_c_lr, indices_c_rl, indices_c_rr in m_of_n_indices:
                    ll_child_counts = expanded_records.loc[indices_c_ll][n - 1].value_counts()
                    rr_child_counts = expanded_records.loc[indices_c_rr][n - 1].value_counts()

                    gini_score = self.__information_gain(ll_child_counts, rr_child_counts, m_)
                    if gini_score <= best_gini_score:
                        best_gini_score = gini_score
                        left_constraints = list(collapse(c_ll, tuple))
                        right_constraints = list(collapse(c_rr, tuple))

                        filtered_children_records_idx_left = self.__filter(records, constraints + left_constraints)
                        filtered_children_records_idx_right = self.__filter(records, constraints + right_constraints)
                        target_left, target_right = self.__compute_target(records,
                                                                     filtered_children_records_idx_left,
                                                                     filtered_children_records_idx_right)
                        left_constraints = list(collapse(left_constraints, tuple))
                        right_constraints = list(collapse(right_constraints, tuple))
                        for i, constraint in enumerate(left_constraints):
                            left_constraints[i] = (constraint[0], constraint[1], constraint[2], target_left)
                        for i, constraint in enumerate(right_constraints):
                            right_constraints[i] = (constraint[0], constraint[1], constraint[2], target_right)

                        splits = [left_constraints, right_constraints]

        return splits

    def __information_gain(self, left_child_counts, right_child_counts, n):
        """Compute information index for @left_child_counts and @right_child_counts"""
        left_child_purity = left_child_counts.min() / left_child_counts.max()
        right_child_purity = right_child_counts.min() / right_child_counts.max()
        sum_left, sum_right = left_child_counts.sum(), right_child_counts.sum()

        return - (sum_left / n) * log2(left_child_purity / n) - (sum_right / n) * log2(right_child_purity / n)

    def __compute_target(self, filtered_records, filtered_records_idx_left, filtered_records_idx_right):
        """Compute the target for the provided left split with indices @filtered_records_idx_left
            and right split with indices @filtered_records_idx_right.

            Arguments:
                filtered_records (DataFrame): The DataFrame of records.
                filtered_records_idx_left (ndarray): 1-dimensional indices for the left partition.
                filtered_records_idx_right (ndarray): 1-dimensional indices for the right partition.

            Returns:
                (tuple): Left target, right target.
        """
        m_, n = filtered_records.shape
        filtered_records[n - 1] = filtered_records[n - 1].astype(int64)
        target_score_left = bincount(filtered_records.loc[filtered_records_idx_left][n - 1]).max() / m_ \
                            if filtered_records_idx_left.shape[0] > 0 else 0
        target_score_right = bincount(filtered_records.loc[filtered_records_idx_right][n - 1]).max() / m_ \
                            if filtered_records_idx_right.shape[0] > 0 else 0

        target_left = 0
        if target_score_left >= target_score_right and filtered_records_idx_left.shape[0] > 0:
            target_left = bincount(filtered_records.loc[filtered_records_idx_left][n - 1]).argmax()
        elif target_score_left >= target_score_right and filtered_records_idx_left.shape[0] == 0:
            target_left = bincount(filtered_records[n - 1]).argmax() % 2
        elif target_score_left < target_score_right and filtered_records_idx_left.shape[0] > 0:
            target_left = bincount(filtered_records.loc[filtered_records_idx_left][n - 1]).argmin()
        elif target_score_left < target_score_right and filtered_records_idx_left.shape[0] == 0:
            target_left = bincount(filtered_records[n - 1]).argmin() % 2

        target_right = (target_left + 1) % 2

        return target_left, target_right


    def __generate_neighborhood(self, types, distributions, populations, oracle, constraints, size=100, eps=.1, delta=.9):
        """
            Generate a neighborhood for @records.

            Arguments:
                records (DataFrame): Records to analyze.
                distributions (Distribution): Features distribution.
                populations (list): List of populations. Discrete features
                                        require a sorted list of entries.
                oracle (Estimator): Oracle with 'predict' method.
                constraints (list): Set of constraints to limit the neighborhood.
                size (int): The neighborhood size.

            Returns:
                (ndarray): The extended neighborhood.
        """
        generated_neighborhood = DataFrame([], columns=range(len(types) - 1))
        predicted_classes = []
        distribution_balance = delta
        while generated_neighborhood.shape[0] < size and distribution_balance < delta:
            neighborhood = DataFrame(sample((size, len(types))))
            for f, distribution, population in zip(range(len(types)), distributions, populations):
                if types[f] != object:
                    neighborhood[f] = distribution.sample(size).squeeze()
                else:
                    neighborhood[f] = choice(a=population, p=distribution, size=size)

            # Apply constraints
            neighborhood = DataFrame(neighborhood).astype(types)
            neighborhood = neighborhood.iloc[self.__filter(neighborhood, constraints), :-1]
            if neighborhood.shape[0] == 0:
                return DataFrame([], columns=range(len(types) - 1))
            oracle_predictions = oracle.predict(neighborhood.values)
            neighborhood = hstack((neighborhood, oracle_predictions.reshape(-1, 1)))
            neighborhood = DataFrame(neighborhood).astype(types)
            generated_neighborhood = generated_neighborhood.append(neighborhood, ignore_index=True)
            predicted_classes += oracle_predictions.tolist()

            distribution_probabilities = bincount(predicted_classes) / len(predicted_classes)
            distribution = rv_discrete(values=([0, 1], distribution_probabilities))
            distribution_balance = 2 * abs(distribution.cdf[0] - .5)

        return generated_neighborhood

    def __filter(self, records, constraints, m_of_n=False):
        """Filter @records according to @constraints.

            Arguments:
                records (DataFrame): Records to analyze.
                constraints (list): Set of constraints to limit the records.
                distributions (list): Features distribution.
                populations (list): List of populations. Discrete features
                                        require a sorted list of entries.
                m_of_n (bool): True if an m-of-n expansion has to be performed,
                                    False otherwise.

            Returns:
                (ndarray): The indices of the filtered records if @m_of_n is False,
                            otherwise a list of ((tuple, tuple, ndarray, ndarray)},
                            holding the left and right constraints and their respective
                            filtered indices.
        """
        records_ = records.copy()

        # Apply constraints
        if not m_of_n:
            for constraint in list(collapse(constraints, base_type=tuple)):
                feature = constraint[0]
                delta = constraint[2]
                if constraint[1] == SPLITS.ANY:
                    continue
                elif constraint[1] == SPLITS.EQ:
                    records_ = records_[records_[feature] == delta]
                elif constraint[1] == SPLITS.NEQ:
                    records_ = records_[records_[feature] != delta]
                elif constraint[1] == SPLITS.GT:
                    records_ = records_[records_[feature] > delta]
                elif constraint[1] == SPLITS.LEQ:
                    records_ = records_[records_[feature] <= delta]
                if records_.shape[0] == 0:
                    break

            return records_.index
        else:
            nr_constraints = len(constraints)
            constraints_left_left = constraints[:-2] + [constraints[-2][0]] + [constraints[-1][0]]
            constraints_left_right = constraints[:-2] + [constraints[-2][0]] + [constraints[-1][1]]
            constraints_right_left = constraints[:-2] + [constraints[-2][1]] + [constraints[-1][0]]
            constraints_right_right = constraints[:-2] + [constraints[-2][1]] + [constraints[-1][1]]

            m_indices = [[i for i in range(nr_constraints) if i != excluded_constraint]
                         for excluded_constraint in range(nr_constraints)] + [list(range(nr_constraints))]

            m_left_left = [[constraints_left_left[i] for i in indices] for indices in m_indices]
            m_left_right = [[constraints_left_right[i] for i in indices] for indices in m_indices]
            m_right_left = [[constraints_right_left[i] for i in indices] for indices in m_indices]
            m_right_right = [[constraints_right_right[i] for i in indices] for indices in m_indices]

            filtered_indices = []
            for c_ll, c_lr, c_rl, c_rr in zip(m_left_left, m_left_right, m_right_left, m_right_right):
                filtered_indices.append((c_ll, c_lr, c_rl, c_rr,
                                         self.__filter(records_, c_ll, False),
                                         self.__filter(records_, c_lr, False),
                                         self.__filter(records_, c_rl, False),
                                         self.__filter(records_, c_rr, False)))

            return filtered_indices


    def _negate(self, constraints):
        """Negate the provided list of constraints."""
        negated_constraints = []
        for constraint in constraints:
            if constraint[2] == SPLITS.EQ:
                sibling_split = SPLITS.NEQ
            elif constraint[2] == SPLITS.NEQ:
                sibling_split = SPLITS.EQ
            elif constraint[2] == SPLITS.LEQ:
                sibling_split = SPLITS.GT
            elif constraint[2] == SPLITS.GT:
                sibling_split = SPLITS.LEQ
            else:
                raise ValueError('Unrecognized split: ' + str(constraint[2]))

            negated_constraints.append((constraint[0], sibling_split, constraint[2], constraint[3]))

        return negated_constraints

    def __compute_distributions(self, records):
        """
            Compute statistical distributions for each feature in @records.

            Arguments:
                records (DataFrame): Records to analyze.

            Returns:
                (tuple): A_ tuple (distributions, populations).
                            Note that continuous distributions have a
                            None population.
        """
        distributions = []
        populations = []
        m, n = records.shape
        types = {t: records.dtypes[t] for t in range(records.shape[1])}
        for f in range(records.shape[1] - 1):
            if types[f] != object:
                kd_estimator = KernelDensity(bandwidth=0.04, kernel='gaussian', algorithm='ball_tree')
                kd_estimator.fit(records[f].values.reshape(-1, 1))
                distributions.append(kd_estimator)
            else:
                distributions.append((itemfreq(records[f])[:, 1] / m).tolist())
            # Populate both discrete and continuous variables
            populations.append(sorted(set(records[f])))

        return distributions, populations
