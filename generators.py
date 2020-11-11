"""Generators module. Generate data provided to GLocalX."""
from abc import abstractmethod

from numpy import hstack, unique, array, argwhere, concatenate

# Stats
from numpy.random import choice, random
from sklearn.neighbors import KernelDensity

from evaluators import MemEvaluator

from logzero import logger


class BudgedExhaustedException(Exception):
    """Exception raised when a given budget is exhausted."""

    pass


class Generator:
    """Data generator interface."""

    @abstractmethod
    def generate(self, sample, oracle=None, size=1000, **kwargs):
        """
        Generate `size` samples miming `oracle` on `sample`.
        Args:
            sample (ndarray): Data sample to enrich.
            oracle (Predictor): Oracle box.
            size (int): Number of samples to generate.
            kwargs: Optional additional arguments.
        Returns:
            numpy.ndarray: Generate samples labelled by `oracle`.
                            Label is in last column.
        """
        pass


class TrePanGenerator(Generator):
    """Generator based on TrePan's generation schema."""

    def __init__(self, oracle=None):
        self.oracle = oracle

    def generate(self, sample, oracle=None, size=1000, budget=50000, **kwargs):
        """
        Generate `size` samples miming `oracle` on `sample`.
        Args:
            sample (ndarray): Data sample to enrich.
            oracle (Predictor): Oracle box. If provided and an
                                oracle was given in construction, this
                                `oracle` is used in its place.
            size (int): Number of samples to generate.
            budget (int): Maximum number of generation tries.
            kwargs: Optional additional arguments.
        Returns:
            numpy.ndarray: Generate samples labelled by `oracle`.
                            Label is in last column.
        """
        rules = kwargs['rules']
        types = self.__compute_types(sample)
        distributions, populations = self.__compute_distributions(sample, types)
        generation_oracle = self.oracle if oracle is None else oracle
        balance_ratio = generation_oracle.predict(sample).round().mean()
        generated_sample = self.__generate_sample(sample, distributions, types, populations, generation_oracle,
                                                  generation_budget=budget,
                                                  size=size, rules=rules, output_mean=balance_ratio)

        return generated_sample

    @staticmethod
    def __compute_types(sample):
        """
        Compute t for the given `sample`.
        Args:
            sample (ndarray): Sample.
        Returns:
            numpy.ndarray: Array of t.
        """
        types = list()
        for t in range(sample.shape[1]):
            # Only two chunks for binary variables
            is_binary = unique(sample[:, t]).size == 2 and 0 in sample[:, t] and 1 in sample[:, t]
            if is_binary:
                types.append('binary')
                continue

            types.append('continuous')
        types = array(types)

        return types

    @staticmethod
    def __filter(samples, rules):
        """
        Filter the given `samples` satisfying `rules`.
        Args:
            samples (ndarray): Samples.
            rules (list): List of rules.

        Returns:
            numpy.ndarray: Samples in `samples` satisfying `rules`.
        """
        evaluator = MemEvaluator(oracle=None)
        coverage_matrix = evaluator.coverage(rules, samples)
        filtered_samples = samples[coverage_matrix.sum(axis=0) > 0]

        return filtered_samples

    def __generate_sample(self, records, distributions, types, populations, oracle, size, rules=None,
                          generation_budget=10000, eps=.1, output_mean=0.5):
        """
            Generate a sample for `records`.

            Arguments:
                records (ndarray): Data.
                distributions (list): Features distributions.
                types (iterable): List of types. 'unary
                populations (list): List of populations. Discrete features require a sorted list of entries.
                oracle (Estimator): Oracle with 'predict' method.
                size (int): The neighborhood size.
                rules (list): List of rules to filter the sample.
                generation_budget (int): Maximum generation tries. Defaults to 1000.
                eps (float): Tolerance for sampling. Defaults to 0.1
                output_mean (float): Defaults to 0.5.

            Returns:
                numpy.ndarray: The generated sample labelled by `oracle`.
        """
        # Joint estimator
        jointly_estimated_samples = distributions[-1].sample(10 * size)
        nr_ones, nr_zeros = int(output_mean * size), int(size - output_mean * size)
        oracle_predictions = oracle.predict(jointly_estimated_samples).round()
        ones = argwhere(oracle_predictions == 1)[:nr_ones, 0]
        zeroes = argwhere(oracle_predictions == 0)[:nr_zeros, 0]
        idx = hstack((ones, zeroes))
        samples = jointly_estimated_samples[idx]

        samples_output_mean = oracle_predictions[idx].mean()
        generated_samples = hstack((samples, oracle_predictions[idx].reshape(-1, 1)))
        if output_mean - eps <= samples_output_mean <= output_mean + eps:
            return generated_samples

        # Ensemble of estimators if joint samples failed
        samples_output_mean = output_mean - 2 * eps
        generated_samples = None
        budget = generation_budget
        budget_bucket = budget // 10

        while not (output_mean - eps <= samples_output_mean <= output_mean + eps) and budget > 0:
            if budget % budget_bucket == 0:
                logger.debug('Budget ' + str(budget) + '/' + str(generation_budget) + '...')
            budget -= 1
            # Generate random samples in over
            samples = random((10 * size, len(distributions)))
            for f, distribution, population in zip(range(len(types)), distributions, populations):
                if types[f] == 'continuous':
                    samples[:, f] = distribution.sample(10 * size).squeeze()
                else:
                    samples[:, f] = choice(a=population, p=distribution, size=10 * size)

            # Filter
            samples = self.__filter(samples, rules)
            samples[:, types == 'binary'] = concatenate([records[:, types == 'binary']] * 10, axis=0)
            if len(samples) < size:
                continue

            nr_ones, nr_zeros = int(output_mean * size), int(size - output_mean * size)
            oracle_predictions = oracle.predict(samples).round()
            ones = argwhere(oracle_predictions == 1)[:nr_ones, 0]
            zeroes = argwhere(oracle_predictions == 0)[:nr_zeros, 0]
            idx = hstack((ones, zeroes))
            samples = samples[idx]

            samples_output_mean = oracle_predictions[idx].mean()
            generated_samples = hstack((samples, oracle_predictions[idx].reshape(-1, 1)))

        # Budget exhausted
        if budget <= 0:
            raise BudgedExhaustedException('Budget ' + str(generation_budget) + ' exhausted')

        return generated_samples

    @staticmethod
    def __compute_distributions(records, types):
        """
            Compute statistical distributions for each feature in @records.

            Arguments:
                records (DataFrame): Records to analyze.

            Returns:
                tuple: A_ tuple (distributions, populations).
                        Note that continuous distributions have a None population.
        """
        distributions = []
        populations = []
        for f in range(records.shape[1]):
            if types[f] == 'continuous':
                kd_estimator = KernelDensity(bandwidth=0.04, kernel='gaussian', algorithm='ball_tree')
                kd_estimator.fit(records[:, f].reshape(-1, 1))
                distributions.append(kd_estimator)
            else:
                mean = records[:, f].mean()
                distributions.append(array([1 - mean, mean]))  # probability for 0 and 1
            # Populate both discrete and continuous variables
            populations.append(sorted(set(records[:, f])))

        # Full dataset estimator
        kd_estimator = KernelDensity(bandwidth=0.04, kernel='gaussian', algorithm='ball_tree')
        kd_estimator.fit(records)
        distributions.append(kd_estimator)
        populations.append(None)

        return distributions, populations
