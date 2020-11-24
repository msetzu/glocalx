"""Common callbacks."""
import json
import os
import pickle

from logzero import logger

import pandas as pd

__all__ = ['Callback', 'full_cb', 'print_cb', 'final_rule_dump_cb']


class Callback:
    """
    Functional object implementing a callback with the `call` method. In order to
    implement a stopping callback, a non-None value should be returned.
    """

    def __init__(self, callback):
        self.callback = callback

    def __call__(self, glocalx, debug=20, **kwargs):
        return self.callback(glocalx, **kwargs)


# noinspection PyUnusedLocal
def __print_iteration(glocalx, **kwargs):
    """Print iteration."""
    logger.info('\tIteration: {0}'.format(kwargs['iteration']))


# noinspection PyUnusedLocal
def __print_log(glocalx, **kwargs):
    """Print the model's  and winner's fidelity."""
    if not kwargs['merged']:
        return

    i = kwargs['iteration']
    name = glocalx.name
    x, y, default = kwargs['x'], kwargs['y'], kwargs['default']
    boundary_size = kwargs['fine_boundary_size']
    m = kwargs['m']
    model_fidelity = glocalx.evaluator.binary_fidelity_model(list(glocalx.rules(data=None)), x, y, default=default)
    logger.info('\tName {0} | Iter {1}/{4} | Fidelity {2} | Boundary size {3}'.format(name, i, model_fidelity,
                                                                                            boundary_size, m))


def __full_log(glocalx, **kwargs):
    """
    Log the model's  and winner's fidelity in `name`.log.csv
    Args:
        some (SOME): The SOME.
        name (str): The log filename.
        i (int): Current iteration.
        **kwargs(Keyword arguments). Entries should be:
            winner (Tuple): Position of the winner's score.
            winner_score (float): Winner's score.
            x_batch (numpy.array): Current batch.
            y_batch (numpy.array): Current batch labels.
            x_vl (numpy.array): Validation set.
            y_vl (numpy.array): Validation labels.
            majority_label (int): Majority default label.
            batch (iterable): Batch ids.
            ids_vl (iterable): Validation set ids.
    """
    if not kwargs['merged']:
        return

    i = kwargs['iteration']
    name = glocalx.name
    x, y, default = kwargs['x'], kwargs['y'], kwargs['default']
    fine_boundary = kwargs['fine_boundary']
    fine_boundary_size = kwargs['fine_boundary_size']

    # Merge stats
    (winner_i, winner_j), rejections = kwargs['winner'], kwargs['rejections']
    bic_union = kwargs['bic_union']
    bic_merge = kwargs['bic_merge']

    # Rules stats
    nr_rules_union, nr_rules_merge = kwargs['nr_rules_union'], kwargs['nr_rules_merge']
    union_mean_rules_len, union_std_rules_len = kwargs['union_mean_rules_len'], kwargs['union_std_rules_len']
    merge_mean_rules_len, merge_std_rules_len = kwargs['merge_mean_rules_len'], kwargs['merge_std_rules_len']

    # Model stats
    model_fidelity = glocalx.evaluator.binary_fidelity_model(glocalx.rules(data=None), x, y, default=default)
    coverage_pct = (glocalx.evaluator.coverage(fine_boundary, x).sum(axis=0) > 0).sum() / x.shape[0]

    data = [i, winner_i, winner_j, rejections, bic_union, bic_merge,
            union_mean_rules_len, merge_mean_rules_len, union_std_rules_len, merge_std_rules_len,
            fine_boundary_size, nr_rules_union, nr_rules_merge,
            model_fidelity, coverage_pct]
    data = [str(d) for d in data]

    cols = ['i', 'winner_i', 'winner_j', 'rejections', 'bic_union', 'bic_merge',
            'union_mean_rules_len', 'merge_mean_rules_len', 'union_std_rules_len', 'merge_std_rules_len',
            'fine_boundary_size', 'nr_rules_union', 'nr_rules_merge',
            'model_fidelity', 'coverage_pct']
    base_df = pd.DataFrame([data])
    base_df.columns = cols
    if os.path.isfile(name + '.log.csv'):
        df = pd.read_csv(name + '.log.csv')
        df = pd.concat([df, base_df], axis='rows')
    else:
        df = base_df

    df.to_csv(name + '.log.csv', index=False)
    with open(name + '.glocalx.pickle', 'wb') as log:
        pickle.dump(glocalx, log)


def __final_rule_dump_callback(glocalx, **kwargs):
    """
    Log the model's rules in `name`.final-rules.log.json
    Args:
        some (SOME): The SOME.
        name (str): The log filename.
        i (int): Current iteration.
        **kwargs(Keyword arguments). Entries should be:
            winner_counts (dict): Winner counts dictionary.
    """
    rules = [r.json() for r in glocalx.rules()]

    with open(kwargs['name'] + '.rules.glocalx.alpha=None.json', 'w') as log:
        json.dump(rules, log)


# Set of default callbacks
iteration_bb = Callback(__print_iteration)
print_cb = Callback(__print_log)
full_cb = Callback(__full_log)
final_rule_dump_cb = Callback(__final_rule_dump_callback)
