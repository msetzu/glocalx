"""API module for GLocalX."""
import sys
import os
import time
import pickle
import json

from callbacks import print_cb, full_cb, final_rule_dump_cb
from glocalx import GLocalX
from generators import TrePanGenerator, BudgedExhaustedException

# Shut up, tensorflow!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import click
import logzero
import numpy as np
from tensorflow.keras.models import load_model as load_tf_model

from logzero import logger

from models import Rule


@click.command()
# Compulsory arguments: input rules, TR set, TS set, oracle
@click.argument('rules', type=click.Path(exists=True))
@click.argument('tr', type=click.Path(exists=True))
@click.option('-o', '--oracle', type=click.Path(exists=True))
# Input descriptions
@click.option('-cbs', '--callbacks',  default=0.1, help='Callback step, either int or float. Defaults to 0.1')
# Output file
@click.option('-m', '--name',  default=None, help='Name of the log files.')
# Use synthetic data
@click.option('--generate', default=None, help='Number of records to generate, if given. Defaults to None.')
# Rule collision
@click.option('-i', '--intersect', default='coverage', help='Whether to use coverage intersection '
                                                            '(\'coverage\') or polyhedra intersection '
                                                            '(\'polyhedra\'). Defaults to \'coverage\'.')
# Weight the BIC function
@click.option('-f', '--fidelity_weight', default=1., help='Fidelity weight. Defaults to 1.')
@click.option('-c', '--complexity_weight', default=1., help='Complexity weight. Defaults to 1.')
# Running options
@click.option('-a', '--alpha',  default=.5, help='Pruning factor. Defaults to 0.5')
@click.option('-b', '--batch', default=128, help='Batch size. Set to -1 for full batch. Defaults to 128.')
@click.option('-u', '--undersample', default=1., help='Undersample size, to use a percentage of the rules. '
                                                      'Defaults to 1.0 (No undersample).')
# Merge options
@click.option('--strict_join', is_flag=True, help='Use to use high concordance.')
@click.option('--strict_cut', is_flag=True, help='Use to use the strong cut.')
@click.option('--global_direction', is_flag=True, help='Use to use the global search direction.')
# Set debug class
@click.option('-d', '--debug', default=20, help='Debug level.')
def cl_run(rules, tr, generate=None, oracle=None, batch=128, alpha=0.5, undersample=1.0,
           fidelity_weight=1., complexity_weight=1., global_direction=False,
           intersect='coverage', strict_join=False, strict_cut=False,
           callbacks=0.1, name=None, debug=20):
    run(rules, tr=tr, oracle=oracle, generate=generate,
        intersecting=intersect, batch_size=batch, alpha=alpha, undersample=undersample,
        strict_join=strict_join, strict_cut=strict_cut, global_direction=global_direction,
        fidelity_weight=fidelity_weight, complexity_weight=complexity_weight,
        callbacks_step=callbacks,
        name=name, debug=debug)


def run(rules, tr, oracle=None, generate=None,
        intersecting='coverage', batch_size=128, alpha=0.5, undersample=1.,
        strict_join=False, strict_cut=False, global_direction=False,
        fidelity_weight=1, complexity_weight=1,
        name=None, callbacks_step=0.1, debug=20):
    """Run the GLocalX framework on a set of rules.
    Arguments:
        rules (str): JSON file with the train set.
        tr (str): Validation set.
        oracle (str): Path to the oracle to use. None to use the dataset labels.
        generate (Union(int, float, None)): Size of the synthetic dataset to use, if not using the training set.
                                            Use float to give size w.r.t. the TR to use.
                                            Defaults to None (use training set).
        intersecting (str): Whether to use coverage intersection ('coverage') or polyhedra intersection ('polyhedra').
                            Defaults to 'coverage'.
        global_direction (bool): False to compute the BIC on the merged theory, True to compute it on the whole model.
                                Defaults to False.
        batch_size (int): Batch size. Defaults to 128.
        fidelity_weight (float): Fidelity weight. Defaults to 1.
        complexity_weight (float): Complexity weight. Defaults to 1.
        alpha (float): Pruning factor. Defaults to 0.5.
        undersample (float): Percentage of rules to use, if < 1, irrelevant otherwise. Defaults to 1.
        name (str): Name for the output logs.
        callbacks_step (Union(int, float)): Callback step, either int or float (percentage). Defaults to 0.1.
        debug (int): Minimum debug level.
        strict_join (bool): If True, join is less stringent: only features on both rules
                            are merged, others are removed
                            If False, join is less stringent: features on both rules are merged.
                            If a feature is present only in one rule, it will be present as-is
                            in the resulting join.
                            Defaults to True.
        strict_cut (bool): If True, the dominant rule cuts the non-dominant rules on all features.
                            If False, the dominant rule cuts the non-dominant rules only on shared features.
                            Defaults to True.
    """
    # Set-up debug
    if debug == 10:
        min_log = logzero.logging.DEBUG
    elif debug == 20:
        min_log = logzero.logging.INFO
    elif debug == 30:
        min_log = logzero.logging.WARNING
    elif debug == 40:
        min_log = logzero.logging.ERROR
    elif debug == 50:
        min_log = logzero.logging.CRITICAL
    else:
        min_log = 0

    logzero.loglevel(min_log)

    if name is None:
        name = tr + '_' + str(time.time())
    elif os.path.exists(name + '.glocalx.pickle'):
        logger.info('GLocalX run already existing: ' + name + '.glocalx.pickle. Exiting!')
        return

    # Info LOG
    logger.info('Rules: '           + str(rules))
    logger.info('name: '            + str(name))
    logger.info('tr: '              + str(tr))
    logger.info('generate: '        + str(generate))
    logger.info('oracle: '          + str(oracle))
    logger.info('intersect: '       + str(intersecting))
    logger.info('global dir.: '     + str(global_direction))
    logger.info('alpha: '           + str(alpha))
    logger.info('undersample: '     + str(undersample))

    # Set up output
    output_dic = dict()
    output_dic['tr'] = tr
    output_dic['oracle'] = oracle
    output_dic['global search'] = global_direction
    output_dic['alpha'] = alpha
    output_dic['generate'] = generate
    output_dic['undersample'] = undersample

    logger.info('Loading data... ')
    # Load data and header
    data = np.genfromtxt(tr, delimiter=',', names=True)
    names = data.dtype.names
    tr_set = data.view(np.float).reshape(data.shape + (-1,))

    # Run GLocalX
    logger.info('Loading ruleset...')
    rules = Rule.from_json(rules, names=names)
    rules = list(set(rules))
    rules = [r for r in rules if len(r) > 0]

    if undersample < 1:
        n = len(rules)
        sample_indices_space = range(n)
        sample_size = int(undersample * n)
        sample_indices = np.random.choice(sample_indices_space, (sample_size,))
        rules = [rules[i] for i in sample_indices]

    logger.info('Loading oracle...')
    if oracle is not None:
        if oracle.endswith('.h5'):
            oracle = load_tf_model(oracle)
        elif oracle.endswith('.pickle'):
            with open(oracle, 'rb') as log:
                oracle = pickle.load(log)
        else:
            return
        oracle_predictions = oracle.predict(tr_set[1:, :-1]).round().reshape((tr_set.shape[0] - 1, 1))
        tr_set = np.hstack((tr_set[1:, :-1], oracle_predictions))

    # Generate data for GLocalX, if needed
    if generate is not None:
        logger.debug('Generating data...')
        try:
            # int
            if '.' not in generate:
                tr_set = TrePanGenerator(oracle=oracle).generate(sample=tr_set[:, :-1],
                                                                 size=int(generate), rules=rules)
            # float

            else:
                tr_set = TrePanGenerator(oracle=oracle).generate(sample=tr_set[:, :-1],
                                                                 size=int(float(generate) * tr_set.shape[0]),
                                                                 rules=rules)
        except BudgedExhaustedException:
            logger.info('Budget exhausted, could not generate data.')
            logger.info('Exiting.')
            sys.exit(-1)

    n = len(rules)
    actual_callbacks_step = max(callbacks_step if isinstance(callbacks_step, int) else int(n * callbacks_step), n)
    logger.info('Merging...')
    glocalx = GLocalX(oracle=oracle)
    glocalx = glocalx.fit(rules, tr_set,
                          batch_size=batch_size if batch_size > 0 else tr_set.shape[0],
                          intersecting=intersecting, global_direction=global_direction,
                          strict_join=strict_join, strict_cut=strict_cut,
                          fidelity_weight=fidelity_weight, complexity_weight=complexity_weight,
                          callback_step=actual_callbacks_step,
                          callbacks=[print_cb, full_cb, final_rule_dump_cb])

    logger.info('Storing output rules ' + name + '...')
    output_rules = glocalx.rules(alpha, tr_set)

    jsonized_rules = [rule.json() for rule in output_rules]
    with open(name + '.rules.glocalx.alpha=' + str(alpha) + '.json', 'w') as log:
        json.dump(jsonized_rules, log)


if __name__ == '__main__':
    cl_run()
