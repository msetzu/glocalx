"""API module for Ethica."""
import sys
import os
import time
import pickle

from tqdm import tqdm

# Shut up, tensorflow!
from glocalx import GLocalX
from generators import TrePanGenerator, BudgedExhaustedException

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import click
import logzero
from numpy import genfromtxt, hstack
from numpy.random import choice
import pandas as pd
from tensorflow.keras.models import load_model

from logzero import logger

from models import Rule


def undersample(rules_file, tr, oracle=None, intersecting='coverage', batch_size=128,
                concordance=False, strong_cut=False,
                fidelity_weight=1., complexity_weight=1., global_direction=True,
                name=None, names=None, sample_pct=None, nr_samples=10):
    """
    Perform `nr_samples` of `sample_pct`% of the given `rules`, and validate
    the aggregated Ethica runs.
    Args:
        rules_file (str): JSON file with the train set.
        tr (ndarray): Train set.
        oracle (str): Path to the oracle to use.
        intersecting (str): Whether to use coverage intersection ('coverage') or polyhedra intersection ('polyhedra').
                            Defaults to 'coverage'.
        batch_size (int): Batch size. Defaults to 128.
        concordance (bool): True to use high concordance, false otherwise. Defaults to False.
        strong_cut (bool): True for a strong cut, false otherwise. Defaults to False.
        global_direction (bool): False to compute the BIC on the merged theory, True to compute it on the whole model.
                                Defaults to False.
        fidelity_weight (float): Fidelity weight. Defaults to 1.
        complexity_weight (float): Complexity weight. Defaults to 1.
        name (str): Name for the output logs.
        names (str): Features names.
        sample_pct (Union(float, list)): Percentage of rules to use, if float. Iterable of percentage of rules
                                                to use, if iterable.
        nr_samples (int): Number of repetitions. Defaults to 10.
    """
    rules = Rule.from_json(rules_file)
    rules = [r for r in rules if len(r) > 0]
    n = len(rules)
    if isinstance(sample_pct, float):
        sample_pcts = [sample_pct]
    else:
        sample_pcts = sample_pct

    for sample_pct in sample_pcts:
        sample_size = int(sample_pct * n)

        models = []
        sample_indices_space = range(n)
        for _ in tqdm(list(range(nr_samples))):
            rules = Rule.from_json(rules_file)
            rules = [r for r in rules if len(r) > 0]
            # Select sample
            sample_indices = choice(sample_indices_space, (sample_size,))
            rules_sample = [rules[i] for i in sample_indices]

            # Run GLocalX
            model = GLocalX(oracle=oracle, intersecting=intersecting, high_concordance=concordance,
                            strong_cut=strong_cut, name=name)
            model = model.fit(rules_sample, tr,
                              batch_size=batch_size,
                              global_direction=global_direction,
                              fidelity_weight=fidelity_weight, complexity_weight=complexity_weight,
                              callback_step=-1,
                              callbacks=[])
            # Append results
            models.append(model)
        with open(name + ';undersample:' + str(sample_pct) + '.pickle', 'wb') as log:
            pickle.dump(models, log)


@click.command()
# Compulsory arguments: input rules, TR set, TS set, oracle
@click.argument('rules', type=click.Path(exists=True))
@click.argument('tr', type=click.Path(exists=True))
@click.argument('experiment')
@click.option('-o', '--oracle', type=click.Path(exists=True))
# Input descriptions
@click.option('--names',  default=None, help='Features names.')
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
@click.option('-b', '--batch', default=128, help='Batch size. Set to -1 for full batch. Defaults to 128.')
@click.option('-u', '--undersample_pct', default=1., help='Undersample size, to use a percentage of the rules. '
                                                          'Defaults to 1.0 (No undersample).')
# Merge options
@click.option('--concordance', is_flag=True, help='Use to use high concordance.')
@click.option('--strong_cut', is_flag=True, help='Use to use the strong cut.')
@click.option('--global_direction', is_flag=True, help='Use to use the global search direction.')
# Set debug class
@click.option('-d', '--debug', default=20, help='Debug level.')
def cl_run(rules, tr, generate=None, oracle=None, batch=128, alpha=0.5, undersample_pct=1.0,
           fidelity_weight=1., complexity_weight=1., global_direction=False,
           intersect='coverage', concordance=False, strong_cut=False,
           name=None, names=None, debug=20):
    run(rules, tr=tr, oracle=oracle, generate=generate,
        intersecting=intersect, batch_size=batch, alpha=alpha, undersample_pct=undersample_pct,
        concordance=concordance, strong_cut=strong_cut, global_direction=global_direction,
        fidelity_weight=fidelity_weight, complexity_weight=complexity_weight,
        name=name, names=names, debug=debug)


def run(rules, tr, oracle=None, generate=None,
        intersecting='coverage', batch_size=128, alpha=0.5,
        concordance=False, strong_cut=False, global_direction=False,
        fidelity_weight=1, complexity_weight=1,
        nr_samples=10, undersample_pct=0.5,
        name=None, names=None, debug=20):
    """Run the Ethica framework on a set of rules.
    Arguments:
        rules (str): JSON file with the train set.
        tr (str): Train set.
        oracle (str): Path to the oracle to use.
        generate (Union(int, float, None)): Size of the synthetic dataset to use, if not using the training set.
                                            Use float to give size w.r.t. the TR to use.
                                            Defaults to None (use training set).
        intersecting (str): Whether to use coverage intersection ('coverage') or polyhedra intersection ('polyhedra').
                            Defaults to 'coverage'.
        batch_size (int): Batch size. Defaults to 128.
        global_direction (bool): False to compute the BIC on the merged theory, True to compute it on the whole model.
                                Defaults to False.
        fidelity_weight (float): Fidelity weight. Defaults to 1.
        complexity_weight (float): Complexity weight. Defaults to 1.
        alpha (float): Pruning factor. Defaults to 0.5.
        nr_samples (int): How many repetitions for the given `undersample_pct`?
        undersample_pct (float): Percentage of rules to use, if < 1, irrelevant otherwise. Defaults to 1.
        concordance (bool): True to use high concordance, false otherwise. Defaults to False.
        strong_cut (bool): True for a strong cut, false otherwise. Defaults to False.
        name (str): Name for the output logs.
        names (str): Features names.
        debug (int): Minimum debug level.
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
        name = str(time.time())
    elif os.path.exists(name + '.ethica.pickle'):
        logger.info('Ethica run already existing: ' + name + '.ethica.pickle. Exiting!')
        return

    if names is not None:
        # For some weird reason '.csv' sometimes becomes '.cs'
        if names.endswith('.cs'):
            names = names + 'v'
    else:
        # Names stored in data/$name_names.csv
        names = 'data/' + name.split('.')[0] + '_names.csv'
    names = pd.read_csv(names, header=None).values.tolist()[0]

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
    tr_set = genfromtxt(tr, delimiter=',')

    logger.info('Loading oracle...')
    if oracle is not None:
        if oracle.endswith('.h5'):
            oracle = load_model(oracle)
        elif oracle.endswith('.pickle'):
            with open(oracle, 'rb') as log:
                oracle = pickle.load(log)
        else:
            return
        oracle_predictions = oracle.predict(tr_set[1:, :-1]).round().reshape((tr_set.shape[0] - 1, 1))
        tr_set = hstack((tr_set[1:, :-1], oracle_predictions))

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

    logger.info('Undersampling...')
    undersample(rules, oracle=oracle, tr=tr_set, sample_pct=undersample_pct,
                nr_samples=nr_samples, intersecting=intersecting, batch_size=batch_size,
                name=name, concordance=concordance, strong_cut=strong_cut,
                fidelity_weight=fidelity_weight, complexity_weight=complexity_weight,
                global_direction=global_direction,
                names=names)


if __name__ == '__main__':
    cl_run()
