"""Serialize and deserialize objects and rules."""
import pickle

from tensorflow.keras.models import load_model as load_tf_model
import numpy as np

import json
import os

from glocalx import GLocalX
from models import Rule


def load_run(run_file):
    """
    The rules file output by GLocalX is slightly different than the input one loaded
    with models.Rule.from_json, as it contains additional information on the run, such as
    the input files paths, hyperparameters, etc.
    Args:
        run_file (str): Path to the rules file.

    Returns:
        dict: Dictionary holding the output rules (`rules` key), the oracle, if any, (`black_box`) key,
                the training set (`tr` key) and the undersample (`undersample` key). Note that regardless
                of whether the black box is loaded or not, the training set preserves its original labels.
                If you want to use the training set with the black box labels you have to re-infer them.
    """
    if not os.path.isfile(run_file):
        raise ValueError('File ' + str(run_file) + ' is not a file.')
    with open(run_file, 'r') as log:
        dic = json.load(log)

    # Load features names from the training file
    tr_path = dic['tr']
    if not os.path.isfile(tr_path):
        raise ValueError('Training file ' + str(tr_path) + ' is not a file.')
    data = np.genfromtxt(tr_path, delimiter=',', names=True)
    names = data.dtype.names
    tr_set = data.view(np.float).reshape(data.shape + (-1,))

    # Load rules
    rules = dic['rules']
    premises = [{int(k): v for k, v in d.items() if k != 'consequence' and k != 'label'} for d in rules]
    consequences = [dic['consequence'] if 'consequence' in dic else dic['label'] for dic in rules]
    rules = [Rule(premises=premise, consequence=consequence, names=names)
             for premise, consequence in zip(premises, consequences)]

    # Load oracle
    oracle = dic['oracle']
    if oracle.endswith('.h5'):
        oracle = load_tf_model(oracle)
    elif oracle.endswith('.pickle'):
        with open(oracle, 'rb') as log:
            oracle = pickle.load(log)
    else:
        return

    # Load undersample
    undersample = dic['undersample']

    results = {
        'rules': rules,
        'oracle': oracle,
        'undersample': undersample,
        'tr': tr_set
    }

    return results


def load_glocalx(rules, is_glocalx_run=False):
    """
    Create a GLocalX instance from `rules_file`. Rules from `rules_file` are considered as
    this instance's output, i.e. its `self.fine_boundary`.
    Args:
        rules (Union(str, set, list)): Path to rules, or directly set/list of rules.
        is_glocalx_run (bool): Whether the given rule file is the output of a GLocalX run or not.
                                GLocalX stores its output file in a different format than the input
                                rules.

    Returns:
        GLocalX: A GLocalX instance as if it were trained and returned `rules`
    """
    if isinstance(rules, str) and not os.path.isfile(rules):
        raise ValueError('Not a valid file')

    if isinstance(rules, str):
        if is_glocalx_run:
            run = load_run(rules)
            fine_boundary = run['rules']
            glocalx = GLocalX(oracle=run['oracle'])
        else:
            fine_boundary = Rule.from_json(rules)
            glocalx = GLocalX(oracle=None)
    elif isinstance(rules, set) or isinstance(rules, list):
        fine_boundary = rules
        glocalx = GLocalX(oracle=None)
    else:
        raise ValueError('Not a str or set or list')

    # Load rules in the boundary
    glocalx.fine_boundary = fine_boundary

    return glocalx
