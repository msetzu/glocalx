import os
import shutil
import subprocess

import numpy as np
import pandas as pd

from collections import defaultdict


wdir = "__rbcache__/"


def clean():
    global wdir
    if os.path.exists(wdir):
         shutil.rmtree(wdir)


class RuleBasedClassifier:

    def __init__(self, alg='CPAR', options='-Xmx4G'):
        if alg != 'CPAR' and alg != 'FOIL':
            raise ValueError("Algorithm must be 'CPAR' or 'FOIL'")

        global wdir
        if not os.path.exists(wdir):
            os.mkdir(wdir)
        self.alg = alg
        self.name = wdir + '_{}_{}'.format(alg, id(self))
        self.options = options
        self.rules = None
        self.class_value_counts = None

    def fit(self, X=None, y=None, deletefile=True, verbose=True, **kwargs):
        if X is None:
            raise Exception('RuleBasedClassifier: no data given to fit')
        if y is None:
            raise Exception('RuleBasedClassifier: no class given to fit')

        class_value, counts = np.unique(y, return_counts=True)
        self.class_value_counts = {k: v for k, v in zip(class_value, counts)}

        # Update
        filename = self.name + '.data'
        nbr_classes = len(np.unique(y))
        cmd = None

        if self.alg == 'CPAR':
            params = {'delta': 0.05, 'alpha': 0.3, 'gain_similarity_ratio': 0.99, 'min_gain_thr': 0.7}
            for k in params:
                if k in kwargs:
                    params[k] = kwargs[k]
            cmd = 'java %s -cp %s CPAR -f %s -n %s -w %s -d %s -s %s -g %s' % (self.options, 'jrbc',
                                                                                 filename, nbr_classes, params['delta'], params['alpha'], params['gain_similarity_ratio'],
                                                                                 params['min_gain_thr'])

        if self.alg == 'FOIL':
            params = {'max_attr_antecedent': 3, 'min_gain_thr': 0.7}
            for k in params:
                if k in kwargs:
                    params[k] = kwargs[k]
            cmd = 'java %s -cp jrbc FOIL -f %s -n %s -m %s -g %s' % (self.options,
                                                                     filename, nbr_classes, params['max_attr_antecedent'], params['min_gain_thr'])

        df = pd.DataFrame(data=np.column_stack((X, y)))
        df.to_csv(filename, sep=' ', header=False, index=False)

        col_values = dict()
        val_column = dict()
        for i in range(0, X.shape[1]):
            col_values[i] = np.unique(X[:, i])
            for v in col_values[i]:
                val_column[v] = i
        # for i in range(0, df.shape[1]):
        #     col_values[i] = np.unique(df[i].values)
        #     for v in col_values[i]:
        #         val_column[v] = i

        if verbose:
            print(cmd)
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

        # if deletefile and os.path.exists(filename):
        #     os.remove(filename)
        # os.chdir('../../../..')

        self.rules = list()
        for output_rule in output.decode('UTF-8').split('\n'):
            output_rule_fields = output_rule.split(';') # class, encoded_int value,
            if len(output_rule_fields) < 3:
                continue
            cons = int(output_rule_fields[0])
            ant = {val_column[int(v)]: int(v) for v in output_rule_fields[1].split(',')}
            laplace = float(output_rule_fields[2])

            rule = tuple([cons, ant, laplace])
            if rule not in self.rules:
                self.rules.append(rule)

        return self

    def predict(self, X, k=5):
        y = list()
        for x in X:
            y.append(self.predict_record(x, k))
        y = np.array(y)
        return y

    def predict_record(self, x, k=5):
        satisfied = self.__get_rules_satisfied(x)
        if len(satisfied) == 0:
            return max(self.class_value_counts, key=self.class_value_counts.get)

        cons_score = dict()
        for cons in satisfied:
            bestrules = sorted(satisfied[cons], key=lambda rule: rule[2], reverse=True)[:k]
            laccuray = [rule[2] for rule in bestrules]
            cons_score[cons] = 0.0 if len(laccuray) == 0 else np.mean(laccuray)

        return max(cons_score, key=cons_score.get)

    def __get_rules_satisfied(self, x):
        satisfied = defaultdict(list)
        for rule in self.rules:
            cons, ant, laplace = rule
            is_satisfied = True
            for col, val in ant.items():
                if x[col] != val:
                    is_satisfied = False
                    break
            if is_satisfied:
                satisfied[cons].append(rule)
        return satisfied
