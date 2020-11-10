from argparse import ArgumentParser

from pandas import read_csv

from transform import treepan_to_rules

from api.utils import rules_to_json
from adversary.trepan import __compute_distributions, trepan

import pickle

with open('../../l2g/data/processed/black_boxes/black_boxes.pickle', 'rb') as log:
    bbs = pickle.load(log)

with open('../../l2g/data/processed/black_boxes/encoders.pickle', 'rb') as log:
    encoders = pickle.load(log)


def run(tr_files, ts_files, black_boxes_files, output_files):
    """Run TREPAN on the provided datasets and black boxes.

        Arguments:
            tr_files (list): List of training file paths.
            ts_files (list): List of test file paths.
            black_boxes_files (list): List of black boxes file paths.
            output_files (list): List of output file paths.

        Return:
            {None} -- Nothing.
    """
    out = {}
    for tr_file, ts_file, black_boxes_file, output_file in zip(tr_files, ts_files, black_boxes_files, output_files):
        dataset_name = output_file.split('.json')[0]
        print('Running on ' + str(dataset_name) + '...')

        out[dataset_name] = {}
        tr = read_csv(tr_file)
        ts = read_csv(ts_file)
        tr.columns = range(tr.shape[1])
        ts.columns = range(ts.shape[1])
        _, populations = __compute_distributions(tr)
        types = {t: tr.dtypes[t] for t in range(tr.shape[1])}
        with open(black_boxes_file, 'rb') as log:
            black_boxes = pickle.load(log)

        for black_box_name, black_box in black_boxes.items():
            print('\t' + black_box_name)
            rules = treepan_to_rules(trepan(tr, black_box), populations, types)
            out[dataset_name][black_box_name] = rules
            rules_to_json(rules, output_file)

    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-tr', metavar='tr', type=str, nargs='+',
                        help='TR set csv files.')
    parser.add_argument('-ts', metavar='ts', type=str, nargs='+',
                        help='TS set csv files, one per TR file.')
    parser.add_argument('-B_', metavar='black_boxes', type=str, nargs='+',
                        help='Black boxes pickle files. Should hold a dictionary'
                             'black box name => black box, one per TR/TS set'
                             ' Black boxes must implement a predict(x) method.')
    parser.add_argument('-o', dest='output_files', action='store_const',
                        help='Output files: defaults to the tr set file names.')

    args = parser.parse_args()
    if len(args.tr) == 0 or len(args.ts) == 0:
        print('Indicate at least one TR set and one TS set.')
        exit(-1)
    if len(args.black_boxes) == 0:
        print('Indicate at least one black box file.')
        exit(-1)
    if len(args.tr) != len(args.ts):
        print('TR and TS files should be in equal numbers, but got ' + str(len(args.tr)) + ', ' + str(len(args.ts)))
        exit(-1)
    if len(args.tr) != len(args.black_boxes) or len(args.ts) != len(args.black_boxes):
        print('TR, TS and black box files should be in equal numbers, but got '
              + str(len(args.tr)) + ', ' + str(len(args.ts) + ', ' + str(len(args.black_boxes))))
        exit(-1)

    if len(str.output_files) == 0:
        names = list(map(lambda file: file.split('/')[-1].split('.csv')[0] + '.json', args.tr))
    else:
        names = list(map(lambda x: x + '.json', args.output_files))
