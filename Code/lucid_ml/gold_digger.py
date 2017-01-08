#!/usr/bin/env python3
import argparse
import pandas as pd
from operator import itemgetter
from collections import Counter
import numpy as np
try:
    import matplotlib.pyplot as plt
    PLOT=True
except ImportError:
    print("WARNING: ImportError on matplotlib, I will not draw graphics")
    PLOT=False


def gather(filehandle, sep=None, ignore=0):
    cnt = Counter()
    for line in filehandle:
        elems = line.strip().split(sep)
        print(elems[ignore:])
        docid = elems[0]
        cnt[docid] += len(elems[ignore:])
    return cnt

def dig(filehandle, sep=None, ignore=0, normalize=False):
    cnt = Counter()
    for norm, line in enumerate(filehandle):
        elems = line.strip().split(sep)[ignore:]
        for elem in elems:
            cnt[elem] += 1

    if normalize:
        norm += 1
        cnt = {k: v / norm for k, v in cnt.items()}

    return cnt

# method for more beauty


def main(ns):
    if PLOT:
        plt.figure(1)
    for fh in ns.file:
        if ns.gather:
            cnt = gather(fh, sep=ns.seperator, ignore=ns.ignore)
        else:
            cnt = dig(fh, sep=ns.seperator, ignore=ns.ignore,
                      normalize=ns.normalize)

        gold = list(sorted(cnt.items(),
                           reverse=True,
                           key=itemgetter(1)))[:ns.top]
        if ns.verbose:
            for i, (label, count) in enumerate(gold):
                print("[{}] '{}' : {}".format(i, label, count))

        gold_labels, gold_values = zip(*gold)
        df = pd.DataFrame(data={'occurrences': gold_values}, index=gold_labels)
        #  filter
        df = df[df.occurrences > ns.min]
        print("[{}] Statistics".format(fh.name), df.describe(), sep='\n')
        if PLOT:
            plt.plot(np.arange(len(gold_values)), gold_values)

    if PLOT:
        # we need a legende
        plt.legend([fh.name for fh in ns.file])

        if ns.outfile is None:
            plt.show()
        else:
            plt.savefig(ns.outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=argparse.FileType(mode='r'), nargs='+',
                        help='Dig for gold in this file(s)')
    parser.add_argument('-s', '--seperator', default='\t',
                        help='Column seperator (defaults to tab)')
    parser.add_argument('-i', '--ignore', type=int, default=1,
                        help='Ignore first IGN columns', metavar='IGN')
    parser.add_argument('-m', '--min', type=float, default=0.0, metavar='MIN',
                        help='Show labels with at least MIN occurences')
    parser.add_argument('-t', '--top', type=int, default=-1,
                        help='Show top N labels', metavar='K')
    parser.add_argument('-o', '--outfile', type=str, default=None,
                        help='Specify OUT file', metavar='OUT')
    parser.add_argument('-g', '--gather', action='store_true', default=False,
                        help='Gather labels per document statistics instead of document per label')
    parser.add_argument('-n', '--normalize', action='store_true',
                        dest='normalize',
                        default=False,
                        help='Do not Normalize with respect to document count')
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    args = parser.parse_args()
    main(args)
