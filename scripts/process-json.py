import os
import argparse
import sys
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('summaries', metavar='N', type=str, nargs='+',
                        help='json summary files')
    parser.add_argument('--fields', metavar='N', type=str, nargs='+',
                        help='json fields for each file')
    parser.add_argument('--legends', metavar='N', type=str, nargs='+',
                        help='legends for each curve')
    parser.add_argument('--output', type=str, default='output', help='output file')

    args = parser.parse_args()

    summaries = []
    for i in args.summaries:
        if os.path.isdir(i):
            summaries.append(os.path.join(i, 'summary.json'))
        else:
            summaries.append(i)

    n = len(summaries) 
    fields = args.fields or ['async/score/avg'] * n
    assert n == len(fields)

    data = []
    m = 0
    for i in range(n):
        s = []
        for line in open(summaries[i]):
            a = json.loads(line)
            s.append(a[fields[i]])
        data.append(s)
        m = max(m, len(s))

    with open(args.output, 'w') as f:
        for j in range(m):
            o = []
            for i in data:
                if j < len(i):
                    o.append(str(i[j]))
                else:
                    o.append('')
            f.write('\t'.join(o) + '\n')

    legends = args.legends
    if legends is None:
        legends = []
        for i, j in zip(summaries, fields):
            legends.append(os.path.dirname(i) + '.' + j)
    assert n == len(legends)
    cmd = "cat {} | ~/tools/tensorpack/scripts/plot-point.py --xlabel 'epoch' --ylabel 'score' --legend {} --decay 0.6".format(args.output, ','.join(legends))
    print('preprocess data done, run the following command to get plot')
    print(cmd)

if __name__ == '__main__':
    main()
