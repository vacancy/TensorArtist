from tartist import io, image
from tartist import get_logger
from tartist.data.kvstore import LMDBKVStore

logger = get_logger(__file__)

import os.path as osp
import argparse
import glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', dest='input_dir', required=True)
parser.add_argument('-o', '--output', dest='output_dir', required=True, nargs=2)
args = parser.parse_args()

logger.critical('Generating lmdb:')
logger.critical('  Source images  : {}/*.jpg'.format(args.input_dir))
logger.critical('  Target database: {} {}'.format(*args.output_dir))

def main():
    dbs = []
    for i in range(2):
        dbdir = osp.realpath(args.output_dir[i])
        io.mkdir(dbdir)
        dbs.append(LMDBKVStore(dbdir, readonly=False))

    files = glob.glob(osp.join(args.input_dir, '*.jpg'))

    with dbs[0].transaction(), dbs[1].transaction():
        for f in tqdm(files):
            img = image.imread(f)
            fid = osp.basename(f)[:-4]
            dbs[0].put(fid, image.jpeg_encode(img[:, :256]))
            dbs[1].put(fid, image.jpeg_encode(img[:, 256:]))


if __name__ == '__main__':
    main()

