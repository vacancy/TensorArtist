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
parser.add_argument('-o', '--output', dest='output_dir', required=True)
args = parser.parse_args()

logger.critical('Generating lmdb:')
logger.critical('  Source images  : {}/*.jpg'.format(args.input_dir))
logger.critical('  Target database: {}'.format(args.output_dir))

def main():
    dbdir = osp.realpath(args.output_dir)
    io.mkdir(dbdir)
    db = LMDBKVStore(dbdir, readonly=False)

    files = glob.glob(osp.join(args.input_dir, '*.jpg'))

    with db.transaction():
        for f in tqdm(files):
            img = image.imread(f)
            fid = osp.basename(f)[:-4]
            img = image.resize_minmax(img, 256, 10000)
            img = image.center_crop(img, 256)
            db.put(fid, image.jpeg_encode(img))


if __name__ == '__main__':
    main()

