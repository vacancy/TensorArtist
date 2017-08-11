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
args = parser.parse_args()


def main():
    db = LMDBKVStore(args.input_dir, readonly=False)

    for f in tqdm(db.keys()):
        img = db.get(f)
        image.imshow(f, img)


if __name__ == '__main__':
    main()

