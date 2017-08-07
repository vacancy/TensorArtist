# -*- coding:utf8 -*-
# File   : start-collector.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 03/07/2017
# 
# This file is part of TensorArtist.

import argparse
from tartist.core import get_logger
from tartist.core.utils.network import get_local_addr
from libhpref.webserver import TrajectoryPairPool, WebServer

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port', dest='port', default=8888)
args = parser.parse_args()


def main():
    test_configs = {
        'title': 'RL Human Preference Collector',
        'author': 'TensorArtist authors',
        'port': args.port
    }

    pool = TrajectoryPairPool()
    server = WebServer(pool, test_configs)

    logger.critical('Starting webserver at http://{}:{}'.format(get_local_addr(), args.port))
    server.mainloop()


if __name__ == '__main__':
    main()
