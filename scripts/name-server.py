# -*- coding:utf8 -*-
# File   : name_server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/2/17
# 
# This file is part of TensorArtist.

from tartist.core import get_logger
from tartist.data.rflow.name_server import NameServer
from tartist.data.rflow import configs
import argparse

logger = get_logger(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--host', dest='host', default=configs.NS_CTL_HOST)
parser.add_argument('-p', '--port', dest='port', default=configs.NS_CTL_PORT)
parser.add_argument('--protocal', dest='protocal', default=configs.NS_CTL_PROTOCAL)
args = parser.parse_args()


if __name__ == '__main__':
    logger.critical('Starting name server at {}://{}:{}.'.format(args.protocal, args.host, args.port))
    NameServer(host=args.host, port=args.port, protocal=args.protocal).mainloop()
