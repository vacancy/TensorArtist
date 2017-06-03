# -*- coding:utf8 -*-
# File   : name_server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/2/17
# 
# This file is part of TensorArtist

from tartist.data.rflow.name_server import NameServer
from tartist.data.rflow import configs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-h', '--host', dest='host', default=configs.NS_CTL_HOST)
parser.add_argument('-p', '--port', dest='port', default=configs.NS_CTL_PORT)
parser.add_argument('--protocal', dest='protocal', default=configs.NS_CTL_PROTOCAL)
args = parser.parse_args()


if __name__ == '__main__':
    NameServer(host=args.host, port=args.port, protocal=args.protocal).mainloop()
