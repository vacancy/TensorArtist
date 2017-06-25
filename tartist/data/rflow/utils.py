# -*- coding:utf8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/28/17
# 
# This file is part of TensorArtist.

import zmq
import socket
import uuid
import json

json_dumpb = lambda x: json.dumps(x).encode('utf-8')
json_loadb = lambda x: json.loads(x.decode('utf-8'))


def router_recv_json(sock, flag=zmq.NOBLOCK, loader=json_loadb):
    try:
        identifier, delim, *payload = sock.recv_multipart(flag)
        return [identifier] + list(map(lambda x: loader(x), payload))
    except zmq.error.ZMQError:
        return None, None


def router_send_json(sock, identifier, *payloads, flag=0, dumper=json_dumpb):
    try:
        buf = [identifier, b'']
        buf.extend(map(lambda x: dumper(x), payloads))
        sock.send_multipart(buf, flags=flag)
    except zmq.error.ZMQError:
        return False
    return True


def req_recv_json(sock, flag=0, loader=json_loadb):
    try:
        response = sock.recv_multipart(flag)
        response = list(map(lambda x: loader(x), response))
        return response[0] if len(response) == 1 else response
    except zmq.error.ZMQError:
        return None


def req_send_json(sock, *payloads, flag=0, dumper=json_dumpb):
    buf = []
    buf.extend(map(lambda x: dumper(x), payloads))
    try:
        sock.send_multipart(buf, flag)
    except zmq.error.ZMQError:
        return False
    return True


def iter_recv(meth, sock):
    while True:
        res = meth(sock, flag=zmq.NOBLOCK)
        succ = res[0] is not None if isinstance(res, (tuple, list)) else res is not None
        if succ:
            yield res
        else:
            break


def req_send_and_recv(sock, *payloads):
    req_send_json(sock, *payloads)
    return req_recv_json(sock)


def push_pyobj(sock, data, flag=zmq.NOBLOCK):
    try:
        sock.send_pyobj(data, flag)
    except zmq.error.ZMQError:
        return False
    return True


def pull_pyobj(sock, flag=zmq.NOBLOCK):
    try:
        response = sock.recv_pyobj(flag)
        return response
    except zmq.error.ZMQError:
        return None


def bind_to_random_ipc(sock, name):
    name = name + uuid.uuid4().hex[:8]
    conn = 'ipc:///tmp/{}'.format(name)
    sock.bind(conn)
    return conn


def uid():
    return socket.gethostname() + '/' + uuid.uuid4().hex


def get_addrv1():
    try:
        return socket.gethostbyname(socket.gethostname())
    except:
        return '127.0.0.1'


def get_addrv2():
    try:
        return _get_addrv2_impl()
    except:
        # fallback to addrv1
        return get_addrv1()


# http://stackoverflow.com/questions/166506/finding-local-ip-addresses-using-pythons-stdlib
def _get_addrv2_impl():
    resolve = [ip for ip in socket.gethostbyname_ex(socket.gethostname())[2] if not ip.startswith("127.")][:1]
    if len(resolve):
        return resolve[0]

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    addr = s.getsockname()[0]
    s.close()
    return addr


get_addr = get_addrv2


def graceful_close(sock):
    if sock is None:
        return
    sock.setsockopt(zmq.LINGER, 0)
    sock.close()
