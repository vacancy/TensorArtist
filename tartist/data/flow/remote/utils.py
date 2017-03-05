# -*- coding:utf8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/28/17
# 
# This file is part of TensorArtist

import zmq
import socket
import uuid
import json


def router_recv_json(sock):
    try:
        identifier, delim, *payload = sock.recv_multipart(zmq.NOBLOCK)
        return [identifier] + list(map(lambda x: json.loads(x.decode('utf-8')), payload))
    except zmq.error.ZMQError:
        return None, None


def router_send_json(sock, identifier, *payloads):
    buf = [identifier, b'']
    buf.extend(map(lambda x: json.dumps(x).encode('utf-8'), payloads))
    return sock.send_multipart(buf)


def req_send_json(sock, *payloads, flag=0):
    buf = []
    buf.extend(map(lambda x: json.dumps(x).encode('utf-8'), payloads))
    try:
        sock.send_multipart(buf, flag)
    except zmq.error.ZMQError:
        return False
    return True


def req_recv_json(sock, flag=0):
    try:
        response = sock.recv_multipart(flag)
        response = list(map(lambda x: json.loads(x.decode('utf-8')), response))
        return response[0] if len(response) == 1 else response
    except zmq.error.ZMQError:
        return None


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


def uid():
    return socket.gethostname() + '/' + uuid.uuid4().hex


def get_addr():
    return socket.gethostbyname(socket.gethostname())


def graceful_close(sock):
    if sock is None:
        return
    sock.setsockopt(zmq.LINGER, 0)
    sock.close()

