# -*- coding:utf8 -*-
# File   : controller.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/28/17
# 
# This file is part of TensorArtist

from . import configs, utils
from ....core.logger import get_logger
from ....core.utils.callback import CallbackManager
import zmq
import os
import threading
import queue
import itertools
import contextlib

logger = get_logger(__file__)


class Controller(object):
    def __init__(self):
        self._uid = utils.uid()
        self._addr = utils.get_addr()

        self._ipipes = {}
        self._opipes = {}
        self._opipes_enque_mark = {}

        self._context = zmq.Context()
        self._context.sndhwm = configs.CTL_DAT_HWM
        self._context.rcvhwm = configs.CTL_DAT_HWM
        self._poller = zmq.Poller()

        self._ns_socket = self._context.socket(zmq.REQ)
        self._ipeers_csock = set()
        self._ipeers_dsock = set()
        self._opeers_dsock = set()

        self._control_router = self._context.socket(zmq.ROUTER)
        self._control_router_port = 0

        self._control_dispatcher = CallbackManager()

        self._ipeers = {}
        self._opeers = {}
        self._control_send_queue = queue.Queue()
        self._data_send_queue = queue.Queue()
        self._data_recv_queue = None

        self._all_threads = []
        self._stop_event = threading.Event()

    def initialize(self, pipes=None):
        pipes = pipes or []

        for pipe in pipes:
            if pipe.direction == 'IN':
                self._ipipes.setdefault(pipe.name, []).append(pipe)
            else:
                assert pipe.direction == 'OUT'
                self._opipes.setdefault(pipe.name, []).append(pipe)
                self._opipes_enque_mark[pipe] = True
            pipe.set_controller(self)

        # TODO:: should be nr_input_pipes
        self._data_recv_queue = queue.Queue(configs.CTL_DAT_HWM * len(pipes))

        # setup ns socket
        self._ns_socket.connect(os.getenv(
            'TART_NAME_SERVER', '{}://localhost:{}'.format(
                configs.NS_CTL_PROTOCAL, configs.NS_CTL_PORT
            )
        ))
        self._poller.register(self._ns_socket, zmq.POLLIN)

        # setup router socket
        self._control_router_port = self._control_router.bind_to_random_port('tcp://*')
        self._poller.register(self._control_router, zmq.POLLIN)

        # register on the name-server
        response = utils.req_send_and_recv(self._ns_socket, {
            'action': configs.Actions.NS_REGISTER_CTL,
            'uid': self._uid,
            'ctl_protocal': 'tcp',
            'ctl_addr': self._addr,
            'ctl_port': self._control_router_port,
            'meta': {}
        })
        assert response['action'] == configs.Actions.NS_REGISTER_CTL_SUCC

        # register pipes on name-server
        response = utils.req_send_and_recv(self._ns_socket, {
            'action': configs.Actions.NS_REGISTER_OPIPE,
            'uid': self._uid,
            'ipipes': list(self._ipipes.keys()),
            'opipes': list(self._opipes.keys())
        })
        assert response['action'] == configs.Actions.NS_REGISTER_OPIPE_SUCC

        # query name-server for ipipes
        response = utils.req_send_and_recv(self._ns_socket, {
            'action': configs.Actions.NS_QUERY_OPIPE,
            'pipe_names': list(self._ipipes.keys())
        })
        assert response['action'] == configs.Actions.NS_QUERY_OPIPE_RESPONSE
        logger.info('ipipes query {}'.format(response['results']))

        all_peers = []
        for k, v in response['results'].items():
            all_peers.extend(v)
        for i in all_peers:
            if i['uid'] not in self._ipeers:
                self._ipeers[i['uid']] = [i, None, None]  # info, control_socket, data_socket
                self._do_setup_ipeer_csock(i['uid'])

        # setup dispatcher
        self._control_dispatcher.register(configs.Actions.CTL_HEARTBEAT, self._do_response_heartbeat)
        self._control_dispatcher.register(configs.Actions.CTL_OPEN, self._do_setup_opeer)
        self._control_dispatcher.register(configs.Actions.CTL_OPENED, self._do_setup_opeer_finished)
        self._control_dispatcher.register(configs.Actions.NS_OPEN_CTL, self._do_ns_open_ctl)
        self._control_dispatcher.register(configs.Actions.NS_CLOSE_CTL, self._do_ns_close_ctl)
        self._control_dispatcher.register(configs.Actions.NS_HEARTBEAT_SUCC, lambda msg: None)
        self._control_dispatcher.register(configs.Actions.CTL_OPEN_RESPONSE, self._do_setup_ipeer_csock_response)

        # run threads
        self._all_threads.append(threading.Thread(target=self.main, name='ctl-main'))
        self._all_threads.append(threading.Thread(target=self.main_ns_heartbeat, name='ctl-main-ns-heartbeat'))
        # all_threads.append(threading.Thread(target=self.main_ctl_heartbeat, name='ctl-main-ctl-heartbeat'))
        for i in self._all_threads:
            i.start()

    def finalize(self):
        self._stop_event.set()
        for i in self._all_threads:
            i.join()
        for sock in itertools.chain(self._ipeers_csock, self._ipeers_dsock, self._opeers_dsock,
                                    [self._ns_socket, self._control_router]):

            utils.graceful_close(sock)

    def main(self):
        while True:
            if self._stop_event.is_set():
                break

            socks = dict(self._poller.poll(100))
            self._main_do_control_recv(socks)
            self._main_do_control_send()
            self._main_do_input(socks)
            self._main_do_output()
            self._main_do_ipipe_enque()
            self._main_do_opipe_enque()

    def main_ns_heartbeat(self):
        while True:
            self._control_send_queue.put({
                'sock': self._ns_socket,
                'countdown': 0,
                'payload': {
                    'action': configs.Actions.NS_HEARTBEAT,
                    'uid': self._uid
                }
            })

            if self._stop_event.wait(configs.NS_HEARTBEAT_INTERVAL):
                break

    def main_ctl_heartbeat(self):
        while True:
            if self._stop_event.wait(configs.CTL_HEARTBEAT_INTERVAL):
                break

    def _main_do_control_send(self):
        nr_send = self._control_send_queue.qsize()
        for i in range(nr_send):
            job = self._control_send_queue.get()
            if 'identifier' in job:
                rc = utils.router_send_json(job['sock'], job['identifier'], job['payload'], flag=zmq.NOBLOCK)
            else:
                rc = utils.req_send_json(job['sock'], job['payload'], flag=zmq.NOBLOCK)
            if not rc:
                job['countdown'] -= 1
                if job['countdown'] >= 0:
                    self._control_send_queue.put(job)

    def _main_do_control_recv(self, socks):
        # ns
        if self._ns_socket in socks and socks[self._ns_socket] == zmq.POLLIN:
            for msg in utils.iter_recv(utils.req_recv_json, self._ns_socket):
                self._control_dispatcher.dispatch(msg['action'], msg)

        # router
        if self._control_router in socks and socks[self._control_router] == zmq.POLLIN:
            for identifier, msg in utils.iter_recv(utils.router_recv_json, self._control_router):
                self._control_dispatcher.dispatch(msg['action'], identifier, msg)

        for k in socks:
            if k in self._ipeers_csock and socks[k] == zmq.POLLIN:
                for msg in utils.iter_recv(utils.req_recv_json, k):
                    self._control_dispatcher.dispatch(msg['action'], msg)

    def _main_do_input(self, socks):
        for k in socks:
            if k in self._ipeers_dsock and socks[k] == zmq.POLLIN:
                if not self._data_recv_queue.full():
                    msg = utils.pull_pyobj(k)
                    if msg is not None:
                        self._data_recv_queue.put(msg)

    def _main_do_output(self):
        nr_send = self._data_send_queue.qsize()
        for i in range(nr_send):
            job = self._data_send_queue.get()
            pipe_name = job['pipe'].name
            succ = 0
            for k, (info, sock, flag) in self._opeers.items():
                if pipe_name in info['ipipes'] and flag:
                    succ += utils.push_pyobj(sock, job['payload'], flag=zmq.NOBLOCK)
            if succ > 0:
                self._opipes_enque_mark[job['pipe']] = True
            else:
                self._data_send_queue.put(job)

    def _main_do_opipe_enque(self):
        for name, pipes in self._opipes.items():
            for pipe in pipes:
                if self._opipes_enque_mark[pipe]:
                    self._do_data_enque(pipe)

    def _main_do_ipipe_enque(self):
        nr_recv = self._data_recv_queue.qsize()
        for i in range(nr_recv):
            msg = self._data_recv_queue.get()
            succ = 0
            for pipe in self._ipipes.get(msg['pipe_name'], []):
                succ += pipe.put_nowait(msg['data'])
            if succ == 0:
                self._data_recv_queue.put(msg)

    def _do_response_heartbeat(self, identifier, msg):
        logger.info('recv heartbeat from {}'.format(msg['uid']))
        utils.router_send_json(self._control_router, identifier, {
            'action': configs.Actions.CTL_HEARTBEAT_SUCC
        })

    def _do_setup_opeer(self, identifier, msg):
        self._opeers[msg['uid']] = record = [{
            'uid': msg['uid'],
            'ipipes': msg['ipipes']
        }, None, False]  # info, socket, established

        record[1] = self._context.socket(zmq.PUSH)
        port = record[1].bind_to_random_port('{}://{}'.format(configs.CTL_DAT_PROTOCAL, configs.CTL_DAT_HOST))
        record[1].set_hwm(configs.CTL_DAT_HWM)
        record[0]['port'] = port

        self._opeers_dsock.add(record[1])
        utils.router_send_json(self._control_router, identifier, {
            'action': configs.Actions.CTL_OPEN_RESPONSE,
            'uid': self._uid,
            'data_protocal': configs.CTL_DAT_PROTOCAL,
            'data_addr': self._addr,
            'data_port': port
        })
        logger.info('Connection opened for {}: pipe={}, port={}'.format(msg['uid'], msg['ipipes'], record[0]['port']))

    def _do_setup_opeer_finished(self, identifier, msg):
        self._opeers[msg['uid']][2] = True
        utils.router_send_json(self._control_router, identifier, {
            'action': configs.Actions.CTL_OPENED_SUCC
        })
        logger.info('Connection established for {}'.format(msg['uid']))

    def _do_setup_ipeer_csock(self, uid):
        record = self._ipeers[uid]

        if record[1] is not None:
            return

        info = record[0]
        record[1] = self._context.socket(zmq.REQ)
        record[1].connect('{}://{}:{}'.format(info['ctl_protocal'], info['ctl_addr'], info['ctl_port']))
        self._ipeers_csock.add(record[1])
        self._poller.register(record[1], zmq.POLLIN)
        self._control_send_queue.put({
            'sock': record[1],
            'countdown': configs.CTL_CTL_SEND_COUNTDOWN,
            'payload': {
                'action': configs.Actions.CTL_OPEN,
                'uid': self._uid,
                'ipipes': list(self._ipipes.keys())
            }
        })
        logger.info('Connecting to {}'.format(info['uid']))

    def _do_setup_ipeer_csock_response(self, msg):
        record = self._ipeers[msg['uid']]

        record[2] = self._context.socket(zmq.PULL)
        record[2].connect('{}://{}:{}'.format(msg['data_protocal'], msg['data_addr'], msg['data_port']))
        record[2].set_hwm(configs.CTL_DAT_HWM)
        self._ipeers_dsock.add(record[2])
        self._poller.register(record[2], zmq.POLLIN)

        self._control_send_queue.put({
            'sock': record[1],
            'countdown': configs.CTL_CTL_SEND_COUNTDOWN,
            'payload': {
                'action': configs.Actions.CTL_OPENED,
                'uid': self._uid
            }
        })
        logger.info('Connection established for {}'.format(record[0]['uid']))

    def _do_ns_open_ctl(self, identifier, msg):
        if msg['uid'] not in self._ipeers:
            self._ipeers[msg['uid']] = [msg['info'], None, None]  # info, control_socket, data_socket
            self._do_setup_ipeer_csock(msg['uid'])
        self._control_send_queue.put({
            'sock': self._control_router,
            'identifier': identifier,
            'payload': {
                'action': configs.Actions.NS_OPEN_CTL_SUCC,
                'uid': self._uid
            }
        })
        logger.info('Found new controller {}'.format(msg['uid']))

    def _do_ns_close_ctl(self, identifier, msg):
        uid = msg['uid']
        if uid in self._ipeers:
            record = self._ipeers.pop(uid)
            if record[1] is not None:
                self._poller.unregister(record[1])
                utils.graceful_close(record[1])
                self._ipeers_csock.remove(record[1])
            if record[2] is not None:
                self._poller.unregister(record[2])
                utils.graceful_close(record[2])
                self._ipeers_dsock.remove(record[2])
        if uid in self._opeers:
            record = self._opeers.pop(uid)
            if record[1] is not None:
                utils.graceful_close(record[1])
                self._opeers_dsock.remove(record[1])
        self._control_send_queue.put({
            'sock': self._control_router,
            'identifier': identifier,
            'payload': {
                'action': configs.Actions.NS_CLOSE_CTL_SUCC,
                'uid': self._uid
            }
        })
        logger.info('Close timeout controller {}'.format(msg['uid']))

    def _do_data_enque(self, pipe):
        data = pipe.get_nowait()
        if data is None:
            return
        self._opipes_enque_mark[pipe] = False

        payload = {
            'uid': self._uid,
            'pipe_name': pipe.name,
            'data': data
        }
        self._data_send_queue.put({
            'payload': payload,
            'pipe': pipe
        })


@contextlib.contextmanager
def control(pipes):
    ctl = Controller()
    ctl.initialize(pipes)
    yield ctl
    ctl.finalize()
