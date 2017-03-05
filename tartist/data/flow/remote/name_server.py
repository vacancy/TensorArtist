# -*- coding:utf8 -*-
# File   : name_server.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 2/28/17
# 
# This file is part of TensorArtist

from . import configs, utils
from ....core.logger import get_logger
from ....core.utils.callback import CallbackManager
from ....core.utils.meta import notnone_property
from ....core.utils.cache import cached_property

import zmq
import time
import threading
import queue

logger = get_logger(__file__)


class NameServerControllerStorage(object):
    def __init__(self):
        self._all_peers = {}
        self._all_peers_req = {}
        self._ipipes = {}
        self._opipes = {}

    def register(self, info, req_sock):
        identifier = info['uid']
        assert identifier not in self._all_peers
        self._all_peers[identifier] = {
            'uid': info['uid'],
            'ctl_protocal': info['ctl_protocal'],
            'ctl_addr': info['ctl_addr'],
            'ctl_port': info['ctl_port'],
            'meta': info.get('meta', {}),
            'ipipes': [],
            'opipes': [],
            'last_heartbeat': time.time()
        }
        self._all_peers_req[identifier] = req_sock

    def register_pipes(self, info):
        controller = info['uid']
        assert controller in self._all_peers
        record = self._all_peers[controller]
        for i in record['ipipes']:
            self._ipipes.get(i, []).remove(controller)
        for i in record['opipes']:
            self._opipes.get(i, []).remove(controller)
        record['ipipes'] = info['ipipes']
        record['opipes'] = info['opipes']
        for i in record['ipipes']:
            self._ipipes.setdefault(i, []).append(controller)
        for i in record['opipes']:
            self._opipes.setdefault(i, []).append(controller)

    def unregister(self, identifier):
        if identifier in self._all_peers:
            info = self._all_peers.pop(identifier)
            for i in info['ipipes']:
                self._ipipes.get(i, []).remove(identifier)
            for i in info['opipes']:
                self._opipes.get(i, []).remove(identifier)
            return info, self._all_peers_req.pop(identifier)
        return None

    def get(self, identifier):
        return self._all_peers.get(identifier, None)

    def get_req_sock(self, identifier):
        return self._all_peers_req.get(identifier, None)

    def get_ipipe(self, name):
        return self._ipipes.get(name, [])

    def get_opipe(self, name):
        return self._opipes.get(name, [])

    def contains(self, identifier):
        return identifier in self._all_peers

    def all(self):
        return list(self._all_peers.keys())


class NameServer(object):
    def __init__(self):
        self.control_storage = NameServerControllerStorage()
        self.storage_lock = threading.Lock()
        self._context = zmq.Context()
        self._router = self._context.socket(zmq.ROUTER)
        self._poller = zmq.Poller()
        self._dispatcher = CallbackManager()
        self._req_socks = set()
        self._all_threads = list()
        self._control_send_queue = queue.Queue()

    def mainloop(self):
        self.initialize()
        try:
            self._all_threads.append(threading.Thread(target=self.main, name='name-server-main'))
            self._all_threads.append(threading.Thread(target=self.main_cleanup, name='name-server-cleanup'))
            for i in self._all_threads:
                i.start()
        finally:
            self.finalize()

    def initialize(self):
        addr = '{}://{}:{}'.format(configs.NS_CTL_PROTOCAL, configs.NS_CTL_HOST, configs.NS_CTL_PORT)
        self._router.bind(addr)
        self._poller.register(self._router, zmq.POLLIN)

        self._dispatcher.register(configs.Actions.NS_REGISTER_CTL, self.register_controller)
        self._dispatcher.register(configs.Actions.NS_REGISTER_OPIPE, self.register_pipes)
        self._dispatcher.register(configs.Actions.NS_QUERY_OPIPE, self.query_output_pipe)
        self._dispatcher.register(configs.Actions.NS_HEARTBEAT, self.response_heartbeat)
        self._dispatcher.register(configs.Actions.NS_OPEN_CTL_SUCC, lambda msg: None)
        self._dispatcher.register(configs.Actions.NS_CLOSE_CTL_SUCC, lambda msg: None)

    def finalize(self):
        for i in self._all_threads:
            i.join()

        for sock in self._req_socks:
            utils.graceful_close(sock)
        self._router.setsockopt(zmq.LINGER, 0)
        self._router.close()
        if not self._context.closed:
            self._context.destroy(0)

    def main_cleanup(self):
        while True:
            with self.storage_lock:
                now = time.time()
                for k in self.control_storage.all():
                    v = self.control_storage.get(k)

                    if (now - v['last_heartbeat']) > configs.NS_CLEANUP_WAIT:
                        info, req_sock = self.control_storage.unregister(k)
                        utils.graceful_close(req_sock)
                        self._req_socks.remove(req_sock)

                        # TODO:: use controller's heartbeat
                        all_peers_to_inform = set()
                        for i in info['ipipes']:
                            for j in self.control_storage.get_opipe(i):
                                all_peers_to_inform.add(j)
                        for i in info['opipes']:
                            for j in self.control_storage.get_ipipe(i):
                                all_peers_to_inform.add(j)
                        print('inform', all_peers_to_inform)

                        for peer in all_peers_to_inform:
                            self._control_send_queue.put({
                                'sock': self.control_storage.get_req_sock(peer),
                                'countdown': configs.CTL_CTL_SEND_COUNTDOWN,
                                    'payload': {
                                    'action': configs.Actions.NS_CLOSE_CTL,
                                    'uid': k
                                },
                            })
                        logger.info('Unregister timeout controller {}'.format(k))
            time.sleep(configs.NS_CLEANUP_WAIT)

    def main(self):
        while True:
            socks = dict(self._poller.poll(100))
            self._main_do_send()
            self._main_do_recv(socks)

    def _main_do_send(self):
        nr_send = self._control_send_queue.qsize()
        for i in range(nr_send):
            job = self._control_send_queue.get()
            rc = utils.req_send_json(job['sock'], job['payload'], flag=zmq.NOBLOCK)
            if not rc:
                job['countdown'] -= 1
                if job['countdown'] >= 0:
                    self._control_send_queue.put(job)
                else:
                    print('drop job: ', job)

    def _main_do_recv(self, socks):
        if self._router in socks and socks[self._router] == zmq.POLLIN:
            while True:
                identifier, msg = utils.router_recv_json(self._router)
                if msg is not None:
                    self._dispatcher.dispatch(msg['action'], identifier, msg)
                else:
                    break
        for k in socks:
            if k in self._req_socks and socks[k] == zmq.POLLIN:
                while True:
                    msg = utils.req_recv_json(k, flag=zmq.NOBLOCK)
                    if msg is not None:
                        self._dispatcher.dispatch(msg['action'], msg)
                    else:
                        break

    def register_controller(self, identifier, msg):
        with self.storage_lock:
            req_sock = self._context.socket(zmq.REQ)
            req_sock.connect('{}://{}:{}'.format(msg['ctl_protocal'], msg['ctl_addr'], msg['ctl_port']))
            self.control_storage.register(msg, req_sock)
            self._req_socks.add(req_sock)
            self._poller.register(req_sock)
        utils.router_send_json(self._router, identifier, {'action': configs.Actions.NS_REGISTER_CTL_SUCC})
        logger.info('Controller registered: {}'.format(msg['uid']))

    def register_pipes(self, identifier, msg):
        with self.storage_lock:
            self.control_storage.register_pipes(msg)

            all_peers_to_inform = set()
            for i in msg['opipes']:
                for j in self.control_storage.get_ipipe(i):
                    all_peers_to_inform.add(j)
            print('inform', all_peers_to_inform)
            for peer in all_peers_to_inform:
                self._control_send_queue.put({
                    'sock': self.control_storage.get_req_sock(peer),
                    'countdown': configs.CTL_CTL_SEND_COUNTDOWN,
                    'payload': {
                        'action': configs.Actions.NS_OPEN_CTL,
                        'uid': msg['uid'],
                        'info': self.control_storage.get(msg['uid'])
                    },
                })
        utils.router_send_json(self._router, identifier, {'action': configs.Actions.NS_REGISTER_OPIPE_SUCC})

        logger.info('Controller pipes registered: in={}, out={} (controller-uid={})'.format(
            msg['ipipes'], msg['opipes'], msg['uid']))

    def query_output_pipe(self, identifier, msg):
        res = {}
        with self.storage_lock:
            for name in msg['pipe_names']:
                all_pipes = self.control_storage.get_opipe(name)
                all_pipes = list(map(self.control_storage.get, all_pipes))
                res[name] = all_pipes

        utils.router_send_json(self._router, identifier, {
            'action': configs.Actions.NS_QUERY_OPIPE_RESPONSE,
            'results': res
        })

    def response_heartbeat(self, identifier, msg):
        with self.storage_lock:
            if self.control_storage.contains(msg['uid']):
                self.control_storage.get(msg['uid'])['last_heartbeat'] = time.time()
                print('Heartbeat {}: time={}'.format(msg['uid'], time.time()))
                utils.router_send_json(self._router, identifier, {
                    'action': configs.Actions.NS_HEARTBEAT_SUCC
                })
