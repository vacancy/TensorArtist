# -*- coding:utf8 -*-
# File   : _web_handlers.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 28/07/2017
# 
# This file is part of TensorArtist.

import json
from tornado.web import RequestHandler


class HPHandlerBase(RequestHandler):
    _collector = None
    _loader = None
    _configs = None

    def initialize(self, collector=None, loader=None, configs=None):
        self._collector = collector
        self._loader = loader
        self._configs = configs

    @property
    def _pool(self):
        return self._collector.pool


class MainHandler(HPHandlerBase):
    def get(self):
        self.write(self._loader.load('index.html').generate(
            site_title=self._configs['title'],
            site_author=self._configs['author'],
        ))


class GetHandler(HPHandlerBase):
    def get(self):
        uid, info = self._pool.pop()

        if uid is not None:
            self.write(json.dumps({
                'rc': 200,
                'id': uid,
                'traj1': '<img src="trajectories/{}/1.gif" />'.format(uid),
                'traj2': '<img src="trajectories/{}/2.gif" />'.format(uid),
            }))
        else:
            self.write(json.dumps({'rc': 404}))


class SubmitHandler(HPHandlerBase):
    def post(self):
        uid = self.get_argument('id')
        pref = self.get_argument('pref')
        self.write('Received: id={}, pref={}.'.format(uid, pref))
        self._collector.post_preference(uid, pref)

