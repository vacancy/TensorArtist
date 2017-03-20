# -*- coding:utf8 -*-
# File   : env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
#
# This file is part of TensorArtist

import enum
import contextlib
import tensorflow as tf

from .node import as_varnode
from .function import Function
from ..tfutils import clean_name
from ...core.logger import get_logger
from ...core.utils.defaults import defaults_manager
from ...core.utils.context import EmptyContext
from ...core.utils.meta import assert_notnone, notnone_property, AttrObject
from ...core.utils.nd import nd_split_n

logger = get_logger(__file__)

__all__ = [
    'select_device', 'reuse_context',
    'Env', 'get_default_env',
    'Network', 'get_default_net',
    'DataParallelController'
]


def select_device(devid, env):
    if devid == 0:
        return tf.device(env.master_device)
    return tf.device(env.slave_devices[devid - 1])


def reuse_context(activate=True):
    if activate:
        return tf.variable_scope(tf.get_variable_scope(), reuse=True)
    else:
        return EmptyContext()


def _on_train_flag(attr_name):
    def compute(self, name):
        attr = getattr(self, attr_name)
        if attr is None:
            return get_default_env().phase is Env.Phase.TRAIN
        if callable(attr):
            return attr(name)
        return bool(attr)
    return compute


class Env(object):
    class SessionFlag(AttrObject):
        log_device_placement = False
        allow_soft_placement = True

        gpu_allocator_type = 'BFC'
        gpu_allow_growth = True
        gpu_mem_fraction = 0.99

        update_batch_normalization = None
        enable_dropout = None

        compute_update_batch_normalization = _on_train_flag('update_batch_normalization')
        compute_enable_dropout = _on_train_flag('enable_dropout')

    class DataParallelFlag(AttrObject):
        pass

    class Phase(enum.Enum):
        TRAIN = 1
        TEST = 2

    def __init__(self, phase=Phase.TEST, master_dev='/gpu:0', flags=None, dpflags=None, graph=None, session=None):
        self.__phase = phase
        self.__session = None
        self.__network = None
        self.__current_dpc = None

        self._master_device = master_dev
        self._slave_devices = []

        self._flags = flags or type(self).SessionFlag()
        self._dpflags = dpflags or type(self).DataParallelFlag()
        self._dpsplitters = []
        self._graph = graph or tf.Graph()

        if session is not None:
            self.__session = session

    @notnone_property
    def network(self):
        return self.__network

    @notnone_property
    def current_dpc(self):
        return self.__current_dpc

    @property
    def graph(self):
        return self._graph

    @contextlib.contextmanager
    def create_network(self):
        assert self.__network is None
        self.__network = Network(self)
        with self.__network.as_default():
            yield self.__network

    def create_dpcontroller(self):
        self.__current_dpc = DataParallelController(self)
        return self.__current_dpc

    def register_dpsplitter(self, splitter):
        self._dpsplitters.append(splitter)

    @property
    def phase(self):
        return self.__phase

    @property
    def session(self):
        if self.__session is None:
            config = tf.ConfigProto()
            config.log_device_placement = self.flags.log_device_placement
            config.allow_soft_placement = self.flags.allow_soft_placement
            config.gpu_options.per_process_gpu_memory_fraction = self.flags.gpu_mem_fraction
            config.gpu_options.allocator_type = self.flags.gpu_allocator_type
            config.gpu_options.allow_growth = self.flags.gpu_allow_growth

            self.__session = tf.Session(config=config)
        return self.__session

    @property
    def flags(self):
        return self._flags

    @property
    def dpflags(self):
        return self._dpflags

    @property
    def master_device(self):
        return self._master_device

    def set_master_device(self, dev):
        self._master_device = dev
        return self

    @property
    def slave_devices(self):
        return self._slave_devices

    @property
    def nr_slave_devices(self):
        return len(self._slave_devices)

    def set_slave_devices(self, devs):
        self._slave_devices = list(devs)
        return self

    @property
    def all_devices(self):
        res = [self.master_device]
        res.extend(self.slave_devices)
        return res

    @property
    def nr_total_devices(self):
        return 1 + len(self._slave_devices)

    def select_device(self, devid):
        return select_device(devid, self)

    @defaults_manager.wrap_custom_as_default
    def as_default(self, *, activate_session=True):
        with self._graph.as_default():
            assert tf.get_default_graph() == self._graph
            if activate_session:
                with self.session.as_default():
                    yield
            else:
                yield

    def make_func(self):
        f = Function(self)
        f.extend_extra_kw_modifiers(self._dpsplitters)
        return f

    def initialize_all_variables(self):
        sess = self.session
        sess.run(tf.variables_initializer(self._graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        return self.session.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    def find_opr_by_name(self, name):
        return self.graph.get_operation_by_name(name)

    def find_var_by_name(self, name):
        try:
            return as_varnode(self.graph.get_tensor_by_name(name))
        except ValueError:
            return as_varnode(self.graph.get_tensor_by_name(name + ':0'))

    def find_in_collection_by_name(self, collection_or_key, name):
        if type(collection_or_key) is str:
            collection_or_key = self.graph.get_collection(collection_or_key)
        for v in collection_or_key:
            if v.name == name:
                return v
        name += ':0'
        for v in collection_or_key:
            if v.name == name:
                return v
        return None

    def get_name_scope(self, use_name=None):
        if use_name:
            name = self.graph.unique_name(use_name, mark_as_used=False)
            return name

        random_str = 'mrm_msh'
        name = self.graph.unique_name(random_str, mark_as_used=False)
        if len(name) > len(random_str):
            return name[-(len(random_str) + 1)]
        return ''

get_default_env = defaults_manager.gen_get_default(Env)


class DataParallelController(object):
    def __init__(self, owner_env, name_scope_prefix='tower'):
        self.__owner_env = owner_env
        self.__activated = False

        self._input_maker = (lambda: [])
        self._forward_func = None

        self._input_names = []
        self._outputs = dict()
        self._output_reduce_methods = dict()
        self._all_outputs = None

        self._nr_towers = 1
        self._current_tower = 0

        self._name_scope_prefix = name_scope_prefix
        self._real_name_scope_prefix = owner_env.get_name_scope(name_scope_prefix)

    @property
    def owner_env(self):
        return self.__owner_env

    def set_input_maker(self, maker):
        self._input_maker = maker
        return self

    def set_forward_func(self, func):
        self._forward_func = func
        return self

    @property
    def outputs(self):
        return self._outputs

    def add_output(self, symbol, name=None, reduce_method='concat'):
        assert reduce_method in ('concat', 'sum')

        symbol = as_varnode(symbol)
        name = name or clean_name(symbol)
        self._outputs[name] = symbol
        self._output_reduce_methods[name] = reduce_method

    @contextlib.contextmanager
    def activate(self):
        assert not self.__activated

        self.__activated = True
        yield
        assert_notnone(self._forward_func, 'dpctl.forward_func')

        self._data_parallel()
        self.owner_env.register_dpsplitter(self._split)

    def _data_parallel(self):
        self._nr_towers = self.owner_env.nr_total_devices

        outputs = dict()

        for i in range(self._nr_towers):
            self._outputs = dict()
            self._current_tower = i

            name_prefix = '{}/{}'.format(self._name_scope_prefix, i)
            with tf.name_scope(name_prefix), select_device(i, self.owner_env), reuse_context(i != 0):
                inputs = self._input_maker()

                if i == 0:
                    for v in inputs:
                        vname = v.name
                        assert vname.startswith(self._real_name_scope_prefix) and vname.endswith(':0'), vname
                        self._input_names.append(vname[len(self._real_name_scope_prefix) + 3:-2])

                self._forward_func(*inputs)

            for k, v in self._outputs.items():
                outputs.setdefault(k, [])
                outputs[k].append(v)

        self._all_outputs = outputs

        if self._nr_towers > 1:
            self._outputs = dict()
            with tf.device(self.owner_env.master_device):
                for k, vs in outputs.items():
                    self._outputs[k] = self._reduce(k, vs)
        else:
            self._outputs = {k: v[0] for k, v in outputs.items()}

    def _reduce(self, name, values):
        from .. import opr as O

        meth = self._output_reduce_methods[name]
        if meth == 'concat':
            return O.concat(values, 0, name=name)
        elif meth == 'sum':
            return O.truediv(O.add_n(values), float(self._nr_towers), name=name)

    def _split(self, kwargs):
        assert self.__activated

        for name in self._input_names:
            if name in kwargs:
                value = kwargs.pop(name)

                if type(value) is list and len(value) == self._nr_towers:
                    pass  # directly use
                else:
                    value = nd_split_n(value, self._nr_towers)

                pattern = self._name_scope_prefix + '/{}/' + name
                kwargs.update({pattern.format(i): value[i] for i in range(self._nr_towers)})

    @property
    def current_tower(self):
        return self._current_tower

    @property
    def is_master_device(self):
        return self._current_tower == 0

    @property
    def is_slave_device(self):
        return self._current_tower != 0


class Network(object):
    def __init__(self, owner_env):
        self.__owner_env = owner_env

        self.__outputs = dict()
        self.__loss = None

    @property
    def owner_env(self):
        return self.__owner_env

    @notnone_property
    def loss(self):
        return self.__loss

    def set_loss(self, loss):
        loss = as_varnode(loss)
        self.__loss = loss
        return self

    @property
    def outputs(self):
        return self.__outputs

    @property
    def merged_summaries(self):
        return self.get_merged_summaries()

    def get_merged_summaries(self, collection='summaries'):
        with self.owner_env.graph.as_default():
            return tf.summary.merge_all(key=collection)

    def add_output(self, symbol, name=None):
        symbol = as_varnode(symbol)
        name = name or clean_name(symbol)
        self.__outputs[name] = symbol
        return self

    def add_all_dpc_outputs(self, dpc, loss_name='loss'):
        for k, v in dpc.outputs.items():
            if k == loss_name:
                self.set_loss(v)
            else:
                self.add_output(v, name=k)
        return self

    def fetch_all_variables_dict(self):
        from ..tfutils import fetch_variable

        all_variables = {}
        for v in self.owner_env.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            all_variables[clean_name(v)] = fetch_variable(v, self.owner_env.session)
        return all_variables

    def assign_all_variables_dict(self, all_variables, verbose=True):
        from ..tfutils import assign_variable

        for v in self.owner_env.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            value = all_variables.get(clean_name(v), None)
            if value is not None:
                if verbose:
                    logger.info('Assign variable from external dict: {}'.format(clean_name(v)))
                assign_variable(v, value, self.owner_env.session)
        return self

    def find_opr_by_name(self, name):
        return self.owner_env.find_opr_by_name(name)

    def find_var_by_name(self, name):
        return self.owner_env.find_var_by_name(name)

    @defaults_manager.wrap_custom_as_default
    def as_default(self):
        yield

get_default_net = defaults_manager.gen_get_default(Network)


# TODO(mjy):: add environ to control this
if True:
    _default_env = Env(master_dev='/cpu:0')
    defaults_manager.set_default(Env, _default_env)
    with _default_env.create_network() as _default_net:
        pass
    _default_net = _default_env.network
    defaults_manager.set_default(Network, _default_net)

