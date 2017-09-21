# -*- coding:utf8 -*-
# File   : env.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/29/16
#
# This file is part of TensorArtist.

import enum
import threading
import contextlib
import tensorflow as tf

from .node import as_varnode
from .function import Function
from .tfqueue import InputQueueDesc, QueuedInputFunction
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
    """
    use tf.device to change the current device, it will use the #devid device in env

    :param devid: device id in env.all_devices, should be int
    :param env: the env
    :return: a tf.device context
    """
    if devid == 0:
        return tf.device(env.master_device)
    return tf.device(env.slave_devices[devid - 1])


def reuse_context(activate=True, name=None):
    """
    enable variable reuse context, without any name

    :param activate: whether use or not, this is useful when you have some conditional parameters to enable
    parameter reuse
    :param name: if provided, use as the name for variable_scope
    :return: if active, return a reuse variable scope, otherwise an empty context
    """
    name = name or tf.get_variable_scope()

    if activate:
        return tf.variable_scope(name, reuse=True)
    else:
        return EmptyContext()


def _on_train_flag(attr_name):
    """
    return a function that compute some flag

    1. if the current flag is manually set to be a bool, return that one
    2. if the current flag is a function, return func(name)
    3. if current phase is train, return True
    4. otherwise, return False
    """
    def compute(self, name):
        attr = getattr(self, attr_name)
        if attr is None:
            return get_default_env().phase is Env.Phase.TRAIN
        if callable(attr):
            e = get_default_env()
            return get_default_env().phase is Env.Phase.TRAIN and attr(e.get_name_scope())
        return bool(attr)
    return compute


class Env(object):
    """
    Env is a environment to perform network construction, this is actually similar to a Graph object
    in tensorflow.
    Each Env object actually contains one tf.Graph, and also a default tf.Session on that graph.

    One important reason to use Env is to perform data-parallel, and also hold some information for network
    construction (like env.phase). You can use env.set_master_device and env.set_slave_devices to set the device
    used by this env. It will then be used for data-parallel maker.

    Besides data-parallel controller, a nn.Network object is associated with each Env, representing the network.

    One important notice is that, this Env object has nothing to do with the core get_env, set_env series methods.
    Although they share the same naming, that one is used for environ variables holding (similar to environ in OS),
    but this one is used for NN construction.
    """

    class SessionFlag(AttrObject):
        """Env flags"""

        """log tensorflow device placement"""
        log_device_placement = False
        """allow tensorflow soft placement"""
        allow_soft_placement = True

        """tensorflow gpu option: aloocator type"""
        gpu_allocator_type = 'BFC'
        """tensorflow gpu option: allow mem growth"""
        gpu_allow_growth = True
        """tensorflow gpu option: mem fraction"""
        gpu_mem_fraction = 0.99

        """whether batch normalization state should be updated during the forwarding, this option can be either:

        1. None (default): it will return True when the current phase is TRAIN
        2. True or False
        3. A function (callback) takes a name to the operation and output True/False for this specific op
        """
        update_batch_normalization = None
        """whether to enable dropout during forwarding, similar to update_batch_normalization"""
        enable_dropout = None

        compute_update_batch_normalization = _on_train_flag('update_batch_normalization')
        compute_enable_dropout = _on_train_flag('enable_dropout')

        input_queue_size = 50

    class DataParallelFlag(AttrObject):
        """Env data parallel flags"""
        pass

    class Phase(enum.Enum):
        """Env phase: either TRAIN or TEST"""
        TRAIN = 1
        TEST = 2

    def __init__(self, phase=Phase.TEST, master_dev='/gpu:0', slave_devs=None, flags=None, dpflags=None,
                 graph=None, session=None, func_lock=None, sync_with=None):
        """

        :param phase: The phase, either TRAIN or TEST.
        :param master_dev: The master device, e.g. "/gpu:0".
        :param slave_devs: The slave devices, should be a list/tuple.
        :param flags: A Env.SessionFlag object.
        :param dpflags: A Env.DataParallelFlag object.
        :param graph: A tf.Graph object, if not provided, a default graph will be created. This parameter is useful if
        you want to share a tf.Graph amount different envs.
        :param session: A tf.Session object. Similarly, a default one be use if not provided.
        :param func_lock: A threading.Lock object.
        :param sync_with: Synchronize everything with another env: graph, session and func_lock
        """

        if sync_with is not None:
            assert graph is None and session is None and func_lock is None
            graph = sync_with.graph
            session = sync_with.session
            func_lock = sync_with.get_or_make_func_lock()

        self.__phase = phase
        self.__session = None
        self.__network = None
        self.__current_dpc = None

        self._master_device = master_dev
        self._slave_devices = slave_devs or []

        self._flags = flags or type(self).SessionFlag()
        self._dpflags = dpflags or type(self).DataParallelFlag()
        self._dpsplitters = []
        self._graph = graph or tf.Graph()
        self._func_lock = func_lock

        if session is not None:
            self.__session = session

        self._use_input_queue = False
        self._input_queue_desc = None

    @notnone_property
    def network(self):
        """Get the current network, raise Exception if the network hasn't yet been created."""
        return self.__network

    def has_current_dpc(self):
        return self.__current_dpc is not None

    @notnone_property
    def current_dpc(self):
        """Current data-parallel controller"""
        return self.__current_dpc

    @property
    def graph(self):
        """Return the tf.Graph associated with the env, it is OK to share a tf.Graph among several Envs."""
        return self._graph

    @contextlib.contextmanager
    def create_network(self):
        """Create the network, and activate it as the default network."""
        assert self.__network is None
        self.__network = Network(self)
        with self.__network.as_default():
            yield self.__network

    def create_dpcontroller(self):
        """Create a data parallel controller"""
        self.__current_dpc = DataParallelController(self)
        return self.__current_dpc

    def register_dpsplitter(self, splitter):
        """Register a data-parallel splitter for function, this will be internally called by
        data-parallel controller."""
        self._dpsplitters.append(splitter)

    @property
    def phase(self):
        return self.__phase

    @property
    def session(self):
        """Return the default session, if it hasn't been created, create it."""
        if self.__session is None:
            self._make_session()
        return self.__session

    def reuse_session(self, session):
        self.__session = session
        return self

    def _make_session(self):
        config = tf.ConfigProto()
        config.log_device_placement = self.flags.log_device_placement
        config.allow_soft_placement = self.flags.allow_soft_placement
        config.gpu_options.per_process_gpu_memory_fraction = self.flags.gpu_mem_fraction
        config.gpu_options.allocator_type = self.flags.gpu_allocator_type
        config.gpu_options.allow_growth = self.flags.gpu_allow_growth

        self.__session = tf.Session(graph=self._graph, config=config)

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
        """Select the #devid device as the current one, the returned object is a context manager by tf.device."""
        return select_device(devid, self)

    @contextlib.contextmanager
    def use_input_queue(self):
        """Return a context manager to enable input queue. the context manager should be activated during the network
        construction, like::

            with env.as_default(), env.use_input_queue():
                make_network(env)

        """
        assert not self._use_input_queue, 'use_input_queue can only be activated once'
        yield
        self._use_input_queue = True
        self._input_queue_desc = InputQueueDesc(self)
        self._input_queue_desc.setup(self._graph)

        # session will be rebuild
        if self.__session is not None:
            self._make_session()

    @notnone_property
    def input_queue_desc(self):
        """Return the input queue description."""
        return self._input_queue_desc

    @defaults_manager.wrap_custom_as_default
    def as_default(self, *, activate_session=True):
        with self._graph.as_default():
            assert tf.get_default_graph() == self._graph
            if activate_session:
                with self.session.as_default():
                    yield
            else:
                yield

    def get_or_make_func_lock(self):
        if self._func_lock is None:
            self._func_lock = threading.Lock()
        return self._func_lock

    def share_func_lock_with(self, env):
        self._func_lock = env.get_or_make_func_lock()
        return self

    def with_func_lock(self):
        if self._func_lock is None:
            return EmptyContext()
        return self._func_lock

    def make_func(self):
        """Make a function associated with this Env, and register the data-parallel splitters."""
        if self._use_input_queue:
            f = QueuedInputFunction(self)
        else:
            f = Function(self)
        f.extend_extra_kw_modifiers(self._dpsplitters)
        return f

    def initialize_all_variables(self):
        """Initialize all global variables."""
        sess = self.session
        sess.run(tf.variables_initializer(self._graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))

    def add_queue_runner(self, qr):
        """Add queue runner."""
        with self.graph.as_default():
            tf.train.add_queue_runner(qr)

    def initialize_all_queues(self):
        """Start all queue runners."""
        sess = self.session
        with self.graph.as_default():
            tf.train.start_queue_runners(sess=sess)

    def clone(self, share_graph=True, share_session=True, share_func_lock=False):
        """
        Clone this graph.

        :param share_graph: Whether the new env will share the graph with this.
        :param share_session: Whether the new env will share the session with this.
        :param share_func_lock: Whether the new env will share the func_lock with this.
        """
        return type(self)(self.phase, master_dev=self.master_device, slave_devs=self.slave_devices,
                          flags=self.flags, dpflags=self.dpflags,
                          graph=self.graph if share_graph else None,
                          session=self.session if share_session else None,
                          func_lock=self.get_or_make_func_lock() if share_func_lock else None)

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
        """Perform session.run use the default session."""
        return self.session.run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata)

    def find_opr_by_name(self, name):
        """Find opr by name."""
        return self.graph.get_operation_by_name(name)

    def find_var_by_name(self, name):
        """
        Find tensor (accurately, varnode) by name. The suffix ":0" can be automatically inferred.

        :param name: The name to the var, ":0" suffix can be ignored
        :return The varnode with given name.
        """
        try:
            return as_varnode(self.graph.get_tensor_by_name(name))
        except ValueError:
            return as_varnode(self.graph.get_tensor_by_name(name + ':0'))

    def add_to_collection(self, name, value):
        return self.graph.add_to_collection(name, value)

    def add_to_collections(self, names, value):
        return self.graph.add_to_collections(names, value)

    def get_collection(self, name, scope=None):
        return self.graph.get_collection(name, scope=scope)

    def get_collection_ref(self, name):
        return self.graph.get_collection_ref(name)

    def get_all_collection_keys(self):
        return self.graph.get_all_collection_keys()

    def find_in_collection_by_name(self, collection_or_key, name):
        """
        Find a op/tensor by name in a collection.
        In tensor case, the suffix ":0" can be automatically inferred

        :param collection_or_key: A collection (a list) or a string as the key to the collection.
        :param name: The name to the op/tensor you want to find.
        :return: The collection element with the given name.
        """
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

    def name_scope(self, name, default_name=None):
        if name is None:
            name = default_name
        return self.graph.name_scope(name)

    def get_name_scope(self, use_name=None):
        """Get current name scope"""
        if use_name:
            name = self.graph.unique_name(use_name, mark_as_used=False)
            return name

        random_str = 'mrm_msh'
        name = self.graph.unique_name(random_str, mark_as_used=False)
        if len(name) > len(random_str):
            return name[:-(len(random_str) + 1)]
        return ''

    def variable_scope(self, name_or_scope, default_name=None, reuse=None):
        with self.graph.as_default():
            return tf.variable_scope(name_or_scope=name_or_scope, default_name=default_name, reuse=reuse)

    def get_variable_scope(self):
        with self.graph.as_default():
            return tf.get_variable_scope()

    def reuse_scope(self, activate=True):
        with self.graph.as_default():
            return reuse_context(activate)

    def get_unique_name(self, scope_name):
        return self.graph.unique_name(scope_name, mark_as_used=False)

    def get_pure_unique_name(self, scope_name):
        prefix = self.get_name_scope()
        scope_name = self.graph.unique_name(scope_name, mark_as_used=False)
        if len(prefix) == 0:
            return scope_name
        return scope_name[len(prefix)+1:]

get_default_env = defaults_manager.gen_get_default(Env)


class DataParallelController(object):
    """
    A data parallel controller letting you perform automatically data-parallel during network construction.
    To do so, you need to make two functions: make_input, and forward. A typical usage is::

        dpc = DataParallelController(env)
        with dpc.active():
            def make_input():
                a = O.placeholder('a')
                return [a]

            def forward(a):
                dpc.add_output(a + 1, name='output', reduce_method='concat)

            dpc.set_input_maker(make_input).set_forward_func(forward)

        # ... then you can get the reduced output by:

        dpc.outputs['output']

    Technically, the dpc will make n (n is the number of devices you set in env) different towers, when constructing
    the i-th tower (residing on i-th device), it will call once input_maker, and call once forward_func by passing
    the input tensors returned by input_maker to the forward_func.

    In forward function, you should call dpc.add_output to state that the specific output will be outputed (and then
    reduced to the master device). Note that you can point the reduction method for each output by using the
    parameter reduce_method when you call dpc.add_output.

    Then, after the activate context, it will collect all towers' outputs, and reduce them to the first device (
    tower 0, i.e. the tower for master_device), and provide access to then by dpc.outputs[name]
    """

    def __init__(self, owner_env, name_scope_prefix='tower'):
        """

        :param owner_env: The owner env.
        :param name_scope_prefix: The name scope prefix: like "tower", it will result in multiple towers naming
        "tower/0", "tower/1", etc.
        """
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

        self._name_scope_prefix = owner_env.get_pure_unique_name(name_scope_prefix)
        self._real_name_scope_prefix = owner_env.get_name_scope(name_scope_prefix)
        self._tower_prefixes = []

    @property
    def owner_env(self):
        """Owner env"""
        return self.__owner_env

    def set_input_maker(self, maker):
        """Set the input maker. See  __init__ for detail."""
        self._input_maker = maker
        return self

    def set_forward_func(self, func):
        """Set the forward func. See __init__ for detail."""
        self._forward_func = func
        return self

    @property
    def outputs(self):
        return self._outputs

    def add_output(self, symbol, name=None, reduce_method='concat'):
        """
        Add an output to dpc.

        :param symbol: The tensor you want to output.
        :param name: The name to the tensor, if None, will use symbol.name.
        :param reduce_method: Either 'concat' or 'sum', used for reducing.
        """
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

            name_prefix = self.owner_env.get_pure_unique_name('{}/{}'.format(self._name_scope_prefix, i))
            self._tower_prefixes.append(name_prefix)
            with tf.name_scope(name_prefix), select_device(i, self.owner_env), reuse_context(i != 0):
                inputs = self._input_maker()

                if i == 0:
                    for v in inputs:
                        vname = v.name
                        assert vname.startswith(self._real_name_scope_prefix) and vname.endswith(':0'), vname
                        self._input_names.append(vname[len(self.owner_env.get_name_scope())+1:-2])

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
                
                for i in range(self._nr_towers):
                    subname = self._tower_prefixes[i] + '/' + name
                    kwargs[subname] = value[i]

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
    """
    A network is a data structure to hold a neural network's structure, like outputs and loss.
    At the same time, it also provide some utility functions related to graph structure.
    """
    def __init__(self, owner_env):
        """
        :param owner_env: The owner env.
        """
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
        """Get he merged summary in the default collection."""
        return self.get_merged_summaries()

    def get_merged_summaries(self, collection='summaries'):
        """Return the merged summary from the given collection."""
        with self.owner_env.graph.as_default():
            return tf.summary.merge_all(key=collection)

    def add_output(self, symbol, name=None):
        """Add an output to the network."""
        symbol = as_varnode(symbol)
        name = name or clean_name(symbol)
        self.__outputs[name] = symbol
        return self

    def add_all_dpc_outputs(self, dpc, loss_name='loss'):
        """
        A utility function to add all outputs in a data-parallel controller's outputs to the networks' output dict.
        Note that if you given the loss_name parameter, the output named with that name will be treated as the loss
        for the network (thus calling net.set_loss).

        :param dpc: A data parallel controller.
        :param loss_name: The loss name.
        :return: self
        """
        for k, v in dpc.outputs.items():
            if k == loss_name:
                self.set_loss(v)
            else:
                self.add_output(v, name=k)
        return self

    def fetch_all_variables_dict(self):
        """Get all variables as a dict."""
        from ..tfutils import fetch_variables

        all_variables = {}
        var_list = self.owner_env.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        value_list = fetch_variables(var_list, self.owner_env.session)
        all_variables = {clean_name(var): value for var, value in zip(var_list, value_list)}
        return all_variables

    def assign_all_variables_dict(self, all_variables, verbose=True):
        """Assign all variables from a dict."""
        from ..tfutils import assign_variables

        var_list, value_list = [], []
        for v in self.owner_env.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            value = all_variables.get(clean_name(v), None)
            if value is not None:
                if verbose:
                    logger.info('Assign variable from external dict: {}.'.format(clean_name(v)))
                var_list.append(v)
                value_list.append(value)

        assign_variables(var_list, value_list, self.owner_env.session)
        return self

    def find_opr_by_name(self, name):
        """Alias for env.find_opr_by_name."""
        return self.owner_env.find_opr_by_name(name)

    def find_var_by_name(self, name):
        """Alias for env.find_var_by_name."""
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
    _default_graph = _default_env.graph

    # this is a hack to add the default graph to the top of the graph stack inside tensorflow
    from tensorflow.python.framework import ops
    ops._default_graph_stack.stack.append(_default_graph)
