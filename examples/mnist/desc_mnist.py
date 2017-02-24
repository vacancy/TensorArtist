# -*- coding:utf8 -*-
# File   : desc_mnist.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 12/30/16
# 
# This file is part of TensorArtist

import tensorflow as tf
import numpy as np
import numpy.random as npr
from tartist.core.logger import get_logger
from tartist.nn import Env, opr, train, optimizer
from tartist.data import flow

logger = get_logger(__file__)


def make_network(env):
    with env.create_network() as net:

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                h, w, c = 28, 28, 1
                img = opr.placeholder('img', shape=(None, h, w, c))
                return [img]

            def forward(img):
                _ = img
                _ = opr.conv2d('conv1', _, 16, (3, 3), padding='SAME', nonlin=tf.nn.relu)
                _ = opr.pooling2d('pool1', _, kernel=2)
                _ = opr.conv2d('conv2', _, 32, (3, 3), padding='SAME', nonlin=tf.nn.relu)
                _ = opr.pooling2d('pool2', _, kernel=2)
                dpc.add_output(_, name='feature')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        _ = dpc.outputs['feature']
        _ = opr.fc('fc1', _, 64)
        _ = opr.fc('fc2', _, 10)

        # it's safe to use tf.xxx and opr.xx together
        prob = tf.nn.softmax(_, name='prob')
        pred = tf.argmax(_, axis=1, name='pred')
        net.add_output(prob, name='prob')
        net.add_output(pred, name='pred')

        if env.phase is env.Phase.TRAIN:
            label = opr.placeholder('label', shape=(None, ), dtype=tf.int32)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(_, label))
            loss = tf.identity(loss, name='loss')
            net.set_loss(loss)


def make_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.SGDOptimizer(0.1))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    env.set_optimizer(wrapper)


def make_dataflow(env):
    global data_fake

    data_fake = dict(img=npr.uniform(size=(16, 28, 28, 1)), label=npr.randint(0, 10, size=16))
    batch_size = 32

    df = flow.DictOfArrayDataFlow(data_fake)
    df = flow.tools.cycle(df)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'img': np.empty(shape=(batch_size, 28, 28, 1), dtype='float32'),
        'label': np.empty(shape=(batch_size, ), dtype='int32')
    })
    return df

def main():
    env = train.TrainerEnv(Env.Phase.TRAIN, '/cpu:0')
    env.set_slave_devices(['/cpu:0', '/cpu:0'])
    with env.as_default():
        make_network(env)
        make_optimizer(env)

    # debug outputs
    for k, s in env.network.get_all_collections().items():
        names = [k] + sorted(['\t{}'.format(v.name) for v in s])
        logger.info('\n'.join(names))

    env.initialize_all_variables()
    with env.session.as_default():
        net = env.network

        f = env.make_optimizable_func()
        f.compile(net.loss)
        
        dfit = iter(make_dataflow(env))
        for i in range(100):
            this_batch = next(dfit)
            l = f(**this_batch)
            logger.info('iter={} loss={}'.format(i, l))

        f = env.make_func()
        f.compile(net.outputs['pred'])
        logger.info('final pred  ={}'.format(f(img=data_fake['img'])))
        logger.info('ground truth={}'.format(data_fake['label']))


if __name__ == '__main__':
    main()

