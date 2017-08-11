# -*- coding:utf8 -*-
# File   : test_nn_multi_predictor.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/19/17
# 
# This file is part of TensorArtist.

import numpy as np
import tensorflow as tf

from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import Env, opr as O, optimizer, summary

logger = get_logger(__file__)

__envs__ = {

}


def make_network(env):
    with env.create_network() as net:
        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                h, w, c = 28, 28, 1
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def forward(img):
                _ = img
                _ = O.conv2d('conv1', _, 16, (3, 3), padding='SAME', nonlin=O.relu)
                _ = O.pooling2d('pool1', _, kernel=2)
                _ = O.conv2d('conv2', _, 32, (3, 3), padding='SAME', nonlin=O.relu)
                _ = O.pooling2d('pool2', _, kernel=2)
                dpc.add_output(_, name='feature')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        _ = dpc.outputs['feature']
        _ = O.fc('fc1', _, 64)
        _ = O.fc('fc2', _, 10)

        # it's safe to use tf.xxx and O.xx together
        prob = O.softmax(_, name='prob')
        pred = _.argmax(axis=1).astype(tf.int32, name='pred')
        net.add_output(prob)
        net.add_output(pred)

        if env.phase is env.Phase.TRAIN:
            label = O.placeholder('label', shape=(None, ), dtype=tf.int32)
            loss = O.sparse_softmax_cross_entropy_with_logits(logits=_, labels=label).mean()
            loss = O.identity(loss, name='loss')
            net.set_loss(loss)

            accuracy = O.eq(label, pred).astype('float32').mean()
            error = 1. - accuracy

            summary.scalar('accuracy', accuracy)
            summary.scalar('error', error)
            summary.inference.scalar('loss', loss)
            summary.inference.scalar('accuracy', accuracy)
            summary.inference.scalar('error', error)


def main():
    env = Env(master_dev='/cpu:0')
    with env.as_default():
        make_network(env)
    fs = []
    for i in range(5):
        new_env = Env(master_dev='/cpu:0', flags=env.flags, dpflags=env.dpflags, graph=env.graph)
        with new_env.as_default():
            with tf.name_scope('predictor/{}'.format(i)):
                with env.variable_scope(tf.get_variable_scope(), reuse=True):
                    make_network(new_env)
            f = new_env.make_func()
            f.compile(outputs=new_env.network.outputs)
            fs.append(f)
    img = np.zeros(shape=(1, 28, 28, 1))
    from IPython import embed; embed()


if __name__ == '__main__':
    main()
