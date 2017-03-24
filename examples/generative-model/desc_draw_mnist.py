# -*- coding:utf8 -*-
# File   : desc_draw_mnist.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 3/22/17
# 
# This file is part of TensorArtist

from tartist.core import get_env, get_logger
from tartist.core.utils.naming import get_dump_directory, get_data_directory
from tartist.nn import opr as O, optimizer, summary

import tensorflow as tf
import draw_opr

logger = get_logger(__file__)

__envs__ = {
    'dir': {
        'root': get_dump_directory(__file__),
        'data': get_data_directory('WellKnown/mnist')
    },
    'trainer': {
        'learning_rate': 0.001,

        'batch_size': 100,
        'epoch_size': 500,
        'nr_epochs': 100,
    },
    'inference': {
        'batch_size': 256,
        'epoch_size': 40
    },
    'demo': {
        'is_reconstruct': True
    }
}


def make_network(env):
    with env.create_network() as net:
        h, w, c = 28, 28, 1
        nr_glimpse = 16
        att_dim = 5
        code_length = 128

        is_reconstruct = get_env('demo.is_reconstruct', False)

        dpc = env.create_dpcontroller()
        with dpc.activate():
            def inputs():
                img = O.placeholder('img', shape=(None, h, w, c))
                return [img]

            def forward(img):
                encoder = tf.contrib.rnn.BasicLSTMCell(256)
                decoder = tf.contrib.rnn.BasicLSTMCell(256)
                canvas = O.zeros_like(img)

                batch_size = img.shape[0]
                enc_state = encoder.zero_state(batch_size, dtype='float32')
                dec_state = decoder.zero_state(batch_size, dtype='float32')
                enc_h, dec_h = enc_state[1], dec_state[1]

                def encode(x, state, reuse):
                    with tf.variable_scope('read_encoder', reuse=reuse):
                        return encoder(O.as_tftensor(x), state)

                def decode(x, state, reuse):
                    with tf.variable_scope('write_decoder', reuse=reuse):
                        return decoder(O.as_tftensor(x), state)

                all_sqr_mus, all_vars, all_log_vars = 0., 0., 0.

                if is_reconstruct or env.phase is env.Phase.TRAIN:
                    for step in range(nr_glimpse):
                        reuse = (step != 0)
                        img_hat = draw_opr.image_diff(img, canvas)

                        with tf.variable_scope('read', reuse=reuse):
                            read_param = O.fc('fc_param', enc_h, 5)

                        with tf.name_scope('read_step{}'.format(step)):
                            cx, cy, delta, var, gamma = draw_opr.split_att_params(h, w, att_dim, read_param)
                            read_inp = O.concat([img, img_hat], axis=3)  # of shape: batch_size x h x w x (2c)
                            read_out = draw_opr.att_read(att_dim, read_inp, cx, cy, delta, var)
                            enc_inp = O.concat([gamma * read_out.flatten2(), dec_h], axis=1)
                        enc_h, enc_state = encode(enc_inp, enc_state, reuse)

                        with tf.variable_scope('sample', reuse=reuse):
                            _ = enc_h
                            sample_mu = O.fc('fc_mu', _, code_length)
                            sample_log_var = O.fc('fc_sigma', _, code_length)

                        with tf.name_scope('sample_step{}'.format(step)):
                            sample_var = O.exp(sample_log_var)
                            sample_std = O.sqrt(sample_var)
                            sample_epsilon = O.random_normal([batch_size, code_length])
                            z = sample_mu + sample_std * sample_epsilon

                        # z = O.callback_injector(z)

                        # accumulate for losses
                        all_sqr_mus += sample_mu ** 2.
                        all_vars += sample_var
                        all_log_vars += sample_log_var

                        dec_h, dec_state = decode(z, dec_state, reuse)
                        with tf.variable_scope('write', reuse=reuse):
                            write_param = O.fc('fc_param', dec_h, 5)
                            write_in = O.fc('fc', dec_h, (att_dim * att_dim * c)).reshape(-1, att_dim, att_dim, c)

                        with tf.name_scope('write_step{}'.format(step)):
                            cx, cy, delta, var, gamma = draw_opr.split_att_params(h, w, att_dim, write_param)
                            write_out = draw_opr.att_write(h, w, write_in, cx, cy, delta, var)

                        canvas += write_out

                    canvas = O.sigmoid(canvas)

                if env.phase is env.Phase.TRAIN:
                    with tf.variable_scope('loss'):
                        img, canvas = img.flatten2(), canvas.flatten2()
                        content_loss = O.raw_cross_entropy_prob('raw_content', canvas, img)
                        content_loss = content_loss.sum(axis=1).mean(name='content')
                        # distrib_loss = 0.5 * (O.sqr(mu) + O.sqr(std) - 2. * O.log(std + 1e-8) - 1.0).sum(axis=1)
                        distrib_loss = -0.5 * (float(nr_glimpse) + all_log_vars - all_sqr_mus - all_vars).sum(axis=1)
                        distrib_loss = distrib_loss.mean(name='distrib')

                        summary.scalar('content_loss', content_loss)
                        summary.scalar('distrib_loss', distrib_loss)

                        loss = content_loss + distrib_loss
                    dpc.add_output(loss, name='loss', reduce_method='sum')

                dpc.add_output(canvas, name='output')

            dpc.set_input_maker(inputs).set_forward_func(forward)

        net.add_all_dpc_outputs(dpc, loss_name='loss')

        if env.phase is env.Phase.TRAIN:
            summary.inference.scalar('loss', net.loss)


def make_optimizer(env):
    wrapper = optimizer.OptimizerWrapper()
    wrapper.set_base_optimizer(optimizer.base.AdamOptimizer(get_env('trainer.learning_rate'), beta1=0.75, beta2=0.5))
    wrapper.append_grad_modifier(optimizer.grad_modifier.LearningRateMultiplier([
        ('*/b', 2.0),
    ]))
    # wrapper.append_grad_modifier(optimizer.grad_modifier.WeightDecay([
    #     ('*/W', 0.0005)
    # ]))
    env.set_optimizer(wrapper)


from data_provider_vae_mnist import *


def main_train(trainer):
    from tartist.plugins.trainer_enhancer import summary
    summary.enable_summary_history(trainer)
    summary.enable_echo_summary_scalar(trainer)

    from tartist.plugins.trainer_enhancer import progress
    progress.enable_epoch_progress(trainer)

    from tartist.plugins.trainer_enhancer import snapshot
    snapshot.enable_snapshot_saver(trainer)

    # from tartist.plugins.trainer_enhancer import inference
    # inference.enable_inference_runner(trainer, make_dataflow_inference)

    trainer.train()
