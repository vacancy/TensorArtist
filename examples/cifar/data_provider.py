# -*- coding:utf8 -*-
# File   : data_provider.py
# Author : Jiayuan Mao
#          Honghua Dong
# Email  : maojiayuan@gmail.com
#          dhh19951@gmail.com
# Date   : 2/27/17
#
# This file is part of TensorArtist.

from tartist import image
from tartist.core import get_env
from tartist.data import flow
from tartist.data.datasets.cifar import load_cifar

import numpy as np

_cifar = []
_cifar_labels_name = {
    10: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    # TODO:: set the label_name
    100: list(range(100))
}
_cifar_img_dim = 32


def ensure_load(cifar_num_classes):
    global _cifar

    if len(_cifar) == 0:
        for xy in load_cifar(get_env('dir.data'), cifar_num_classes):
            _cifar.append(dict(img=xy[0].astype('float32').reshape(-1, _cifar_img_dim, _cifar_img_dim, 3), label=xy[1]))


def make_dataflow_train(env):
    num_classes = get_env('dataset.nr_classes')
    ensure_load(num_classes)
    batch_size = get_env('trainer.batch_size')

    df = _cifar[0]
    df = flow.DOARandomSampleDataFlow(df)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'img': np.empty(shape=(batch_size, _cifar_img_dim, _cifar_img_dim, 3), dtype='float32'),
        'label': np.empty(shape=(batch_size, ), dtype='int32')
    })

    return df


def make_dataflow_inference(env):
    num_classes = get_env('dataset.nr_classes')
    ensure_load(num_classes)
    batch_size = get_env('inference.batch_size')
    epoch_size = get_env('inference.epoch_size')

    df = _cifar[1]  # use validation set actually
    df = flow.DictOfArrayDataFlow(df)
    df = flow.tools.cycle(df)
    df = flow.BatchDataFlow(df, batch_size, sample_dict={
        'img': np.empty(shape=(batch_size, _cifar_img_dim, _cifar_img_dim, 3), dtype='float32'),
        'label': np.empty(shape=(batch_size, ), dtype='int32')
    })
    df = flow.EpochDataFlow(df, epoch_size)

    return df


def make_dataflow_demo(env):
    num_classes = get_env('dataset.nr_classes')
    ensure_load(num_classes)

    # return feed_dict, extra_info
    def split_data(img, label):
        return dict(img=img[np.newaxis].astype('float32')), dict(label=label)

    df = _cifar[1]  # use validation set actually
    df = flow.DictOfArrayDataFlow(df)
    df = flow.tools.cycle(df)
    df = flow.tools.ssmap(split_data, df)

    return df


def demo(feed_dict, result, extra_info):
    nr_classes = get_env('dataset.nr_classes')

    img = feed_dict['img'][0]
    label = extra_info['label']
    labels_name = _cifar_labels_name[nr_classes]

    img = img.astype('uint8')
    img = image.resize_minmax(img, 256)
    outputs = [img, np.zeros(shape=[50, 256, 3], dtype='uint8')]
    outputs = np.vstack(outputs)

    text = 'Pred: {}'.format(labels_name[result['pred'][0]])
    text += ' Gt: {}'.format(labels_name[int(label)])
    # cv2.putText(outputs, text, (20, 256 + 25), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255))
    print(text)

    image.imshow('demo', outputs)
