import chainer
from chainer import datasets
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
from chainer import Variable

import numpy as np
from tqdm import tqdm

import argparse
import os, sys

from .sobamchan_iterator import Iterator
from .sobamchan_log import Log
from .sobamchan_dataset import MovieReview
from .sobamchan_vocabulary import Vocabulary
from .sobamchan_utility import Utility
from .sobamchan_chainer import Model
from .sobamchan_chainer_link import PreTrainedEmbedId
from .sobamchan_vocabulary import Vocabulary
util = Utility()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', dest='bs', default=128, type=int)
    parser.add_argument('--epoch', dest='epoch', default=1, type=int)
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int)
    parser.add_argument('--output-dirname', dest='output_dirname', required=True)
    parser.add_argument('--w2v-path', dest='w2v_path', required=True)

    return parser.parse_args()

def train(opts):
    
    args = get_args()

    bs = args.bs
    epoch = args.epoch
    gpu = args.gpu
    output_dirname = args.output_dirname
    w2v_path = args.w2v_path

    print('loading dataset')
    mr = MovieReview('./data/')
    train_t, train_data, test_t, test_data = mr.load_dataset()
    train_n = len(train_t)
    test_n = len(test_t)

    print('building vocabulary')
    vocab = Vocabulary()
    for d in train_data + test_data:
        vocab.new(d)

    print('encoding')
    train_x = np.array([util.pad_to_max(vocab.encode(td), 150, 0) for td in train_data])
    train_t = np.array(train_t)
    test_x = np.array([util.pad_to_max(vocab.encode(td), 150, 0) for td in test_data])
    test_t = np.array(test_t)

    optimizer = opts['optimizer']
    model = opts['model']

    print('setting model up')
    class_n = 2
    vocab_n = len(vocab)
    d = 200
    fpath = w2v_path
    model = model(class_n, vocab_n, d, vocab, fpath)
    xp = model.check_gpu(gpu)
    optimizer.setup(model)

    train_loss_log = Log()
    test_loss_log = Log()
    test_acc_log = Log()

    print('starting training')
    for _ in tqdm(range(epoch)):

        order = np.random.permutation(train_n)
        train_x_iter = Iterator(train_x, bs, order, shuffle=False)
        train_t_iter = Iterator(train_t, bs, order, shuffle=False)

        loss_sum = 0
        for x, t in zip(train_x_iter, train_t_iter):
            model.cleargrads()
            x_n = len(x)
            x = model.prepare_input(x, dtype=xp.int32, xp=xp)
            t = model.prepare_input(t, dtype=xp.int32, xp=xp)
            loss, _ = model(x, t, train=True)
            loss_sum += loss.data * x_n

            loss.backward()
            optimizer.update()
        loss_mean = float(loss_sum/train_n)
        train_loss_log.add(loss_mean)
        print('train loss: {}'.format(loss_mean))
        del loss

        order = np.random.permutation(test_n)
        test_x_iter = Iterator(test_x, bs, order)
        test_t_iter = Iterator(test_t, bs, order)
        loss_sum = 0
        acc_sum = 0
        for x, t in zip(test_x_iter, test_t_iter):
            model.cleargrads()
            x_n = len(x)
            x = model.prepare_input(x, dtype=xp.int32, xp=xp)
            t = model.prepare_input(t, dtype=xp.int32, xp=xp)
            loss, acc = model(x, t, train=False)
            loss_sum += loss.data * x_n
            acc_sum += acc.data * x_n
        loss_mean = float(loss_sum / test_n)
        acc_mean = float(acc_sum / test_n)
        test_loss_log.add(loss_mean)
        test_acc_log.add(acc_mean)
        print('test loss: {}'.format(loss_mean))
        print('test acc: {}'.format(acc_mean))
        del loss
        del acc

    output_path = './results/{}'.format(output_dirname)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    train_loss_log.save('{}/train_loss_log'.format(output_path))
    train_loss_log.save_graph('{}/train_loss_log'.format(output_path))
    test_loss_log.save('{}/test_loss_log'.format(output_path))
    test_loss_log.save_graph('{}/test_loss_log'.format(output_path))
    test_acc_log.save('{}/test_acc_log'.format(output_path))
    test_acc_log.save_graph('{}/test_acc_log'.format(output_path))


if __name__ == '__main__':
    opts = {}
    opts['model'] = MLP
    opts['optimizer'] = optimizers.AdaGrad()
    train(opts)
