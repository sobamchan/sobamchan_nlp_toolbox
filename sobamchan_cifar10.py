import chainer
from chainer import datasets
from chainer import optimizers
import numpy as np
from tqdm import tqdm

import argparse
import os

from .sobamchan_iterator import Iterator
from .sobamchan_log import Log



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', dest='bs', default=128, type=int)
    parser.add_argument('--epoch', dest='epoch', default=1, type=int)
    parser.add_argument('--gpu', dest='gpu', default=-1, type=int)
    parser.add_argument('--output-dirname', dest='output_dirname', required=True)

    return parser.parse_args()

def train(opts):

    args = get_args()

    bs = args.bs
    epoch = args.epoch
    gpu = args.gpu
    output_dirname = args.output_dirname

    optimizer = opts['optimizer']
    model = opts['model']()
    xp = model.check_gpu(gpu)
    optimizer.setup(model)

    train, test = datasets.get_cifar10()
    train_x = xp.array([x[0] for x in train])
    train_t = xp.array([x[1] for x in train])
    test_x = xp.array([x[0] for x in test])
    test_t = xp.array([x[1] for x in test])

    train_n = len(train_t)
    test_n = len(test_t)

    train_loss_log = Log()
    test_loss_log = Log()
    test_acc_log = Log()

    for _ in tqdm(range(epoch)):

        order = np.random.permutation(train_n)
        train_x_iter = Iterator(train_x, bs, order, shuffle=False)
        train_t_iter = Iterator(train_t, bs, order, shuffle=False)

        loss_sum = 0
        for x, t in zip(train_x_iter, train_t_iter):
            model.cleargrads()
            x_n = len(x)
            x = model.prepare_input(x, dtype=xp.float32, xp=xp)
            t = model.prepare_input(t, dtype=xp.int32, xp=xp)
            loss, _ = model(x, t, train=True)
            loss_sum += loss.data * x_n

            loss.backward()
            optimizer.update()
        loss_mean = float(loss_sum/train_n)
        train_loss_log.add(loss_mean)
        print('train loss: {}'.format(loss_mean)) 

        order = np.random.permutation(test_n)
        test_x_iter = Iterator(test_x, bs, order)
        test_t_iter = Iterator(test_t, bs, order)
        loss_sum = 0
        acc_sum = 0
        for x, t in zip(test_x_iter, test_t_iter):
            model.cleargrads()
            x_n = len(x)
            x = model.prepare_input(x, dtype=xp.float32, xp=xp)
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


    output_path = './results/{}'.format(output_dirname)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    train_loss_log.save('{}/train_loss_log'.format(output_path))
    test_loss_log.save('{}/test_loss_log'.format(output_path))
    test_acc_log.save('{}/test_acc_log'.format(output_path))

if __name__ == '__main__':
    train()
