import numpy as np
import chainer
from chainer import serializers
from chainer import Chain
import os
import datetime

class Model(Chain):

    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

    def save_model(self, filename='./model.model', data_format=None):
        path, ext = os.path.splitext(filename)
        now = datetime.datetime.now().strftime('%Y%m%d%H%M')
        save_path = '{}_{}{}'.format(path, now, ext)
        serializers.save_hdf5(save_path, self)

    def load_model(self, filename):
        serializers.load_hdf5(filename, self)

    def prepare_input(self, x, dtype=np.float32, xp=None, volatile=False):
        if xp:
            x = xp.asarray(x, dtype=dtype)
        else:
            x = np.asarray(x, dtype=dtype)


        return chainer.Variable(x, volatile=volatile)
