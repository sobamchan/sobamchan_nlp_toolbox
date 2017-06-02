import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sobamchan_utility
util = sobamchan_utility.Utility()

class Log(object):

    def __init__(self):
        self.values = []

    def add(self, value):
        self.values.append(value)

    @property
    def latest(self):
        values = self.values
        return values[len(values)-1]

    def save(self, filepath=''):
        if filepath == '':
            print('You need to set filepath to save image')
        if 'json' not in filepath:
            filepath = filepath + '.json'
        util.save_json(self.values, filepath)

    def save_graph(self, filepath='', title=None, xlabel=None, ylabel=None):
        if filepath == '':
            print('You need to set filepath to save image')
        if 'png' not in filepath:
            filepath = filepath + '.png'
        Y = self.values
        X = range(0, len(Y))

        ax = plt.subplot()
        ax.grid(True)
        ax.set_yscale('linear')
        ax.plot(X, Y)
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)

        plt.savefig(filepath)
        plt.clf()

    def load_data(self, fpath=''):
        if fpath == '':
            print('You need to set filepath to load data')
        if 'json' not in fpath:
            fpath = fpath + '.json'
        self.values = util.load_json(fpath)
        return self.values

if __name__ == '__main__':
    import math
    log = Log()
    x = np.arange(-10, 10, 0.1)
    for xx in x:
        log.add(math.sin(xx))
    log.save_graph('./logtest', title='test', xlabel='x', ylabel='y')
