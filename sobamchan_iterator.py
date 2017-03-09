import numpy as np

class Iterator(object):

    def __init__(self, dataset, batch_size, order=None, shuffle=True):
        self._i = 0
        self._N = len(dataset)
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._dataset = dataset
        if order is not None:
            self._order = order
        elif shuffle:
            self._order = np.random.permutation(self._N)
        else:
            self._order = list(range(self._N))

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= self._N:
            raise StopIteration
        i = self._i
        batch_size = self._batch_size
        batch = self._dataset[self._order[i:i+batch_size]]
        self._i = i + batch_size

        return batch
