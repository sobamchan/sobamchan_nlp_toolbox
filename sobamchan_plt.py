import matplotlib.pyplot as plt
import sobamchan_utility
utility = sobamchan_utility.Utility()


class Plt(object):
    def __init__(self):
        pass

    @staticmethod
    def plt_xy(x, y, x_label=None, y_label=None):
        X = x
        Y = y
        plt.plot(X, Y)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
