import numpy as np

class Graph(object):
    """
    Graph object.

    Args:
        - x: array, *optional*
        - y: array, *optional*
        - xerr: array, *optional*
        - yerr: array, *optional*
    """
    def __init__(self, x=[], y=[], xerr=None, yerr=None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        if xerr is None:
            self.xerr = np.zeros(self.x.size)
        else:
            self.xerr = np.array(xerr)
        if yerr is None:
            self.yerr = np.zeros(self.x.size)
        else:
            self.yerr = np.array(yerr)

        if self.y.size != self.x.size or self.xerr.size != self.x.size \
               or self.yerr.size != self.x.size:
            raise ValueError('array size mismatch')

        self.size = self.x.size
