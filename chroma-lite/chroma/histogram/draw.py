import numpy as np
import matplotlib.pyplot as plt
from .histogram import Histogram
from .histogramdd import HistogramDD

def draw(obj, title='', xlabel='', ylabel='', **kwargs):
    if isinstance(obj, Histogram):
        plt.figure()
        plt.bar(obj.bins[:-1], obj.hist, np.diff(obj.bins), yerr=obj.errs,
                color='white', edgecolor='black', ecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.draw()

    if isinstance(obj, HistogramDD):
        if len(obj.hist.shape) != 2:
            raise TypeError("drawing only supported for 2-d histograms.")

        plt.figure()
        cs = plt.contour(obj.bincenters[0], obj.bincenters[1], obj.hist)
        plt.clabel(cs, fontsize=9, inline=1)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.draw()
