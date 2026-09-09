
import numpy as np
import numpy.ma as ma
from copy import copy, deepcopy

try:
    from uncertainties import ufloat, umath
    from uncertainties.unumpy import uarray
except ImportError:
    from warnings import warn
    warn('unable to import uncertainties package')

class HistogramDD(object):
    """
    Multi-dimensional histogram.

    Args
        - bins: sequence, *optional*
            - A sequence of arrays describing the bin edges along each
            dimension.
            - The number of bins for each dimension (nx,ny,...=bins)
        - range: sequence, *optional*
            A sequence of lower and upper bin edges to be used if the edges
            are not given explicitly bin `bins`.

    .. note::
        (from numpy.histogram, which is used to fill bins)

        All but the last (righthand-most) bin is half-open. In other words,
        if `bins` is ``[1,2,3,4]`` along some axis then the first bin along
        that axis is ``[1,2)`` (including 1, but excluding 2) and the second
        ``[2,3)``. The last bin, however, is ``[3,4]``, which *includes* 4.

    Example
        >>> h = Histogram((10,10), [(-0.5, 9,5), (-0.5, 9.5)])
        >>> h.fill(np.random.normal(5,2,size=(1000,2)))
    """
    def __init__(self, bins=(10,10), range=[(-0.5, 9.5), (-0.5, 9.5)]):
        try:
            D = len(bins)
        except TypeError:
            raise TypeError("bins must be a sequence.")

        self.bins = D*[None]
        self.bincenters = D*[None]

        nbins = D*[None]

        for i in np.arange(D):
            if np.isscalar(bins[i]):
                self.bins[i] = np.linspace(range[i][0], range[i][1], bins[i]+1)
            else:
                self.bins[i] = np.asarray(bins[i], float)

                if (np.diff(bins[i]) < 0).any():
                    raise AttributeError("bins must increase monotonically.")

            self.bincenters[i] = (self.bins[i][:-1] + self.bins[i][1:])/2

            nbins[i] = self.bins[i].size-1

        self.hist = np.zeros(nbins, float)
        self.errs = np.zeros(nbins, float)

        self.nentries = 0

    def fill(self, x):
        """Fill histogram with values from the array `x`."""
        x = np.asarray(x)

        if len(x.shape) == 1:
            x = np.array([x,])

        add = np.histogramdd(np.asarray(x), self.bins)[0]
        self.hist += add
        self.errs = np.sqrt(self.errs**2 + add)

        self.nentries += x.shape[0]

    def findbin(self, x):
        """
        Find the bin index corresponding to the value `x`.
        
        Args
            - x: sequence
                A list of arrays for the values along each axis, i.e. the
                first element of `x` should be the values along the first
                bin axis, the second element is the values along the second
                bin axis, etc.

        .. note::
            This syntax might seem unintuitive, but it allows numpy functions
            to be called directly which will allow for fast access to the
            histogram data.

        Returns
            - out: sequence
                A list of arrays for the bin index along each axis.

        Example
            >>> h = HistogramDD()
            >>> h.fill(np.random.normal(5,2,size=(1000,2)))

            Now, we get the bin indices along the diagonal of the histogram
            at (0,0), (1,1), ..., (9,9).

            >>> bins = h.findbin([range(10), range(10)])

            At this point, bins is a list with two entries. The first entry
            is the bin indices along the first axis, the second is the bin
            indices along the second axis. Now, we get the value at (0,0).

            >> h.hist[bins[0][0]][bins[1][0]]

            or,

            >> h.hist[bins[0][0], bins[1,0]]

            Or, we can get all the values at once.

            >> h.hist[bins]
        """
        # the new numpy histogram has a closed right interval on the last
        # bin, but this function will not give you that bin if called with
        # the last bin edge
        return [np.searchsorted(self.bins[i], x[i], side='right') - 1 for \
                    i in range(len(self.bins))]

    def eval(self, x, fill_value=0):
        """
        Return the histogram value at `x`.

        See findbin().
        """
        bins = self.findbin(x)
        mbins = [ma.masked_outside(bins[i], 0, self.hist[i].size-1, False) \
                 for i in range(len(bins))]
        value = ma.array(\
            self.hist[[mbins[i].filled(0) for i in range(len(mbins))]],
            mask=np.logical_or.reduce([ma.getmaskarray(mbins[i]) \
                                           for i in range(len(mbins))]))

        return value.filled(fill_value)

    def ueval(self, x, fill_value=0, fill_err=0):
        """
        Return the histogram value and uncertainty at `x`.

        See findbin().
        """
        bins = self.findbin(x)
        mbins = [ma.masked_outside(bins[i], 0, self.hist[i].size-1, False) \
                     for i in range(len(bins))]
        valuemask = np.logical_or.reduce([ma.getmaskarray(mbins[i]) \
                                              for i in range(len(mbins))])

        filledbins = [mbins[i].filled(0) for i in range(len(mbins))]
        value, err = ma.array(self.hist[filledbins], mask=valuemask), \
            ma.array(self.errs[filledbins], mask=valuemask)

        return uarray((value.filled(fill_value), err.filled(fill_err)))

    def reset(self):
        """Reset all bin contents/errors to zero."""
        self.hist[:] = 0.0
        self.errs[:] = 0.0

        self.nentries = 0

    def sum(self):
        """Return the sum of the bin contents."""
        return np.sum(self.hist)

    def usum(self):
        """Return the sum of the bin contents and uncertainty."""
        return np.sum(uarray((self.hist, self.errs)))

    def scale(self, c):
        """Scale bin values and errors by `c`."""
        self.hist *= c
        self.errs *= c

    def normalize(self):
        """
        Normalize the histogram such that the sum of the bin contents is 1.
        """
        self.scale(1/self.sum())
