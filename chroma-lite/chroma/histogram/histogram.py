
import numpy as np
import numpy.ma as ma
from copy import copy, deepcopy

try:
    from uncertainties import ufloat, umath
    from uncertainties.unumpy import uarray
except ImportError:
    from warnings import warn
    warn('unable to import uncertainties package')

class Histogram(object):
    """
    Histogram object.

    Args
        - bins: int or sequence of scalars
            If `bins` is an int, it defines the number of equal-width bins
            in the range given by `interval`. If 'bins is a sequence, it
            defines the bin edges, including the rightmost edge.
        - range: (float, float), *optional*
            The lower and upper range of the `bins` when `bins` is an int.

    .. note::
        (from numpy.histogram, which is used to fill bins)

        All but the last (righthand-most) bin is half-open. In other words,
        if `bins` is ``[1,2,3,4]`` then the first bin is ``[1,2)``
        (including 1, but excluding 2) and the second ``[2, 3)``. The last 
        bin, however, is ``[3, 4]``, which *includes* 4.

    Example
        >>> h = Histogram(100, (0, 100))
        >>> h.fill(np.random.exponential(3, size=10000)
    """
    def __init__(self, bins=10, range=(-0.5,9.5)):
        if np.isscalar(bins):
            self.bins = np.linspace(range[0], range[1], bins+1)
        else:
            self.bins = np.asarray(bins, float)

            if (np.diff(self.bins) < 0).any():
                raise AttributeError('bins must increase monotonically.')

        self.bincenters = (self.bins[:-1] + self.bins[1:])/2

        self.errs = np.zeros(self.bins.size - 1)
        self.hist = np.zeros(self.bins.size - 1)

        self.nentries = 0

    def fill(self, x):
        """Fill histogram with values from the array `x`."""
        add = np.histogram(np.asarray(x), self.bins)[0]
        self.hist += add
        self.errs = np.sqrt(self.errs**2 + add)

        self.nentries += np.sum(add)

    def findbin(self, x):
        """Find the bin index corresponding to the value `x`."""
        # the new numpy histogram has a closed right interval
        # on the last bin, but this function will not give you that bin
        # if called with the last bin edge
        return np.searchsorted(self.bins, x, side='right') - 1

    def eval(self, x, fill_value=0):
        """Return the histogram value at `x`."""
        mbins = ma.masked_outside(self.findbin(x), 0, self.hist.size-1)
        value = ma.masked_where(mbins.mask, self.hist[mbins.filled(0)])

        if np.iterable(x):
            return value.filled(fill_value)
        else:
            return value.filled(fill_value).item()

    def ueval(self, x, fill_value=0, fill_err=0):
        """Return the histogram value and uncertainty at `x`."""
        mbins = ma.masked_outside(self.findbin(x), 0, self.hist.size-1)
        value, err = ma.masked_where(mbins.mask, self.hist[mbins.filled(0)]), \
            ma.masked_where(mbins.mask, self.errs[mbins.filled(0)])

        if np.iterable(x):
            return uarray((value.filled(fill_value), err.filled(fill_err)))
        else:
            return ufloat((value.filled(fill_value).item(), \
                               err.filled(fill_err).item()))

    def interp(self, x):
        """
        Interpolate the histogram value at `x`.

        .. warning::
            ``interp()`` will return 0.0 for bincenter[0] < `x` < bincenter[-1]
        """
        return np.interp(x, self.bincenters, self.hist, left=0.0, right=0.0)
        
    def mean(self):
        """Return the mean of the histogram along the bin axis."""
        return np.dot(self.bincenters, self.hist)/np.sum(self.hist)

    def reset(self):
        """Reset all bin contents/errors to zero."""
        self.hist[:] = 0.0
        self.errs[:] = 0.0

        self.nentries = 0

    def sum(self, width=False):
        """
        Return the sum of the bin contents.

        Args
            - width: boolean, *optional*
                If `width` is True, multiply bin contents by bin widths.
        """
        if width:
            return np.dot(np.diff(self.bins), self.hist)
        else:
            return np.sum(self.hist)

    def usum(self, width=False):
        """
        Return the sum of the bin contents and uncertainty.

        See sum().
        """
        if width:
            return np.dot(np.diff(self.bins), uarray((self.hist, self.errs)))
        else:
            return np.sum(uarray((self.hist, self.errs)))

    def integrate(self, x1, x2, width=False):
        """
        Return the integral of the bin contents from `x1` to `x2`.

        Args
            - width: boolean, *optional*
                If `width` is True, multiply bin contents by bin widths.
        """
        i1, i2 = self.findbin([x1,x2])

        if width:
            return np.dot(np.diff(self.bins[i1:i2+2]), self.hist[i1:i2+1])
        else:
            return np.sum(self.hist[i1:i2+1])

    def uintegrate(self, x1, x2, width=False):
        """
        Return the integral of the bin contents from `x1` to `x2` and
        uncertainty.

        See integrate().
        """
        i1, i2 = self.findbin([x1,x2])

        if width:
            return np.dot(np.diff(self.bins[i1:i2+2]),
                          uarray((self.hist[i1:i2+1], self.errs[i1:i2+1])))
        else:
            return np.sum(uarray((self.hist[i1:i2+1], self.errs[i1:i2+1])))

    def scale(self, c):
        """Scale bin contents and errors by `c`."""
        self.hist *= c
        self.errs *= c

    def normalize(self, width=False):
        """
        Normalize the histogram such that the sum of the bin contents is 1.

        Args
            - width: boolean, *optional*
                if `width` is True, multiply bin values by bin width
        """
        self.scale(1/self.sum(width))

    def fit(self, func, pars=(), xmin=None, xmax=None, fmin=None, **kwargs):
        """
        Fit the histogram to the function `func` and return the optimum
        parameters.

        Args
            - func: callable
                Function to fit histogram to; should be callable as 
                ``func(x, *pars)``.
            - pars: tuple, *optional*
                Set of parameters passed to `func`, i.e. ``f(x, *pars)``.
            - xmin: float, *optional*
                   Minimum value along the bin axis to fit.
            - xmax: float, *optional*
                   Maximum value along the bin axis to fit.
            - fmin: callable, *optional*
                   Minimization function to use. Defaults to
                   ``scipy.optimize.fmin_powell``. See scipy.optimize
                   documenation for calling syntax.
                   
        Returns
            - xopt: float
                Optimized parameters.
            - chi2: float
                Minimum chi^2 value

        .. warning::
            ``func(x, *pars)`` **must** accept `x` as an array and return
            an array of mapped values.
        """
        import scipy.optimize

        if fmin is None:
            fmin = scipy.optimize.fmin

        return fmin(lambda x: self.chi2(func, x, xmin, xmax), pars, **kwargs)

    def chi2(self, func, pars=(), xmin=None, xmax=None):
        """
        Return the chi2 test value of the histogram with respect to the
        function `func`.

        Args:
            - func: callable
                   Function which chi2 of histogram is computed against.
                   Should be callable as ``func(x, *pars)``.
            - pars: tuple, *optional*
                   Set of parameters passed to func, i.e. ``f(x, *pars)``.
            - xmin: float, *optional*
                   Minimum value along the bin axis to compute chi2 for.
            - xmax: float, *optional*
                   Maximum value along the bin axis to compute chi2 for.

        Returns:
            - chi2 : float
                chi^2 value

        .. warning::
            ``func(x, *pars)`` **must** accept `x` as an array and return
            an array of mapped values.

        Example
            >>> from scipy.stats import norm
            >>> h = Histogram(100)
            >>> h.fill(np.random.normal(5,1,1000))
            >>> h.normalize(width=True)
            >>> h.chi2(norm.pdf, (5,1))
            39.358879146386258
        """

        if xmin is None:
            xmin = self.bins[0]
        if xmax is None:
            xmax = self.bins[-1]

        afunc = func(self.bincenters, *pars)
        amask = (xmin < self.bincenters)*(xmax > self.bincenters)*\
            (self.errs != 0.0)

        return np.sum(((afunc[amask] - self.hist[amask])/self.errs[amask])**2)
