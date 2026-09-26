import chroma.parabola as parabola
import numpy
from uncertainties import ufloat, unumpy
from .unittest_find import unittest
from numpy.testing import assert_almost_equal

class Test1D(unittest.TestCase):
    def setUp(self):
        self.x = numpy.array([[-1.0], [0.0], [1.0]])
        self.y = unumpy.uarray([2.0, 1.0, 6.0], [0.1, 0.1, 0.1])
        self.a = 1.0
        self.b = numpy.array([2.0])
        self.c = numpy.array([[3.0]])

    def test_parabola_eval(self):
        y = parabola.parabola_eval(self.x, self.a, self.b, self.c)
        assert_almost_equal(y, unumpy.nominal_values(self.y))

    def test_solve(self):
        points = list(zip(self.x, self.y))
        a, b, c, chi2, prob = parabola.parabola_fit(points)
        
        assert_almost_equal(a.nominal_value, 1.0)
        assert_almost_equal(b[0].nominal_value, 2.0)
        assert_almost_equal(c[0,0].nominal_value, 3.0)

        # Compare to ROOT TGraph fitting uncerts
        assert_almost_equal(a.std_dev, 0.1)
        assert_almost_equal(b[0].std_dev, 0.0707107)
        assert_almost_equal(c[0,0].std_dev, 0.122474, decimal=5)


class Test2D(unittest.TestCase):
    def setUp(self):
        self.x = numpy.array([[-1.0,-1.0], [-1.0, 0.0], [-1.0, 1.0],
                              [ 0.0,-1.0], [ 0.0, 0.0], [ 0.0, 1.0],
                              [ 1.0,-1.0], [ 1.0, 0.0], [ 1.0, 1.0]])
        
        self.a = 1.0
        self.b = numpy.array([2.0, 3.0])
        self.c = numpy.array([[3.0, 1.0],[1.0, 4.0]])
        
        self.y = numpy.zeros(len(self.x), dtype=object)
        for i, (x0, x1) in enumerate(self.x):
            value = self.a \
                + x0 * self.b[0] + x1 * self.b[1] \
                + x0**2 * self.c[0,0] + x0 * x1 * self.c[0,1] \
                + x1 * x0 * self.c[1,0] + x1**2 * self.c[1,1]
            self.y[i] = ufloat(value, 0.1)

    def test_parabola_eval(self):
        y = parabola.parabola_eval(self.x, self.a, self.b, self.c)
        assert_almost_equal(y, unumpy.nominal_values(self.y))

    def test_solve(self):
        points = list(zip(self.x, self.y))
        a, b, c, chi2, prob = parabola.parabola_fit(points)
        assert_almost_equal(chi2, 0.0)
        assert_almost_equal(a.nominal_value, 1.0)
        assert_almost_equal(b[0].nominal_value, 2.0)
        assert_almost_equal(b[1].nominal_value, 3.0)
        assert_almost_equal(c[0,0].nominal_value, 3.0)
        assert_almost_equal(c[0,1].nominal_value, 1.0)
        assert_almost_equal(c[1,0].nominal_value, 1.0)
        assert_almost_equal(c[1,1].nominal_value, 4.0)

