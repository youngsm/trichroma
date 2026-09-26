import numpy as np
import time
import ROOT
ROOT.gROOT.SetStyle('Plain')
from .histogram import Histogram
from .graph import Graph

def rootify(obj, *pars, **kwargs):
    if type(obj) is Histogram:
        return rootify_histogram(obj, *pars, **kwargs)
    if type(obj) is Graph:
        return rootify_graph(obj, *pars, **kwargs)
    if callable(obj):
        return rootify_function(obj, *pars, **kwargs)
    raise Exception("i don't know how to rootify %s" % type(obj))

def rootify_function(f, pars=(), name='', xmin=-1, xmax=50):
    def func(xbuf, pars=()):
        x = [x for x in xbuf]
        return f(x[0], *pars)

    if name == '':
        name = 'f%s' % len(ROOT.gROOT.GetListOfFunctions())

    froot = ROOT.TF1(name, func, xmin, xmax, len(pars))

    for i, par in enumerate(pars):
        froot.SetParameter(i, par)

    return froot

def rootify_graph(g, name='', title='', **kwargs):
    groot = ROOT.TGraphErrors(g.size,
                              np.asarray(g.x, dtype=np.double),
                              np.asarray(g.y, dtype=np.double),
                              np.asarray(g.xerr, dtype=np.double),
                              np.asarray(g.yerr, dtype=np.double))
    groot.SetTitle(title)
    return groot

def rootify_histogram(h, name='', title='', **kwargs):
    if name == '':
        name = time.asctime()

    hroot = ROOT.TH1D(name, title, h.hist.size, h.bins)
    for i in range(h.hist.size):
        hroot[i+1] = h.hist[i]
        hroot.SetBinError(i+1, h.errs[i])

    if 'linecolor' in kwargs:
        hroot.SetLineColor(kwargs['linecolor'])
        
    return hroot

def update_histogram(h, hroot):
    for i in range(h.hist.size):
        hroot[i+1] = h.hist[i]
        hroot.SetBinError(i+1, h.errs[i])

def getcanvas(log=False):
    c = ROOT.TCanvas('c%s' % len(ROOT.gROOT.GetListOfCanvases()), '', 800, 600)
    if log:
        c.SetLogy()
    return c
