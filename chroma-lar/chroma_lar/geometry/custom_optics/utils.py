import numpy as np

def pdf_to_cdf(x,y):
    yc = np.cumsum((y[1:]+y[:-1])*(x[1:]-x[:-1]))
    yc = np.concatenate([[0],yc])
    return yc / yc[-1]

def make_prop(a,b):
    return np.array(list(zip(a, b)), dtype=np.float32)
    
def exponential_decay_cdf(decays,weights,t_rise=None,times=np.arange(0,1000,0.05)):
     if t_rise is not None:
         cdf = np.sum([a*(t*(1.0-np.exp(-times/t))+t_rise*(np.exp(-times/t_rise)-1))/(t-t_rise) for t,a in zip(decays,weights)],axis=0)
     else:
         cdf = np.sum([a*(1.0-np.exp(-times/t)) for t,a in zip(decays,weights)],axis=0)
     return make_prop(times,cdf)
