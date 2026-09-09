import numpy as np
import matplotlib.cm as cm

def map_to_color(a, range=None, map=cm.jet_r, weights=None):
    a = np.asarray(a,dtype=np.float32)
    
    if range is None:
        range = (a.min(), a.max())

    ax = (a - float(range[0]))/(float(range[1])-float(range[0]))

    frgba = map(ax)

    if weights is not None:
        frgba[:,0] *= weights
        frgba[:,1] *= weights
        frgba[:,2] *= weights

    rgba = (frgba*255).astype(np.uint32)

    return rgba[:,0] << 16 | rgba[:,1] << 8 | rgba[:,2]
