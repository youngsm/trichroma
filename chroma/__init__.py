
import os
if 'PYGAME_HIDE_SUPPORT_PROMPT' not in os.environ:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

try:
    from chroma.camera import Camera, EventViewer, view, build
except ImportError:
    pass # Allow chroma usage when pygame not present
from chroma import geometry
from chroma import event
from chroma.backend import backend_name as _backend_name
try:
    if _backend_name() == "cuda":
        from chroma import gpu
    from chroma.sim import Simulation
except ImportError:
    print("WARNING: GPU context init failed. Chroma cannot perform simulations!")
from chroma import itertoolset
#from chroma import likelihood
#from chroma.likelihood import Likelihood
from chroma import make
from chroma.demo import optics
from chroma import sample
from chroma.stl import mesh_from_stl
from chroma import transform
