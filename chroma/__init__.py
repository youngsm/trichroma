
import os
if 'PYGAME_HIDE_SUPPORT_PROMPT' not in os.environ:
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

_VIEWER = ("Camera", "EventViewer", "view", "build")


def __getattr__(name):
    # The viewer needs PyCUDA and pygame: import it only when it is used.
    if name in _VIEWER:
        from chroma import camera
        return getattr(camera, name)
    raise AttributeError("module 'chroma' has no attribute %r" % name)


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
