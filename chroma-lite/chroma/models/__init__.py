import os.path
import glob
import sys

from chroma.stl import mesh_from_stl

class Loader(object):
    def __init__(self, filename):
        self.filename = filename
    def __call__(self):
        return mesh_from_stl(self.filename)

# Create functions to load
this_module = sys.modules[__name__]
for filename in glob.glob(os.path.join(os.path.dirname(__file__),'*.stl*')):
    name, ext = os.path.splitext(os.path.basename(filename))
    while ext != '':
        name, ext = os.path.splitext(name)
    setattr(this_module, name, Loader(filename))

