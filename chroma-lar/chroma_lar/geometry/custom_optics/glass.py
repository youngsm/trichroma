import chroma.geometry as geometry

"""
note: we set the refractive index and absorption length to match 450 nm, mimicking traversal
of glass after TPB reemission.
"""


glass = geometry.Material('glass')
glass.density = 1.0 # g/cm^3
glass.composition = { 'Si' : 0.4675, 'O' : 0.5325 } # fraction by mass
# glass.set('refractive_index', 
#     wavelengths=[60.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1000.0],
#     value=[1.707, 1.662, 1.589, 1.551, 1.531, 1.521, 1.516, 1.513, 1.512, 1.512]
#     )

glass.set("refractive_index", 1.525) # ~450 nm

glass.set('scattering_length', 1e6)
# glass.set('absorption_length', 
#     wavelengths=[60.0, 200.0, 280.0, 300.0, 350.0, 500.0, 600.0, 770.0, 800.0, 1000.0],
#     value=[0.1e-3, 0.1e-3, 0.1e-3, 1e0, 1.0e3, 2.0e3, 1.0e3, 1.0e3, 1.0e3, 1.0e3]
#     )
glass.set('absorption_length', 1.5e3) # ~450 nm

__exports__ = ['glass']
