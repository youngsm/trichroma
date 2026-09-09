import chroma.geometry as geometry

vacuum = geometry.Material('vacuum')
vacuum.set('refractive_index', 1.0)
vacuum.set('absorption_length', 1e6)
vacuum.set('scattering_length', 1e6)

__exports__ = ['vacuum']
