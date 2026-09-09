from chroma.geometry import Material, Solid, Surface
import numpy as np
#***************************************************************************
msuprasil_material = Material('msuprasil')
msuprasil_material.set('refractive_index', 1.57)
msuprasil_material.set('absorption_length', 1.152736998)
#msuprasil.set('absorption_length', .52736998)
msuprasil_material.set('scattering_length', 1e6)
msuprasil_material.density = 2.2
#***************************************************************************
teflon_material = Material('teflon')
teflon_material.set('refractive_index', 1.38)
teflon_material.set('absorption_length', 1)
teflon_material.set('scattering_length', 0)
teflon_material.density = 2.2
teflon_material.composition = {'F' : .9969, 'C' : .00063}
#***************************************************************************
steel_material = Material('steel')
steel_material.set('refractive_index', 1.07)
steel_material.set('absorption_length', 0)
steel_material.set('scattering_length', 0)
steel_material.density = 8.05
steel_material.composition = {'C' : .0008, 'Mg' : .02, 'P' : .0004, 'S' : .0003, 'Si' : .0075, 'Ni' : .08, 'Cr' : .18, 'Fe' : .711}
#***************************************************************************
copper_material = Material('copper')
copper_material.set('refractive_index', 1.3)
copper_material.set('absorption_length', 0)
copper_material.set('scattering_length',0)
copper_material.density = 8.96
copper_material.composition = {'Cu' : 1.00}
#***************************************************************************
ls_material = Material('ls')
ls_material.set('refractive_index', 1.5)
ls_material.set('absorption_length', 1e6)
ls_material.set('scattering_length', 1e6)
ls_material.density = 0.780
ls_material.composition = {'C' : .9, 'H' : .1}
#***************************************************************************
vacuum_material = Material('vac')
vacuum_material.set('refractive_index', 1.0)
vacuum_material.set('absorption_length', 1e6)
vacuum_material.set('scattering_length', 1e6)
vacuum_material.density = 1
#***************************************************************************
lensmat_material = Material('lensmat')
lensmat_material.set('refractive_index', 2.0)
lensmat_material.set('absorption_length', 1e6)
lensmat_material.set('scattering_length', 1e6)
#***************************************************************************
full_absorb_material = Material('full_absorb')
full_absorb_material.set('absorb', 1)
full_absorb_material.set('refractive_index', 1.5)
full_absorb_material.set('absorption_length', 1E20)
full_absorb_material.set('scattering_length', 1E20)
full_absorb_material.density = 1
#***************************************************************************
quartz_material = Material('quartz')
quartz_material.set('refractive_index', 1.6)
quartz_material.set('absorption_length', 9.49122)
quartz_material.set('scattering_length',1e6)
quartz_material.density = 2.65
#***************************************************************************
gold_material = Material('gold')
gold_material.set('refractive_index', 1.5215)    #according to https://refractiveindex.info/?shelf=main&book=Au&page=Werner
gold_material.set('absorption_length', 1e20)
gold_material.set('scattering_length',1e20)
gold_material.density = 19.32
#***************************************************************************
MgF2_material = Material('MgF2')
MgF2_material.set('refractive_index', 1.44)    #according to http://www.esourceoptics.com/vuv_material_properties.html
MgF2_material.set('absorption_length',1e20)
MgF2_material.set('scattering_length',1e20)
MgF2_material.density = 3.15
#***************************************************************************
ceramic_material = Material('ceramic')
ceramic_material.set('refractive_index', 1.94)	#https://refractiveindex.info/?shelf=main&book=Al2O3&page=Malitson-o
ceramic_material.set('absorption_length',20)
ceramic_material.set('scattering_length',100)
ceramic_material.density = 3.15
#***************************************************************************
SiO2_material = Material('SiO2')
SiO2_material.set('refractive_index', 1.97)
SiO2_material.set('absorption_length',100)
SiO2_material.set('scattering_length',100)
SiO2_material.density = 3.15
#***************************************************************************


__exports__ = ["msuprasil_material", "teflon_material", "steel_material", "copper_material", "ls_material", "vacuum_material", "lensmat_material", "full_absorb_material", "quartz_material", "gold_material", "MgF2_material", "ceramic_material", "SiO2_material"]