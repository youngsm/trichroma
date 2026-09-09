import chroma.geometry as geometry
import numpy as np
import os.path as path

def load_csv(file,delimeter=','):
    return np.loadtxt(path.join(path.dirname(__file__),file),delimiter=delimeter)

#Fake perfect PMT
perfect_pmt_photocathode = geometry.Surface('perfect_pmt_photocathode', 0)
perfect_pmt_wvl,perfect_pmt_prob = load_csv('perfect_pmt_qe.csv').T
perfect_pmt_prob /= 100. #percent to prob
perfect_pmt_photocathode.set('detect', wavelengths=perfect_pmt_wvl, value=perfect_pmt_prob)
perfect_pmt_photocathode.set('reflect_diffuse', wavelengths=perfect_pmt_wvl, value=(1.-perfect_pmt_prob)*0.5) #half of remaining photons reflect diffuse

# r5912-mod
r5912_mod_photocathode = geometry.Surface("r5912_mod_photocathode", 0)
r5912_mod_wvl = [60.0, 200.0, 260.0, 270.0, 280.0, 285.0, 290.0, 300.0, 310.0, 330.0, 370.0, 420.0, 475.0, 500.0, 530.0, 570.0, 600.0, 630.0, 670.0, 700.0, 800.0, ]
r5912_mod_prob = [0.0, 0.0, 0.0, 0.01, 0.05, 0.10, 0.15, 0.18, 0.20, 0.25, 0.27, 0.25, 0.20, 0.17, 0.10, 0.05, 0.025, 0.01, 0.001, 0.0, 0.0, ]

r5912_mod_prob = np.array(r5912_mod_prob)
r5912_mod_wvl = np.array(r5912_mod_wvl)

r5912_mod_photocathode.set("detect", wavelengths=r5912_mod_wvl, value=r5912_mod_prob)
r5912_mod_photocathode.set("reflect_diffuse", wavelengths=r5912_mod_wvl, value=(1.0 - r5912_mod_prob) * 0.5)  # half of remaining photons reflect diffuse

__exports__ = [
               'perfect_pmt_photocathode',
               'r5912_mod_photocathode',
               ]
