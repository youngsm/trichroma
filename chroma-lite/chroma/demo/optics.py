import numpy as np
from chroma.geometry import Material, Surface

vacuum = Material('vacuum')
vacuum.set('refractive_index', 1.0)
vacuum.set('absorption_length', 1e6)
vacuum.set('scattering_length', 1e6)

lambertian_surface = Surface('lambertian_surface')
lambertian_surface.set('reflect_diffuse', 1)

black_surface = Surface('black_surface')
black_surface.set('absorb', 1)

shiny_surface = Surface('shiny_surface')
shiny_surface.set('reflect_specular', 1)

glossy_surface = Surface('glossy_surface')
glossy_surface.set('reflect_diffuse', 0.5)
glossy_surface.set('reflect_specular', 0.5)

red_absorb_surface = Surface('red_absorb')
red_absorb_surface.set('absorb', [0.0, 0.0, 1.0], [465, 545, 685])
red_absorb_surface.set('reflect_diffuse', [1.0, 1.0, 0.0], [465, 545, 685])

# r7081hqe photocathode material surface
# source: hamamatsu supplied datasheet for r7081hqe pmt serial number zd0062
r7081hqe_photocathode = Surface('r7081hqe_photocathode')
r7081hqe_photocathode.detect = \
    np.array([(260.0,  0.00),
              (270.0,  0.04), (280.0,  0.07), (290.0,  0.77), (300.0,  4.57),
              (310.0, 11.80), (320.0, 17.70), (330.0, 23.50), (340.0, 27.54),
              (350.0, 30.52), (360.0, 31.60), (370.0, 31.90), (380.0, 32.20),
              (390.0, 32.00), (400.0, 31.80), (410.0, 30.80), (420.0, 30.16),
              (430.0, 29.24), (440.0, 28.31), (450.0, 27.41), (460.0, 26.25),
              (470.0, 24.90), (480.0, 23.05), (490.0, 21.58), (500.0, 19.94),
              (510.0, 18.48), (520.0, 17.01), (530.0, 15.34), (540.0, 12.93),
              (550.0, 10.17), (560.0,  7.86), (570.0,  6.23), (580.0,  5.07),
              (590.0,  4.03), (600.0,  3.18), (610.0,  2.38), (620.0,  1.72),
              (630.0,  0.95), (640.0,  0.71), (650.0,  0.44), (660.0,  0.25),
              (670.0,  0.14), (680.0,  0.07), (690.0,  0.03), (700.0,  0.02),
              (710.0,  0.00)])
# convert percent -> fraction
r7081hqe_photocathode.detect[:,1] /= 100.0
# roughly the same amount of detected photons are absorbed without detection
r7081hqe_photocathode.absorb = r7081hqe_photocathode.detect
# remaining photons are diffusely reflected
r7081hqe_photocathode.set('reflect_diffuse', 1.0 - r7081hqe_photocathode.detect[:,1] - r7081hqe_photocathode.absorb[:,1], wavelengths=r7081hqe_photocathode.detect[:,0])

# glass data comes from 'glass_sno' material in SNO+ optics database
glass = Material('glass')
glass.set('refractive_index', 1.49)
glass.absorption_length = \
    np.array([(200, 0.1e-6), (300, 0.1e-6), (330, 1000.0), (500, 2000.0), (600, 1000.0), (770, 500.0), (800, 0.1e-6), (1000, 0.1e-6)])
glass.set('scattering_length', 1e6)

# From WCSim
water = Material('water')
water.density = 1.0 # g/cm^3
water.composition = { 'H' : 0.1119, 'O' : 0.8881 } # fraction by mass
hc_over_GeV = 1.2398424468024265e-06 # h_Planck * c_light / GeV / nanometer
wcsim_wavelengths = hc_over_GeV / np.array([ 1.56962e-09, 1.58974e-09, 1.61039e-09, 1.63157e-09, 
       1.65333e-09, 1.67567e-09, 1.69863e-09, 1.72222e-09, 
       1.74647e-09, 1.77142e-09,1.7971e-09, 1.82352e-09, 
       1.85074e-09, 1.87878e-09, 1.90769e-09, 1.93749e-09, 
       1.96825e-09, 1.99999e-09, 2.03278e-09, 2.06666e-09,
       2.10169e-09, 2.13793e-09, 2.17543e-09, 2.21428e-09, 
       2.25454e-09, 2.29629e-09, 2.33962e-09, 2.38461e-09, 
       2.43137e-09, 2.47999e-09, 2.53061e-09, 2.58333e-09, 
       2.63829e-09, 2.69565e-09, 2.75555e-09, 2.81817e-09, 
       2.88371e-09, 2.95237e-09, 3.02438e-09, 3.09999e-09,
       3.17948e-09, 3.26315e-09, 3.35134e-09, 3.44444e-09, 
       3.54285e-09, 3.64705e-09, 3.75757e-09, 3.87499e-09, 
       3.99999e-09, 4.13332e-09, 4.27585e-09, 4.42856e-09, 
       4.59258e-09, 4.76922e-09, 4.95999e-09, 5.16665e-09, 
       5.39129e-09, 5.63635e-09, 5.90475e-09, 6.19998e-09 ])[::-1] #reversed

water.set('refractive_index', 
                wavelengths=wcsim_wavelengths,
                value=np.array([1.32885, 1.32906, 1.32927, 1.32948, 1.3297, 1.32992, 1.33014, 
                          1.33037, 1.3306, 1.33084, 1.33109, 1.33134, 1.3316, 1.33186, 1.33213,
                          1.33241, 1.3327, 1.33299, 1.33329, 1.33361, 1.33393, 1.33427, 1.33462,
                          1.33498, 1.33536, 1.33576, 1.33617, 1.3366, 1.33705, 1.33753, 1.33803,
                          1.33855, 1.33911, 1.3397, 1.34033, 1.341, 1.34172, 1.34248, 1.34331,
                          1.34419, 1.34515, 1.3462, 1.34733, 1.34858, 1.34994, 1.35145, 1.35312,
                          1.35498, 1.35707, 1.35943, 1.36211, 1.36518, 1.36872, 1.37287, 1.37776,
                          1.38362, 1.39074, 1.39956, 1.41075, 1.42535])[::-1] #reversed
)
water.set('absorption_length',
                wavelengths=wcsim_wavelengths,
                value=np.array([22.8154, 28.6144, 35.9923, 45.4086, 57.4650,
                 72.9526, 75,      81.2317, 120.901, 160.243,
                 193.797, 215.045, 227.786, 243.893, 294.113,
                 321.735, 342.931, 362.967, 378.212, 449.602,
                 740.143, 1116.06, 1438.78, 1615.48, 1769.86,
                 2109.67, 2304.13, 2444.97, 3076.83, 4901.5,
                 6666.57, 7873.95, 9433.81, 10214.5, 10845.8,
                 15746.9, 20201.8, 22025.8, 21142.2, 15083.9,
                 11751,   8795.34, 8741.23, 7102.37, 6060.68,
                 4498.56, 3039.56, 2232.2,  1938,    1811.58,
                 1610.32, 1338.7,  1095.3,  977.525, 965.258,
                 1082.86, 876.434, 633.723, 389.87,  142.011])[::-1] * 10.0 # reversed, cm->mm
                )
      
water.set('scattering_length',
                wavelengths=wcsim_wavelengths,
                value=np.array([167024.4, 158726.7, 150742,
                          143062.5, 135680.2, 128587.4,
                          121776.3, 115239.5, 108969.5,
                          102958.8, 97200.35, 91686.86,
                          86411.33, 81366.79, 76546.42,
                          71943.46, 67551.29, 63363.36,
                          59373.25, 55574.61, 51961.24,
                          48527.00, 45265.87, 42171.94,
                          39239.39, 36462.50, 33835.68,
                          31353.41, 29010.30, 26801.03,
                          24720.42, 22763.36, 20924.88,
                          19200.07, 17584.16, 16072.45,
                          14660.38, 13343.46, 12117.33,
                          10977.70, 9920.416, 8941.407,
                          8036.711, 7202.470, 6434.927,
                          5730.429, 5085.425, 4496.467,
                          3960.210, 3473.413, 3032.937,
                          2635.746, 2278.907, 1959.588,
                          1675.064, 1422.710, 1200.004,
                          1004.528, 833.9666, 686.1063])[::-1] * 10.0 * 0.625 # reversed, cm -> mm, * magic tuning constant
          )
