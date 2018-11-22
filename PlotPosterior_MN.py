
from __future__ import print_function, division

import numpy as np
import math

from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams["font.size"] = 16

import os
import sys

import GetPosterior
import global_imports
from getdist import plots, MCSamples
from matplotlib.ticker import MultipleLocator, MaxNLocator, AutoLocator, AutoMinorLocator

# For 1 spot

names = ['mass', 'radius', 'inclination', 'colatitude', 'spotradius',
             'temperature', 'elsewhere', 'phase', 'distance', 'compactness']

# the hard bounds imposed above
bounds = {'mass': (1.0, 2.6),
          'radius': (8.0, 16.0),
          'inclination': (1.0 * math.pi / 180.0, 179.0 * math.pi / 180.0),
          'colatitude': (1.0 * math.pi / 180.0, 179.0 * math.pi / 180.0),
          'spotradius': (0.01 * math.pi / 180.0, 89.0 * math.pi / 180.0),
          'temperature': (0.1, 4.),
          'elsewhere': (0.1, 2.),
          'phase': (0.0, 1.0),
         'distance': (1.,10.),
         'compactness': (0,0.3)}

# LaTeX compatible labels
labels = [r"M\;\mathrm{[}M_{\mathrm{\odot}}\mathrm{]}",
          r"R\;\mathrm{[km]}",
          r"i\;\mathrm{[rad]}",
          r"\Theta\;\mathrm{[rad]}",
          r"\zeta\;\mathrm{[rad]}",
          r"T\;[\mathrm{keV}]",
          r"T_{E}\;[\mathrm{keV}]",
          r"\phi",
         r"D",
         r"M/R"]
         
analysis_settings = {'ignore_rows': 100,
                     'contours': [0.683, 0.954, 0.997],
                     #'contours': [0.1, 0.995, 0.999,0.9999, 0.99999, 1.-1e-10,],
                     'credible_interval_threshold': 0.01,
                     'range_ND_contour': 0,
                     'range_confidence': 0.01,
                     'fine_bins': 2048,
                     'smooth_scale_1D': -1,
                     'boundary_correction_order': 1,
                     'mult_bias_correction_order': 1,
                     'smooth_scale_2D': -1,
                     'max_corr_2D': 0.99,
                     'fine_bins_2D': 1024}
M = 1.6
R = 11.0
i = math.pi/2.0 #(from 0 to pi)
col = math.pi/2.0
Rs = 10.*math.pi/180. #(~0.17)
T = 1.
Te = 0.68
phi = 0.5
D = 8.
C = 6.

# the parameter recovery exercise is not blind, so we know the injected values
truths = {'radius': R,
        'mass': M,
          'inclination': i,
          'colatitude': col,
          'spotradius': Rs,
          'temperature': T,
          'elsewhere': Te,
          'phase': phi,
          'distance': D,
          'compactness':C}
samples = GetPosterior.MultiNestSampleContainer(ID = r'MultiNest',
                                      root = '../J1814_paper/testdir/ctest50',
                                      names = names,
                                      bounds = bounds,
                                      labels = labels,
                                      lines = {'lw': 2, 'color': 'black', 'alpha': 0.8},
                                      contours = {'color': 'darkred', 'alpha': 0.8},
                                      analysis_settings = analysis_settings,
                                      truths = truths)


# ['radius', 'mass', 'inclination', 'colatitude', 'spotradius', 'temperature', 'elsewhere', 'phase', 'distance']
plotter_settings = {'axes_fontsize': 16.0,
                    'lab_fontsize': 16.0,
                    'x_label_rotation': 60.0,
                    'num_plot_contours': 3,
                    'progress': False,
                    'subplot_size_inch': 2,
                    'solid_contour_palefactor' : 0.4}

order = ['compactness','radius', 'mass', 'inclination','colatitude', 'spotradius', 'temperature', 'elsewhere', 'phase', 'distance']

posterior = GetPosterior.GetPosterior(samples, plotter_settings, usetex = False)


#param_limits = {'mass' : (0.,3), 'radius' : (10.,18.)} 
posterior.plot_triangle(names = order, filled = True, crosshairs = False, write = False, normalise = True)

prior_samples = np.loadtxt('../J1814_paper/testdir/sampled_prior_c.txt')

#posterior.plot_triangle(names = order, filled = True, crosshairs = None, write = False)

for j in range(len(order)):
    for i in range(j,len(order)):
        ax = posterior.plotter.subplots[i,j]
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2, prune='both'))
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        
        if i == j:
            if i == 1:
                density = np.histogram(prior_samples[:,2], bins='fd', density=True)
            elif i == 2:
                density = np.histogram(prior_samples[:,1], bins='fd', density=True)
            else:
                density = np.histogram(prior_samples[:,i], bins='fd', density=True)

            ax.plot(0.5*(density[1][1:] + density[1][:-1]), density[0], 'k--', lw=1.0)
            print(density)
    for i in range(j):
        ax = posterior.plotter.subplots[j,i]
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=2, prune='both'))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
#posterior.plotter.export(fname = <insert>, dpi = 300)

#plt.savefig('pos_test40.png', dpi = 300)
