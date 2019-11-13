#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import shutil

import os
import sys
import numpy
import sphere
from permeabilitycalculator import *
import matplotlib.pyplot as plt
import scipy.optimize

import seaborn as sns
#sns.set(style='ticks', palette='Set2')
sns.set(style='ticks', palette='colorblind')
#sns.set(style='ticks', palette='muted')
#sns.set(style='ticks', palette='pastel')
sns.despine() # remove right and top spines

pressures = True
zflow = False
contact_forces = False
smooth_friction = True


sids = [
        'halfshear-darcy-sigma0=20000.0-k_c=3.5e-13-mu=1.797e-06-velfac=1.0-shear',
        'halfshear-darcy-sigma0=20000.0-k_c=3.5e-14-mu=1.797e-06-velfac=1.0-shear',
        'halfshear-darcy-sigma0=20000.0-k_c=3.5e-15-mu=1.797e-06-velfac=1.0-shear',

        #'halfshear-darcy-sigma0=20000.0-k_c=3.5e-15-mu=3.594e-07-velfac=1.0-shear',
        #'halfshear-darcy-sigma0=20000.0-k_c=3.5e-15-mu=6.11e-07-velfac=1.0-shear',

        'halfshear-darcy-sigma0=20000.0-k_c=3.5e-15-mu=1.797e-07-velfac=1.0-shear',
        'halfshear-darcy-sigma0=20000.0-k_c=3.5e-15-mu=1.797e-08-velfac=1.0-shear',


        #'halfshear-sigma0=20000.0-shear'

        # from halfshear-darcy-perm.sh
        #'halfshear-darcy-sigma0=20000.0-k_c=3.5e-13-mu=1.797e-07-velfac=1.0-shear',
        #'halfshear-darcy-sigma0=20000.0-k_c=3.5e-13-mu=1.797e-08-velfac=1.0-shear',
        #'halfshear-darcy-sigma0=20000.0-k_c=3.5e-13-mu=1.797e-09-velfac=1.0-shear',
        ]
fluids = [
        True,
        True,
        True,
        True,
        True,
        #False,
        True,
        True,
        True,
        ]

def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=numpy.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))

    if window == 'flat': #moving average
        w = numpy.ones(window_len,'d')
    else:
        w = getattr(numpy, window)(window_len)
    y = numpy.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


max_step = 400
friction = numpy.zeros(max_step)

velratios = []
peakfrictions = []

sim = sphere.sim(sids[0], fluid=True)
sim.readfirst(verbose=False)
fig = plt.figure(figsize=(3.5, 3))

if False:
    it=0
    for sid in sids:

        print '\n' + sid
        sim.id(sid)
        sim.fluid=fluids[it]
        it += 1

        sim.readfirst(verbose=False)

        velratio = 1.0
        if sim.fluid:
            if sim.k_c[0] > 3.5e-15 and sim.mu[0] < 1.797e-06:
                print 'Large permeability and low viscosity'
                velratio = 3.5e-15/sim.k_c[0] \
                    * sim.mu[0]/1.797e-06
            elif sim.k_c[0] > 3.5e-15:
                print 'Large permeability'
                velratio = 3.5e-15/sim.k_c[0]
            elif sim.mu < 1.797e-06:
                print 'Low viscosity'
                velratio = sim.mu[0]/1.797e-06
            elif numpy.isclose(sim.k_c[0], 3.5e-15) and \
                    numpy.isclose(sim.mu[0], 1.797e-06):
                print 'Normal permeability and viscosity'
                velratio = 1.
            else:
                raise Exception('Could not determine velratio')
        else:
            velratio = 0.001

        print 'velratio = ' + str(velratio)

        for i in numpy.arange(max_step):
            sim.readstep(i+1, verbose=False)

            friction[i] = sim.shearStress('effective')/\
                    sim.currentNormalStress('defined')

        smoothfriction = smooth(friction, 20, window='hanning')
        peakfriction = numpy.amax(smoothfriction[14:])

        plt.plot(numpy.arange(max_step), friction, alpha=0.3, color='b')
        plt.plot(numpy.arange(max_step), smoothfriction, color='b')
        plt.title(str(peakfriction))
        plt.savefig(sid + '-friction.pdf')
        plt.clf()

        velratios.append(velratio)
        peakfrictions.append(peakfriction)
else:
    velratios =\
        [0.01,
         0.099999999999999992,
         1.0,
         0.10000000000000001,
         #0.010000000000000002,
         #0.001,
         #0.0001,
         #1.0000000000000001e-05
         ]
    peakfrictions = \
        [0.619354807290315,
         0.6536161052814875,
         0.70810354077280957,
         0.64649301787774571,
         #0.65265739261434697,
         #0.85878138368962764,
         #1.6263903846405066,
         #0.94451353171977692
         ]


#def frictionfunction(velratio, a, b=numpy.min(peakfrictions)):
def frictionfunction(x, a, b):
    #return numpy.exp(a*velratio) + numpy.min(peakfrictions)
    #return -numpy.log(a*velratio) + numpy.min(peakfrictions)
    #return a*numpy.log(velratio) + b
    return a*10**(x) + b

popt, pvoc = scipy.optimize.curve_fit(
        frictionfunction,
        peakfrictions,
        velratios)
print popt
print pvoc
a=popt[0]
b=popt[1]

#a = 0.025
#b = numpy.min(peakfrictions)
#b = numpy.max(peakfrictions)

shearvel = sim.shearVel()/1000.*numpy.asarray(velratios) * (3600.*24.*365.25)

'''
plt.semilogx(
        #[1, numpy.min(shearvel)],
        [1, 6],
        #[1, 1000],
        #[numpy.min(peakfrictions), numpy.min(peakfrictions)],
        [0.615, 0.615],
        '--', color='gray', linewidth=2)
'''

#plt.semilogx(velratios, peakfrictions, 'o')
#plt.semilogx(shearvel, peakfrictions, 'o')
plt.semilogx(shearvel, peakfrictions, 'o')
#plt.plot(velratios, peakfrictions, 'o')

xfit = numpy.linspace(numpy.min(velratios), numpy.max(velratios))
#yfit = frictionfunction(xfit, popt[0], popt[1])
yfit = frictionfunction(xfit, a, b)
#plt.semilogx(xfit, yfit)
#plt.plot(xfit, yfit)

plt.xlabel('Shear velocity [m a$^{-1}$]')
plt.ylabel('Peak shear friction, $\\max(\\tau/\\sigma_0)$ [-]')
#plt.xlim([0.8, 110])
#plt.xlim([0.008, 1.1])
plt.xlim([1., 1000,])
plt.ylim([0.6, 0.72])
plt.tight_layout()
sns.despine() # remove right and top spines
filename = 'halfshear-darcy-rate-strength.pdf'
plt.savefig(filename)
plt.close()
print(filename)
