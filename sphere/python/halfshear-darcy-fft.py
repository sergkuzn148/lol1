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
import scipy.fftpack

#sigma0_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])
#sigma0 = 10.0e3
sigma0 = 20000.0
k_c_vals = [3.5e-13, 3.5e-15]
mu_f = 1.797e-06
velfac = 1.0
#cvals = [1.0, 0.1, 0.01]
#cvals = [1.0]


shear_strain = [[], [], [], []]
friction = [[], [], [], []]
dilation = [[], [], [], []]
p_min = [[], [], [], []]
p_mean = [[], [], [], []]
p_max = [[], [], [], []]
f_n_mean = [[], [], [], []]
f_n_max  = [[], [], [], []]
v_f_z_mean  = [[], [], [], []]
t_total = []

fluid=True

# dry shear
#sid = 'shear-sigma0=' + sys.argv[1] + '-hw'
# halfshear-darcy-sigma0=20000.0-k_c=3.5e-13-mu=1.797e-06-velfac=1.0-shear
sid = 'halfshear-sigma0=' + str(sigma0) + '-shear'
sim = sphere.sim(sid)
sim.readlast(verbose=False)
sim.visualize('shear')
shear_strain[0] = sim.shear_strain
#shear_strain[0] = numpy.arange(sim.status()+1)
friction[0] = sim.tau/sim.sigma_eff
dilation[0] = sim.dilation
t_total.append(sim.time_total[0])


# wet shear
c = 1
for c in numpy.arange(1,len(k_c_vals)+1):
    k_c = k_c_vals[c-1]

    # halfshear-darcy-sigma0=20000.0-k_c=3.5e-13-mu=1.797e-06-velfac=1.0-shear
    sid = 'halfshear-darcy-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
            '-mu=' + str(mu_f) + '-velfac=' + str(velfac) + '-shear'
    #sid = 'halfshear-sigma0=' + str(sigma0) + '-c_v=' + str(c_v) +\
            #'-c_a=0.0-velfac=1.0-shear'
    if os.path.isfile('../output/' + sid + '.status.dat'):

        sim = sphere.sim(sid, fluid=fluid)
        shear_strain[c] = numpy.zeros(sim.status())
        friction[c] = numpy.zeros_like(shear_strain[c])
        dilation[c] = numpy.zeros_like(shear_strain[c])

        sim.readlast(verbose=False)
        sim.visualize('shear')
        t_total.append(sim.time_total[0])
        shear_strain[c] = sim.shear_strain
        #shear_strain[c] = numpy.arange(sim.status()+1)
        friction[c] = sim.tau/sim.sigma_eff
        dilation[c] = sim.dilation

    else:
        print(sid + ' not found')

    # produce VTK files
    #for sid in sids:
        #sim = sphere.sim(sid, fluid=True)
        #sim.writeVTKall()
    c += 1

fig = plt.figure(figsize=(8,8)) # (w,h)
#fig.subplots_adjust(hspace=0.0)

#ax1 = plt.subplot(111)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212, sharex=ax1)
alpha = 1.0
#ax1.plot(shear_strain[0], friction[0], label='dry', linewidth=1, alpha=alpha)

color = ['b','g','r','c']
for c in numpy.arange(0,len(k_c_vals)+1):

    if c == 0:
        label = 'dry'
    elif c == 1:
        label = 'wet, relatively permeable'
    elif c == 2:
        label = 'wet, relatively impermeable'
    else:
        label = '$k_c$ = %.1e m$^2$' % (k_c_vals[c-1])

    str_arr = shear_strain[c][200:1999]
    dil_arr = dilation[c][200:1999]
    t = numpy.linspace(0.0, sim.time_total[0], shear_strain[c].size)

    freqs = scipy.fftpack.fftfreq(str_arr.size, t[1]-t[0])
    str_yf    = numpy.abs(scipy.fftpack.fft(str_arr))
    dil_yf    = numpy.abs(scipy.fftpack.fft(dil_arr))

    ax1.plot(freqs, str_yf, label=label, linewidth=1, alpha=alpha)
    ax2.plot(freqs, dil_yf, label=label, linewidth=1, alpha=alpha)

ax2.set_xlabel('Frequency [s$^{-1}$]')

#ax1.set_ylabel('Shear friction $\\tau/\\sigma\'$ [-]')
#ax2.set_ylabel('Dilation $\\Delta h/(2r)$ [-]')

ax1.set_xlim([0.0,12.0])
plt.setp(ax1.get_xticklabels(), visible=False)

ax1.grid()
ax2.grid()

legend_alpha=0.5
ax1.legend(loc='best', prop={'size':18}, fancybox=True,
        framealpha=legend_alpha)
#ax2.legend(loc='lower right', prop={'size':18}, fancybox=True,
        #framealpha=legend_alpha)

#ax1.set_ylim([-0.1, 1.9])

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
#filename = 'shear-' + str(int(sigma0/1000.0)) + 'kPa-stress-dilation.pdf'
filename = 'halfshear-darcy-fft.pdf'
#print(os.getcwd() + '/' + filename)
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2/graphics/' + filename)
print(filename)
