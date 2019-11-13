#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
import os
import shutil

import sphere
import numpy
import matplotlib.pyplot as plt
#import diffusivitycalc

c_phi = 1.0
c_grad_p = 1.0
#sigma0_list = numpy.array([5.0e3, 10.0e3, 20.0e3, 40.0e3, 80.0e3, 160.0e3])
sigma0_list = numpy.array([10.0e3, 20.0e3, 40.0e3, 80.0e3, 160.0e3])
#alpha = numpy.empty_like(sigma0_list)
#phi_bar = numpy.empty_like(sigma0_list)
load = numpy.array([])
alpha = numpy.array([])
phi_bar = numpy.array([])

#dc = diffusivitycalc.DiffusivityCalc()

i = 0
for sigma0 in sigma0_list:

    sid = 'cons-sigma0=' + str(sigma0) + '-c_phi=' + \
                     str(c_phi) + '-c_grad_p=' + str(c_grad_p)
    if os.path.isfile('../output/' + sid + '.status.dat'):
        sim = sphere.sim(sid, fluid=True)

        #sim.visualize('walls')
        sim.plotLoadCurve()
        load = numpy.append(load, sigma0)
        alpha = numpy.append(alpha, sim.c_v)
        phi_bar = numpy.append(phi_bar, sim.phi_bar)
        #sim.writeVTKall()

    #else:
        #print(sid + ' not found')

    i += 1

fig, ax1 = plt.subplots()
load /= 1000.0

#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.plot(load, alpha, 'o-k')
ax1.set_xlabel('Normal stress $\\sigma_0$ [kPa]')
ax1.set_ylabel('Hydraulic diffusivity $\\alpha$ [m$^2$s$^{-1}$]')
#ax1.ticklabel_format(style='plain', axis='y')
#ax1.grid()

ax2 = ax1.twinx()
#color = 'black'
#ax2.plot(load, phi_bar, 'o--' + color)
ax2.plot(load, phi_bar, 'o--', color='black')
ax2.set_ylabel('Mean porosity $\\bar{\\phi}$ [-]')
#ax2.set_ylabel('Mean porosity $\\bar{\\phi}$ [-]', color=color)
ax2.get_yaxis().get_major_formatter().set_useOffset(False)
#for tl in ax2.get_yticklabels():
    #tl.set_color(color)

filename = 'diffusivity-sigma0-vs-alpha.pdf'
plt.tight_layout()
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
#print(os.getcwd() + '/' + filename)
print(filename)
