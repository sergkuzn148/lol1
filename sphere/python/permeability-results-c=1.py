#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})

import os
import numpy
import sphere
from permeabilitycalculator import *
import matplotlib.pyplot as plt

dp_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])

sids = []
for dp in dp_list:
    sids.append('permeability-dp=' + str(dp))

K = numpy.empty(len(sids))
dp = numpy.empty_like(K)
Q = numpy.empty_like(K)
phi_bar = numpy.empty_like(K)
i = 0

for sid in sids:
    pc = PermeabilityCalc(sid, plot_evolution=False, print_results=False,
            verbose=False)
    K[i] = pc.conductivity()
    pc.findPressureGradient()
    pc.findCrossSectionalFlux()
    dpdz[i] = pc.dPdL[2]
    Q[i] = pc.Q[2]
    pc.findMeanPorosity()
    phi_bar[i] = pc.phi_bar

    i += 1

# produce VTK files
#for sid in sids:
    #sim = sphere.sim(sid, fluid=True)
    #sim.writeVTKall()

fig = plt.figure()

#plt.subplot(3,1,1)
plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [kPa m$^{-1}$]')
plt.ylabel('Hydraulic conductivity $K$ [ms$^{-1}$]')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
dpdz /= 1000.0
plt.plot(dpdz, K, 'o-k')
plt.grid()

#plt.subplot(3,1,2)
#plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [Pa m$^{-1}$]')
#plt.ylabel('Hydraulic flux $Q$ [m$^3$s$^{-1}$]')
#plt.plot(dpdz, Q, '+')
#plt.grid()

#plt.subplot(3,1,3)
#plt.xlabel('Pressure gradient $\\Delta p/\\Delta z$ [Pa m$^{-1}$]')
#plt.ylabel('Mean porosity $\\bar{\\phi}$ [-]')
#plt.plot(dpdz, phi_bar, '+')
#plt.grid()

plt.tight_layout()
filename = 'permeability-dpdz-vs-K.pdf'
plt.savefig(filename)
print(os.getcwd() + '/' + filename)
plt.savefig(filename)
