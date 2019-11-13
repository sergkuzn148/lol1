#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import shutil

import os
import numpy
import sphere
from permeabilitycalculator import *
import matplotlib.pyplot as plt

#dp_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])
dp_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3])
cvals = [1.0, 0.1, 0.01]
c_phi = 1.0

K = [[], [], []]
dpdz = [[], [], []]
Q = [[], [], []]
phi_bar = [[], [], []]
Re = [[], [], []]
fp_fsum = [[], [], []]


c = 0
for c_grad_p in cvals:

    sids = []
    for dp in dp_list:
        if c_grad_p == 1.0:
            sids.append('permeability-dp=' + str(dp))
        else:
            sids.append('permeability-dp=' + str(dp) + '-c_phi=' + \
                    str(c_phi) + '-c_grad_p=' + str(c_grad_p))

    K[c] = numpy.zeros(len(sids))
    dpdz[c] = numpy.zeros_like(K[c])
    Q[c] = numpy.zeros_like(K[c])
    phi_bar[c] = numpy.zeros_like(K[c])
    Re[c] = numpy.zeros_like(K[c])
    fp_fsum[c] = numpy.zeros_like(K[c])
    i = 0

    for sid in sids:
        if os.path.isfile('../output/' + sid + '.status.dat'):
            pc = PermeabilityCalc(sid, plot_evolution=False,
                    print_results=False, verbose=False)
            K[c][i] = pc.conductivity()
            pc.findPressureGradient()
            pc.findCrossSectionalFlux()
            dpdz[c][i] = pc.dPdL[2]
            Q[c][i] = pc.Q[2]
            pc.findMeanPorosity()
            #pc.plotEvolution()
            phi_bar[c][i] = pc.phi_bar

            sim = sphere.sim(sid, fluid=True)
            sim.readlast(verbose=False)
            Re[c][i] = numpy.mean(sim.ReynoldsNumber())

            #sim.writeVTKall()

            # find magnitude of fluid pressure force and total interaction force
            '''
            fp_magn = numpy.empty(sim.np)
            fsum_magn = numpy.empty(sim.np)
            for i in numpy.arange(sim.np):
                fp_magn[i] = sim.f_p[i,:].dot(sim.f_p[i,:])
                fsum_magn[i] = sim.f_sum[i,:].dot(sim.f_sum[i,:])

            fp_fsum[c][i] = numpy.mean(fp_magn/fsum_magn)
            # interaction forces not written in these old output files!
            '''


        else:
            print(sid + ' not found')

        i += 1

    # produce VTK files
    #for sid in sids:
        #sim = sphere.sim(sid, fluid=True)
        #sim.writeVTKall()
    c += 1

fig = plt.figure(figsize=(8,12))

ax1 = plt.subplot(3,1,1)
ax2 = plt.subplot(3,1,2, sharex=ax1)
ax3 = plt.subplot(3,1,3, sharex=ax1)
#ax4 = plt.subplot(4,1,4, sharex=ax1)
colors = ['g', 'r', 'c', 'y']
#lines = ['-', '--', '-.', ':']
lines = ['-', '-', '-', '-']
#markers = ['o', 'x', '^', '+']
markers = ['x', 'x', 'x', 'x']
for c in range(len(cvals)):
    dpdz[c] /= 1000.0
    #plt.plot(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogx(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogy(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    ax1.loglog(dpdz[c], K[c], label='$c$ = %.2f' % (cvals[c]),
            linestyle=lines[c], marker=markers[c], color=colors[c])
    ax2.semilogx(dpdz[c], phi_bar[c], label='$c$ = %.2f' % (cvals[c]),
            linestyle=lines[c], marker=markers[c], color=colors[c])
    ax3.loglog(dpdz[c], Re[c], label='$c$ = %.2f' % (cvals[c]),
            linestyle=lines[c], marker=markers[c], color=colors[c])
    #ax4.loglog(dpdz[c], fp_fsum[c], label='$c$ = %.2f' % (cvals[c]),
            #linestyle=lines[c], marker=markers[c], color='black')

ax1.set_ylabel('Hydraulic conductivity $K$ [ms$^{-1}$]')
#ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax2.set_ylabel('Mean porosity $\\bar{\\phi}$ [-]')

ax3.set_ylabel('Mean Reynolds number $\\bar{Re}$ [-]')

#ax4.set_ylabel('$\\bar{\\boldsymbol{f}_{\\Delta p}/\\bar{\\boldsymbol{f}_\\text{pf}}$ [-]')

ax3.set_xlabel('Pressure gradient $\\Delta p/\\Delta z$ [kPa m$^{-1}$]')

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

ax1.grid()
ax2.grid()
ax3.grid()
#ax4.grid()

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

ax1.legend(loc='best', prop={'size':18}, fancybox=True, framealpha=0.5)
ax2.legend(loc='best', prop={'size':18}, fancybox=True, framealpha=0.5)
ax3.legend(loc='best', prop={'size':18}, fancybox=True, framealpha=0.5)

plt.tight_layout()
plt.subplots_adjust(hspace = .12)
filename = 'permeability-dpdz-vs-K-vs-c.pdf'
#print(os.getcwd() + '/' + filename)
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)
