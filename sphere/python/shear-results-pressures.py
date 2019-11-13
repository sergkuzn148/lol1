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
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams['image.cmap'] = 'bwr'

sigma0 = float(sys.argv[1])
#c_grad_p = 1.0
c_grad_p = [1.0, 0.1]
#c_grad_p = [1.0, 0.1, 0.01, 1e-07]
c_phi = 1.0


#sid = 'shear-sigma0=' + str(sigma0) + '-c_phi=' + \
#                str(c_phi) + '-c_grad_p=' + str(c_grad_p) + '-hi_mu-lo_visc'
sid = 'halfshear-sigma0=' + str(sigma0) + '-c=' + str(c_grad_p[0]) + '-shear'
sim = sphere.sim(sid, fluid=True)
sim.readfirst(verbose=False)

# cell midpoint cell positions
zpos_c = numpy.zeros((len(c_grad_p), sim.num[2]))
dz = sim.L[2]/sim.num[2]
for c in numpy.arange(len(c_grad_p)):
    for i in numpy.arange(sim.num[2]):
        zpos_c[c,i] = i*dz + 0.5*dz


shear_strain = [[], [], [], []]
dev_pres = [[], [], [], []]
pres_static = [[], [], [], []]
pres = [[], [], [], []]

for c in numpy.arange(len(c_grad_p)):
    sim.sid = 'halfshear-sigma0=' + str(sigma0) + '-c=' + str(c_grad_p[c]) \
            + '-shear'

    shear_strain[c] = numpy.zeros(sim.status())
    dev_pres[c] = numpy.zeros((sim.num[2], sim.status()))
    pres_static[c] = numpy.ones_like(dev_pres[c])*sim.p_f[0,0,-1]
    pres[c] = numpy.zeros_like(dev_pres[c])

    for i in numpy.arange(sim.status()):

        sim.readstep(i, verbose=False)

        pres[c][:,i] = numpy.average(numpy.average(sim.p_f, axis=0), axis=0)

        dz = sim.L[2]/sim.num[2]
        wall0_iz = int(sim.w_x[0]/dz)
        for z in numpy.arange(0, wall0_iz+1):
            pres_static[c][z,i] = \
                    (wall0_iz*dz - zpos_c[c,z] + 0.5*dz)\
                    *sim.rho_f*numpy.abs(sim.g[2])\
                    + sim.p_f[0,0,-1]

        shear_strain[c][i] = sim.shearStrain()

    dev_pres[c] = pres[c] - pres_static[c]

#fig = plt.figure(figsize=(8,6))
#fig = plt.figure(figsize=(8,12))
fig = plt.figure(figsize=(8,5*len(c_grad_p)+2))


#cmap = matplotlib.colors.ListedColormap(['b', 'w', 'r'])
#bounds = [min_p, 0, max_p]
#norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

ax = []
for c in numpy.arange(len(c_grad_p)):

    ax.append(plt.subplot(len(c_grad_p), 1, c+1))

    max_p_dev = numpy.max((numpy.abs(numpy.min(dev_pres[c])),
            numpy.max(dev_pres[c])))
    #max_p = numpy.min(dev_pres)
    min_p = -max_p_dev/1000.0
    max_p = max_p_dev/1000.0
    #min_p = -5.0
    #max_p = 5.0

    im1 = ax[c].pcolormesh(shear_strain[c], zpos_c[c], dev_pres[c]/1000.0,
            vmin=min_p, vmax=max_p, rasterized=True)
    #im1 = ax[c].pcolormesh(shear_strain[c], zpos_c[c], dev_pres[c]/1000.0,
            #rasterized=True)
    #im1 = ax[c].pcolormesh(shear_strain[c], zpos_c[c], pres[c]/1000.0,
            #rasterized=True)
    #if c == 0:
    ax[c].set_xlim([0, numpy.max(shear_strain[c])])
    ax[c].set_ylim([zpos_c[0,0], sim.w_x[0]])
    ax[c].set_xlabel('Shear strain $\\gamma$ [-]')
    ax[c].set_ylabel('Vertical position $z$ [m]')

    #plt.text(0.0, 0.15, '$c = %.2f$' % (c_grad_p[c]),\
    #        horizontalalignment='left', fontsize=22,
    #        transform=ax[c].transAxes)
    ax[c].set_title('$c = %.2f$' % (c_grad_p[c]))

    #cb = plt.colorbar(im1, orientation='horizontal')
    cb = plt.colorbar(im1)
    cb.set_label('$p_\\text{f} - p^\\text{hyd}_\\text{f}$ [kPa]')
    cb.solids.set_rasterized(True)

    # annotate plot
    #ax1.text(0.02, 0.15, 'compressive',
            #bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

    #ax1.text(0.12, 0.25, 'dilative',
            #bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

#cb = plt.colorbar(im1, orientation='horizontal')
#cb.set_label('$p_\\text{f} - p^\\text{hyd}_\\text{f}$ [kPa]')
#cb.solids.set_rasterized(True)
'''
ax2 = plt.subplot(312)
im2 = ax2.pcolormesh(shear_strain, zpos_c, pres/1000.0, rasterized=True)
#ax2.set_xlim([0, shear_strain[-1]])
#ax2.set_ylim([zpos_c[0], sim.w_x[0]])
ax2.set_xlabel('Shear strain $\\gamma$ [-]')
ax2.set_ylabel('Vertical position $z$ [m]')
cb2 = plt.colorbar(im2)
cb2.set_label('Pressure $p_\\text{f}$ [kPa]')
cb2.solids.set_rasterized(True)

ax3 = plt.subplot(313)
im3 = ax3.pcolormesh(shear_strain, zpos_c, pres_static/1000.0, rasterized=True)
#ax3.set_xlim([0, shear_strain[-1]])
#ax3.set_ylim([zpos_c[0], sim.w_x[0]])
ax3.set_xlabel('Shear strain $\\gamma$ [-]')
ax3.set_ylabel('Vertical position $z$ [m]')
cb3 = plt.colorbar(im3)
cb3.set_label('Static Pressure $p_\\text{f}$ [kPa]')
cb3.solids.set_rasterized(True)
'''


#plt.MaxNLocator(nbins=4)
plt.tight_layout()
plt.subplots_adjust(wspace = .05)
#plt.MaxNLocator(nbins=4)

filename = 'shear-' + str(int(sigma0/1000.0)) + 'kPa-pressures.pdf'
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)
