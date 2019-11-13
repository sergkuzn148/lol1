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

#steps = [5, 10, 100]
#steps = [5, 10]
steps = sys.argv[3:]
nsteps_avg = 5 # no. of steps to average over

sigma0 = float(sys.argv[1])
#c_grad_p = 1.0
c_grad_p = float(sys.argv[2])
c_phi = 1.0

#sid = 'shear-sigma0=' + str(sigma0) + '-c_phi=' + \
#                str(c_phi) + '-c_grad_p=' + str(c_grad_p) + '-hi_mu-lo_visc'
sid = 'halfshear-sigma0=' + str(sigma0) + '-c=' + str(c_grad_p) + '-shear'
sim = sphere.sim(sid, fluid=True)
sim.readfirst(verbose=False)

# particle z positions
zpos_p = numpy.zeros((len(steps), sim.np))

# cell midpoint cell positions
zpos_c = numpy.zeros((len(steps), sim.num[2]))
dz = sim.L[2]/sim.num[2]
for i in numpy.arange(sim.num[2]):
    zpos_c[:,i] = i*dz + 0.5*dz

# particle x displacements
xdisp = numpy.zeros((len(steps), sim.np))

# particle-fluid force per particle
f_pf  = numpy.zeros_like(xdisp)

# pressure - hydrostatic pressure
#dev_p = numpy.zeros((len(steps), sim.num[2]))

# mean porosity
phi_bar = numpy.zeros((len(steps), sim.num[2]))

# mean porosity change
dphi_bar = numpy.zeros((len(steps), sim.num[2]))

# mean per-particle values
xdisp_mean = numpy.zeros((len(steps), sim.num[2]))
f_pf_mean = numpy.zeros((len(steps), sim.num[2]))

shear_strain = numpy.zeros(len(steps))

s = 0
for step_str in steps:

    step = int(step_str)

    if os.path.isfile('../output/' + sid + '.status.dat'):

        for substep in numpy.arange(nsteps_avg):

            if step + substep > sim.status():
                raise Exception(
                        'Simulation step %d not available (sim.status = %d).'
                        % (step, sim.status()))

            sim.readstep(step + substep, verbose=False)

            zpos_p[s,:] += sim.x[:,2]/nsteps_avg

            xdisp[s,:] += sim.xyzsum[:,0]/nsteps_avg

            '''
            for i in numpy.arange(sim.np):
                f_pf[s,i] += \
                        sim.f_sum[i].dot(sim.f_sum[i])/nsteps_avg
                        '''
            f_pf[s,:] += sim.f_sum[:,2]

            #dev_p[s,:] += \
                    #numpy.average(numpy.average(sim.p_f, axis=0), axis=0)\
                    #/nsteps_avg

            phi_bar[s,:] += \
                    numpy.average(numpy.average(sim.phi, axis=0), axis=0)\
                    /nsteps_avg

            dphi_bar[s,:] += \
                    numpy.average(numpy.average(sim.dphi, axis=0), axis=0)\
                    /nsteps_avg/sim.time_dt

            shear_strain[s] += sim.shearStrain()/nsteps_avg

        # calculate mean values of xdisp and f_pf
        for iz in numpy.arange(sim.num[2]):
            z_bot = iz*dz
            z_top = (iz+1)*dz
            I = numpy.nonzero((zpos_p[s,:] >= z_bot) & (zpos_p[s,:] < z_top))
            if len(I) > 0:
                xdisp_mean[s,iz] = numpy.mean(xdisp[s,I])
                f_pf_mean[s,iz] = numpy.mean(f_pf[s,I])

    else:
        print(sid + ' not found')
    s += 1


#fig = plt.figure(figsize=(8,4*(len(steps))+1))
fig = plt.figure(figsize=(8,5*(len(steps))+1))

ax = []
for s in numpy.arange(len(steps)):


    ax.append(plt.subplot(len(steps)*100 + 31 + s*3))
    ax.append(plt.subplot(len(steps)*100 + 32 + s*3, sharey=ax[s*4+0]))
    ax.append(plt.subplot(len(steps)*100 + 33 + s*3, sharey=ax[s*4+0]))
    ax.append(ax[s*4+2].twiny())

    ax[s*4+0].plot(xdisp[s], zpos_p[s], ',', color = '#888888')
    ax[s*4+0].plot(xdisp_mean[s], zpos_c[s], color = 'k')

    # remove particles with 0.0 pressure force
    I = numpy.nonzero(numpy.abs(f_pf[s]) > .01)
    f_pf_nonzero = f_pf[s][I]
    zpos_p_nonzero = zpos_p[s][I]
    I = numpy.nonzero(numpy.abs(f_pf_mean[s]) > .01)
    f_pf_mean_nonzero = f_pf_mean[s][I]
    zpos_c_nonzero = zpos_c[s][I]

    #ax[s*4+1].plot(f_pf[s],  zpos_p[s], ',', color = '#888888')
    ax[s*4+1].plot(f_pf_nonzero,  zpos_p_nonzero, ',', color = '#888888')
    #ax[s*4+1].plot(f_pf_mean[s][1:-2], zpos_c[s][1:-2], color = 'k')
    ax[s*4+1].plot(f_pf_mean_nonzero, zpos_c_nonzero, color = 'k')
    #ax[s*4+1].plot([0.0, 0.0], [0.0, sim.L[2]], '--', color='k')

    #ax[s*4+2].plot(dev_p[s]/1000.0, zpos_c[s], 'k')
    #ax[s*4+2].plot(phi_bar[s,1:], zpos_c[s,1:], '-k', linewidth=3)
    ax[s*4+2].plot(phi_bar[s,1:], zpos_c[s,1:], '-k')

    #phicolor = '#888888'
    #ax[s*4+3].plot(phi_bar[s], zpos_c[s], '-', color = phicolor)
    #for tl in ax[s*4+3].get_xticklabels():
        #tl.set_color(phicolor)
    ax[s*4+3].plot(dphi_bar[s,1:], zpos_c[s,1:], '--k')
    #ax[s*4+3].plot(dphi_bar[s,1:], zpos_c[s,1:], '-k', linewidth=3)
    #ax[s*4+3].plot(dphi_bar[s,1:], zpos_c[s,1:], '-w', linewidth=2)

    max_z = numpy.max(zpos_p)
    ax[s*4+0].set_ylim([0, max_z])

    #ax[s*4+1].set_xlim([0.15, 0.46]) # f_pf
    ax[s*4+1].set_xlim([0.235, 0.409]) # f_pf
    ax[s*4+1].set_ylim([0, max_z])

    ax[s*4+2].set_ylim([0, max_z])
    ax[s*4+2].set_xlim([0.33, 0.6])      # phi
    ax[s*4+3].set_xlim([-0.09, 0.024])  # dphi/dt

    #plt.plot(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogx(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogy(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.loglog(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))

    ax[s*4+0].set_ylabel('Vertical position $z$ [m]')
    ax[s*4+0].set_xlabel('$\\boldsymbol{x}^x_\\text{p}$ [m]')
    ax[s*4+1].set_xlabel('$\\boldsymbol{f}^z_\\text{pf}$ [N]')
    #ax[s*4+2].set_xlabel('$\\bar{p_\\text{f}}$ [kPa]')
    #ax[s*4+3].set_xlabel('$\\bar{\\phi}$ [-]', color=phicolor)
    ax[s*4+2].set_xlabel('$\\bar{\\phi}$ [-] (solid)')
    ax[s*4+3].set_xlabel('$\\delta \\bar{\\phi}/\\delta t$ [-] (dashed)')
    plt.setp(ax[s*4+1].get_yticklabels(), visible=False)
    plt.setp(ax[s*4+2].get_yticklabels(), visible=False)

    ax[s*4+0].get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    ax[s*4+1].get_xaxis().set_major_locator(MaxNLocator(nbins=5))
    ax[s*4+2].get_xaxis().set_major_locator(MaxNLocator(nbins=5))

    plt.setp(ax[s*4+0].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*4+1].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*4+2].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*4+3].xaxis.get_majorticklabels(), rotation=90)

    #if s == 0:
        #y = 0.95
    #if s == 1:
        #y = 0.55

    strain_str = 'Shear strain $\\gamma = %.3f$' % (shear_strain[s])
    #fig.text(0.1, y, strain_str, horizontalalignment='left', fontsize=22)
    #ax[s*4+0].annotate(strain_str, xytext=(0,1.1), textcoords='figure fraction',
            #horizontalalignment='left', fontsize=22)
    plt.text(0.05, 1.06, strain_str, horizontalalignment='left', fontsize=22,
            transform=ax[s*4+0].transAxes)
    #ax[s*4+0].set_title(strain_str)

    ax[s*4+0].grid()
    ax[s*4+1].grid()
    ax[s*4+2].grid()
    #ax1.legend(loc='lower right', prop={'size':18})
    #ax2.legend(loc='lower right', prop={'size':18})

plt.tight_layout()
plt.subplots_adjust(wspace = .05)
plt.MaxNLocator(nbins=4)

filename = 'shear-' + str(int(sigma0/1000.0)) + 'kPa-forces.pdf'
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)
