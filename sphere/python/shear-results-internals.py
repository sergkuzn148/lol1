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
#nsteps_avg = 5 # no. of steps to average over
nsteps_avg = 100 # no. of steps to average over

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

# particle z velocity
v_z_p = numpy.zeros((len(steps), sim.np))

# fluid z velocity
v_z_f = numpy.zeros((len(steps), sim.num[0], sim.num[1], sim.num[2]))

# pressure - hydrostatic pressure
dev_p = numpy.zeros((len(steps), sim.num[2]))
p     = numpy.zeros((len(steps), sim.num[2]))

# mean per-particle values
v_z_p_bar = numpy.zeros((len(steps), sim.num[2]))
v_z_f_bar = numpy.zeros((len(steps), sim.num[2]))

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

shear_strain_start = numpy.zeros(len(steps))
shear_strain_end = numpy.zeros(len(steps))

s = 0
for step_str in steps:

    step = int(step_str)

    if os.path.isfile('../output/' + sid + '.status.dat'):

        for substep in numpy.arange(nsteps_avg):

            if step + substep > sim.status():
                raise Exception(
                        'Simulation step %d not available (sim.status = %d).'
                        % (step + substep, sim.status()))

            sim.readstep(step + substep, verbose=False)

            zpos_p[s,:] += sim.x[:,2]/nsteps_avg

            xdisp[s,:] += sim.xyzsum[:,0]/nsteps_avg
            v_z_p[s,:] += sim.vel[:,2]/nsteps_avg

            '''
            for i in numpy.arange(sim.np):
                f_pf[s,i] += \
                        sim.f_sum[i].dot(sim.f_sum[i])/nsteps_avg
                        '''
            f_pf[s,:] += sim.f_sum[:,2]

            dz = sim.L[2]/sim.num[2]
            wall0_iz = int(sim.w_x[0]/dz)
            for z in numpy.arange(0, wall0_iz+1):
                dev_p[s,z] += \
                        (numpy.average(sim.p_f[:,:,z]) -
                        ((wall0_iz*dz - zpos_c[s][z] + 0.5*dz)
                                *sim.rho_f*numpy.abs(sim.g[2])\
                        + sim.p_f[0,0,-1])) \
                        /nsteps_avg

            p[s,:] += numpy.average(numpy.average(sim.p_f[:,:,:], axis=0),\
                    axis=0)/nsteps_avg

            v_z_f[s,:] += sim.v_f[:,:,:,2]/nsteps_avg

            v_z_f_bar[s,:] += \
                    numpy.average(numpy.average(sim.v_f[:,:,:,2], axis=0), axis=0)\
                    /nsteps_avg

            phi_bar[s,:] += \
                    numpy.average(numpy.average(sim.phi, axis=0), axis=0)\
                    /nsteps_avg

            dphi_bar[s,:] += \
                    numpy.average(numpy.average(sim.dphi, axis=0), axis=0)\
                    /nsteps_avg/sim.time_dt


            if substep == 0:
                shear_strain_start[s] = sim.shearStrain()
            else:
                shear_strain_end[s] = sim.shearStrain()

        # calculate mean values of xdisp and f_pf
        for iz in numpy.arange(sim.num[2]):
            z_bot = iz*dz
            z_top = (iz+1)*dz
            I = numpy.nonzero((zpos_p[s,:] >= z_bot) & (zpos_p[s,:] < z_top))
            if len(I) > 0:
                xdisp_mean[s,iz] = numpy.mean(xdisp[s,I])
                v_z_p_bar[s,iz] = numpy.mean(v_z_p[s,I])
                f_pf_mean[s,iz] = numpy.mean(f_pf[s,I])

    else:
        print(sid + ' not found')
    s += 1

#fig = plt.figure(figsize=(8,4*(len(steps))+1))
#fig = plt.figure(figsize=(8,5*(len(steps))+1))
fig = plt.figure(figsize=(16,5*(len(steps))+1))

def color(c):
    return 'black'
    if c == 1.0:
        return 'green'
    elif c == 0.1:
        return 'red'
    elif c == 0.01:
        return 'cyan'
    else:
        return 'blue'

ax = []
for s in numpy.arange(len(steps)):

    #strain_str = 'Shear strain\n $\\gamma = %.3f$' % (shear_strain[s])

    if s == 0:
        strain_str = 'Dilating state\n$\\gamma = %.2f$ to $%.2f$\n$c = %.2f$' % \
        (shear_strain_start[s], shear_strain_end[s], c_grad_p)
    else:
        strain_str = 'Steady state\n$\\gamma = %.2f$ to $%.2f$\n$c = %.2f$' % \
        (shear_strain_start[s], shear_strain_end[s], c_grad_p)

    n = 7
    if s == 0:
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+1)) # 0: xdisp
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+2, sharey=ax[s*n+0])) # 1: phi
        ax.append(ax[s*n+1].twiny()) # 2: dphi/dt
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+3, sharey=ax[s*n+0])) # 3: v_z^p
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+4, sharey=ax[s*n+0])) # 4: p_f
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+5, sharey=ax[s*n+0])) # 5: f_pf_z
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+6, sharey=ax[s*n+0])) # 6: v_z^f
    else:
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+1, sharex=ax[0])) # 0: xdisp
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+2, sharey=ax[s*n+0],
                sharex=ax[1])) # 1: phi
        ax.append(ax[s*n+1].twiny()) # 2: dphi/dt
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+3, sharey=ax[s*n+0],
                sharex=ax[3])) # 3: v_z^p
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+4, sharey=ax[s*n+0],
                sharex=ax[4])) # 4: p_f
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+5, sharey=ax[s*n+0],
                sharex=ax[5])) # 5: f_pf_z
        ax.append(plt.subplot(len(steps), n-1, s*(n-1)+6, sharey=ax[s*n+0],
                sharex=ax[6])) # 6: v_z^f

    ax[s*n+0].plot(xdisp[s], zpos_p[s], ',', color = '#888888')
    ax[s*n+0].plot(xdisp_mean[s], zpos_c[s], color=color(c_grad_p))

    #ax[s*4+2].plot(dev_p[s]/1000.0, zpos_c[s], 'k')
    #ax[s*4+2].plot(phi_bar[s,1:], zpos_c[s,1:], '-k', linewidth=3)
    ax[s*n+1].plot(phi_bar[s,1:], zpos_c[s,1:], '-', color=color(c_grad_p))

    #phicolor = '#888888'
    #ax[s*4+3].plot(phi_bar[s], zpos_c[s], '-', color = phicolor)
    #for tl in ax[s*4+3].get_xticklabels():
        #tl.set_color(phicolor)
    ax[s*n+2].plot(dphi_bar[s,1:], zpos_c[s,1:], '--', color=color(c_grad_p))
    #ax[s*4+3].plot(dphi_bar[s,1:], zpos_c[s,1:], '-k', linewidth=3)
    #ax[s*4+3].plot(dphi_bar[s,1:], zpos_c[s,1:], '-w', linewidth=2)

    ax[s*n+3].plot(v_z_p[s]*100.0, zpos_p[s], ',', alpha=0.5,
            color='#888888')
            #color=color(c_grad_p))
    ax[s*n+3].plot(v_z_p_bar[s]*100.0, zpos_c[s], color=color(c_grad_p))
    #ax[s*n+0].plot([0.0,0.0], [0.0, sim.L[2]], '--', color='k')

    # hydrostatic pressure distribution
    #ax[s*n+4].plot(dev_p[s]/1000.0, zpos_c[s], color=color(c_grad_p))
    ax[s*n+4].plot(p[s]/1000.0, zpos_c[s], color=color(c_grad_p))
    dz = sim.L[2]/sim.num[2]
    wall0_iz = int(sim.w_x[0]/dz)
    y_top = wall0_iz*dz + 0.5*dz
    x_top = sim.p_f[0,0,-1]
    y_bot = 0.0
    x_bot = x_top + (wall0_iz*dz - zpos_c[s][0] + 0.5*dz)*sim.rho_f*numpy.abs(sim.g[2])
    ax[s*n+4].plot([x_top/1000.0, x_bot/1000.0], [y_top, y_bot], '--', color='k')
    #ax[s*n+1].set_title(strain_str)
    #ax[s*n+1].set_title('   ')

    # remove particles with 0.0 pressure force
    I = numpy.nonzero(numpy.abs(f_pf[s]) > .01)
    f_pf_nonzero = f_pf[s][I]
    zpos_p_nonzero = zpos_p[s][I]
    I = numpy.nonzero(numpy.abs(f_pf_mean[s]) > .01)
    f_pf_mean_nonzero = f_pf_mean[s][I]
    zpos_c_nonzero = zpos_c[s][I]

    ax[s*n+5].plot(f_pf_nonzero,  zpos_p_nonzero, ',', alpha=0.5,
            color='#888888')
            #color=color(c_grad_p))
    #ax[s*4+1].plot(f_pf_mean[s][1:-2], zpos_c[s][1:-2], color = 'k')
    ax[s*n+5].plot(f_pf_mean_nonzero, zpos_c_nonzero, color=color(c_grad_p))
    #ax[s*4+1].plot([0.0, 0.0], [0.0, sim.L[2]], '--', color='k')

    ax[s*n+6].plot(v_z_f_bar[s]*100.0, zpos_c[s], color=color(c_grad_p))
    #ax[s*n+2].plot([0.0,0.0], [0.0, sim.L[2]], '--', color='k')


    #plt.plot(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogx(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogy(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.loglog(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))

    #for x in numpy.arange(sim.num[0]):
        #for y in numpy.arange(sim.num[1]):
            #ax[s*n+2].plot(v_z_f[s,x,y,:], zpos_c[s], ',', color = '#888888')


    #phicolor = '#888888'
    #ax[s*n+3].plot(phi_bar[s], zpos_c[s], '-', color = phicolor)
    #for tl in ax[s*n+3].get_xticklabels():
        #tl.set_color(phicolor)

    max_z = numpy.max(zpos_p)
    ax[s*n+0].set_ylim([0, max_z])
    #ax[s*n+1].set_ylim([0, max_z])
    #ax[s*n+2].set_ylim([0, max_z])

    #ax[s*n+0].set_xlim([-0.01,0.01])
    #ax[s*n+0].set_xlim([-0.005,0.005])
    #ax[s*n+0].set_xlim([-0.25,0.75])
    #ax[s*n+4].set_xlim([595,625])   # p_f
    #ax[s*n+2].set_xlim([-0.0005,0.0005])
    #ax[s*n+2].set_xlim([-0.08,0.08])

    #ax[s*4+1].set_xlim([0.15, 0.46]) # f_pf
    #ax[s*n+1].set_xlim([0.235, 0.409]) # f_pf

    ax[s*n+1].set_xlim([0.33, 0.6])     # phi
    ax[s*n+2].set_xlim([-0.09, 0.035])  # dphi/dt
    ax[s*n+3].set_xlim([-1.50, 1.50])   # v_z_p
    ax[s*n+5].set_xlim([5.0, 8.0])      # f_z_pf


    #plt.plot(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogx(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.semilogy(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))
    #plt.loglog(dpdz[c], K[c], 'o-', label='$c$ = %.2f' % (cvals[c]))

    ax[s*n+0].set_ylabel('Vertical position $z$ [m]')
    ax[s*n+0].set_xlabel('$\\boldsymbol{x}^x_\\text{p}$ [m]')
    ax[s*n+1].set_xlabel('$\\bar{\\phi}$ [-] (solid)')
    ax[s*n+2].set_xlabel('$\\delta \\bar{\\phi}/\\delta t$ [-] (dashed)')
    ax[s*n+3].set_xlabel('$\\boldsymbol{v}^z_\\text{p}$ [cms$^{-1}$]')
    ax[s*n+4].set_xlabel('$\\bar{p_\\text{f}}$ [kPa]')
    #ax[s*n+4].set_xlabel('$\\bar{p_\\text{f}} - p_\\text{hyd}$ [kPa]')
    ax[s*n+5].set_xlabel('$\\boldsymbol{f}^z_\\text{pf}$ [N]')
    ax[s*n+6].set_xlabel('$\\bar{\\boldsymbol{v}}^z_\\text{f}$ [cms$^{-1}$]')

    # align x labels
    labely = -0.3
    ax[s*n+0].xaxis.set_label_coords(0.5, labely)
    ax[s*n+1].xaxis.set_label_coords(0.5, labely)
    #ax[s*n+2].xaxis.set_label_coords(0.5, labely)
    ax[s*n+3].xaxis.set_label_coords(0.5, labely)
    ax[s*n+4].xaxis.set_label_coords(0.5, labely)
    ax[s*n+5].xaxis.set_label_coords(0.5, labely)
    ax[s*n+6].xaxis.set_label_coords(0.5, labely)

    plt.setp(ax[s*n+1].get_yticklabels(), visible=False)
    plt.setp(ax[s*n+2].get_yticklabels(), visible=False)
    plt.setp(ax[s*n+3].get_yticklabels(), visible=False)
    plt.setp(ax[s*n+4].get_yticklabels(), visible=False)
    plt.setp(ax[s*n+5].get_yticklabels(), visible=False)
    plt.setp(ax[s*n+6].get_yticklabels(), visible=False)

    #nbins = 4
    #ax[s*n+0].get_xaxis().set_major_locator(MaxNLocator(nbins=nbins))
    #ax[s*n+1].get_xaxis().set_major_locator(MaxNLocator(nbins=nbins))
    #ax[s*n+2].get_xaxis().set_major_locator(MaxNLocator(nbins=nbins))

    plt.setp(ax[s*n+0].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*n+1].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*n+2].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*n+3].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*n+4].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*n+5].xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax[s*n+6].xaxis.get_majorticklabels(), rotation=90)

    #if s == 0:
        #y = 0.95
    #if s == 1:
        #y = 0.55

    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #ax[s*n+0].ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
    #ax[s*n+1].ticklabel_format(style='sci', axis='x', scilimits=(-3,3))
    #ax[s*n+2].ticklabel_format(style='sci', axis='x', scilimits=(-3,3))

    #fig.text(0.1, y, strain_str, horizontalalignment='left', fontsize=22)
    #ax[s*4+0].annotate(strain_str, xytext=(0,1.1), textcoords='figure fraction',
            #horizontalalignment='left', fontsize=22)
    #plt.text(0.05, 1.06, strain_str, horizontalalignment='left', fontsize=22,
            #transform=ax[s*n+0].transAxes)
    #ax[s*4+0].set_title(strain_str)

    #ax[s*n+0].set_title('a')
    #ax[s*n+1].set_title('b')
    #ax[s*n+3].set_title('c')
    #ax[s*n+4].set_title('d')
    #ax[s*n+5].set_title('e')
    #ax[s*n+6].set_title('f')

    ax[s*n+0].grid()
    ax[s*n+1].grid()
    #ax[s*n+2].grid()
    ax[s*n+3].grid()
    ax[s*n+4].grid()
    ax[s*n+5].grid()
    ax[s*n+6].grid()
    #ax1.legend(loc='lower right', prop={'size':18})
    #ax2.legend(loc='lower right', prop={'size':18})

    #strain_str = 'Shear strain $\\gamma = %.3f$' % (shear_strain[s])
    #fig.text(0.1, y, strain_str, horizontalalignment='left', fontsize=22)
    #ax[s*4+0].annotate(strain_str, xytext=(0,1.1), textcoords='figure fraction',
            #horizontalalignment='left', fontsize=22)
    plt.text(-0.38, 1.10, strain_str, horizontalalignment='left', fontsize=22,
            transform=ax[s*n+0].transAxes)

#plt.title('  ')
plt.MaxNLocator(nbins=4)
    #ax1.legend(loc='lower right', prop={'size':18})
    #ax2.legend(loc='lower right', prop={'size':18})

#plt.title('  ')
#plt.MaxNLocator(nbins=4)
#plt.subplots_adjust(wspace = .05)
#plt.subplots_adjust(hspace = 1.05)
plt.tight_layout()
#plt.MaxNLocator(nbins=4)

filename = 'shear-' + str(int(sigma0/1000.0)) + 'kPa-internals.pdf'
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)
