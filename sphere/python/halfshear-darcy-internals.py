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
import subprocess

import seaborn as sns
#sns.set(style='ticks', palette='Set2')
#sns.set(style='ticks', palette='colorblind')
#sns.set(style='ticks', palette='Set2')
sns.set(style='ticks', palette='Blues')

#sigma0 = float(sys.argv[1])
sigma0_list = [20000.0, 80000.0]
#k_c = 3.5e-13
#k_c = float(sys.argv[1])
k_c_list = [3.5e-13, 3.5e-14, 3.5e-15]

#if k_c == 3.5e-15:
#    steps = [1232, 1332, 1433, 1534, 1635]
#elif k_c == 3.5e-13:
#    steps = [100, 200, 300, 410, 515]
#else:
#    steps = [10, 50, 100, 1000, 1999]
nsteps_avg = 1 # no. of steps to average over
#nsteps_avg = 100 # no. of steps to average over

steps = [1, 51, 101, 152, 204]

for sigma0 in sigma0_list:
    for k_c in k_c_list:

        sid = 'halfshear-darcy-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
                '-mu=1.797e-06-velfac=1.0-shear'
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

        # fluid permeability
        k = numpy.zeros((len(steps), sim.num[0], sim.num[1], sim.num[2]))
        k_bar = numpy.zeros((len(steps), sim.num[2]))

        # pressure
        p = numpy.zeros((len(steps), sim.num[2]))

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

        #fig = plt.figure(figsize=(8,4*(len(steps))+1))
        #fig = plt.figure(figsize=(8,4.5))
        fig = plt.figure(figsize=(3.74*2,3.00))
        ax = []
        n = 4
        ax.append(plt.subplot(1, n, 1)) # 0: xdisp
        ax.append(plt.subplot(1, n, 2, sharey=ax[0])) # 3: k
        ax.append(plt.subplot(1, n, 3, sharey=ax[0])) # 5: p_f
        ax.append(plt.subplot(1, n, 4, sharey=ax[0])) # 6: f_pf_z

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

                    '''
                    for i in numpy.arange(sim.np):
                        f_pf[s,i] += \
                                sim.f_sum[i].dot(sim.f_sum[i])/nsteps_avg
                                '''
                    f_pf[s,:] += sim.f_p[:,2]

                    dz = sim.L[2]/sim.num[2]
                    wall0_iz = int(sim.w_x[0]/dz)

                    p[s,:] += numpy.average(numpy.average(sim.p_f[:,:,:], axis=0),\
                            axis=0)/nsteps_avg

                    sim.findPermeabilities()
                    k[s,:] += sim.k[:,:,:]/nsteps_avg

                    k_bar[s,:] += \
                            numpy.average(numpy.average(sim.k[:,:,:], axis=0), axis=0)\
                            /nsteps_avg

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
                        f_pf_mean[s,iz] = numpy.mean(f_pf[s,I])

                k_bar[s][0] = k_bar[s][1]



                #ax[0].plot(xdisp[s], zpos_p[s], ',', color = '#888888')
                ax[0].plot(xdisp_mean[s], zpos_c[s], label='$\gamma$ = %.3f' %
                        (shear_strain_start[s]))

                ax[1].semilogx(k_bar[s], zpos_c[s], label='$\gamma$ = %.3f' %
                        (shear_strain_start[s]))

                ax[2].plot(p[s]/1000.0, zpos_c[s], label='$\gamma$ = %.3f' %
                        (shear_strain_start[s]))

                # remove particles with 0.0 pressure force
                I = numpy.nonzero(numpy.abs(f_pf[s]) > .01)
                f_pf_nonzero = f_pf[s][I]
                zpos_p_nonzero = zpos_p[s][I]
                I = numpy.nonzero(numpy.abs(f_pf_mean[s]) > .01)
                f_pf_mean_nonzero = f_pf_mean[s][I]
                zpos_c_nonzero = zpos_c[s][I]
                #f_pf_mean_nonzero = numpy.append(f_pf_mean_nonzero, 0.0)
                #zpos_c_nonzero = numpy.append(zpos_c_nonzero, zpos_c[s][zpos_c_nonzero.size])

                #ax[3].plot(f_pf_nonzero,  zpos_p_nonzero, ',', alpha=0.5,
                        #color='#888888')
                #ax[3].plot(f_pf_mean_nonzero, zpos_c_nonzero, label='$\gamma$ = %.3f' %
                        #(shear_strain_start[s]))
                ax[3].plot(f_pf_mean_nonzero/(4.0*numpy.pi*sim.radius[0]**2)/1e3,
                    zpos_c_nonzero, label='$\gamma$ = %.3f' %
                        (shear_strain_start[s]))

            else:
                print(sid + ' not found')
            s += 1


        #max_z = numpy.max(zpos_p)
        max_z = 0.5
        #max_z = numpy.max(zpos_c[-2])
        ax[0].set_ylim([0, max_z])
        #ax[0].set_xlim([0, 0.5])
        ax[0].set_xlim([0, 0.05])

        if k_c == 3.5e-15:
            #ax[1].set_xlim([1e-14, 1e-12])
            ax[1].set_xlim([1e-16, 1e-14])
        elif k_c == 3.5e-14:
            ax[1].set_xlim([1e-15, 1e-13])
        elif k_c == 3.5e-13:
            #ax[1].set_xlim([1e-12, 1e-10])
            ax[1].set_xlim([1e-14, 1e-12])

        ax[0].set_ylabel('Vertical position $z$ [m]')
        ax[0].set_xlabel('$\\bar{\\boldsymbol{x}}^x_\\text{p}$ [m]')
        ax[1].set_xlabel('$\\bar{k}$ [m$^{2}$]')
        ax[2].set_xlabel('$\\bar{p_\\text{f}}$ [kPa]')
        #ax[3].set_xlabel('$\\boldsymbol{f}^z_\\text{i}$ [N]')
        ax[3].set_xlabel('$\\bar{\boldsymbol{\sigma}}^z_\\text{i}$ [kPa]')

        # align x labels
        #labely = -0.3
        #ax[0].xaxis.set_label_coords(0.5, labely)
        #ax[1].xaxis.set_label_coords(0.5, labely)
        #ax[2].xaxis.set_label_coords(0.5, labely)
        #ax[3].xaxis.set_label_coords(0.5, labely)

        plt.setp(ax[1].get_yticklabels(), visible=False)
        plt.setp(ax[2].get_yticklabels(), visible=False)
        plt.setp(ax[3].get_yticklabels(), visible=False)

        plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax[2].xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax[3].xaxis.get_majorticklabels(), rotation=90)

        '''
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        ax[3].grid()
        '''

        sns.despine() # remove chartjunk

        for i in range(4):
            # vertical grid lines
            ax[i].get_xaxis().grid(True, linestyle=':', linewidth=0.5)
            # horizontal grid lines
            ax[i].get_yaxis().grid(True, linestyle=':', linewidth=0.5)

            if i>0:
                ax[i].spines['left'].set_visible(False)
                ax[i].get_yaxis().set_ticks_position('none')

        legend_alpha=0.5
        #ax[0].legend(loc='lower center', prop={'size':12}, fancybox=True,
                #framealpha=legend_alpha)
        ax[0].legend(loc='lower right')

        #plt.subplots_adjust(wspace = .05)  # doesn't work with tight_layout()
        #plt.MaxNLocator(nbins=1)  # doesn't work?
        ax[0].locator_params(nbins=4)
        ax[2].locator_params(nbins=5)
        ax[3].locator_params(nbins=5)

        plt.tight_layout()

        filename = 'halfshear-darcy-internals-k_c=%.0e.pdf' % (k_c)
        if sigma0 == 80000.0:
            filename = 'halfshear-darcy-internals-k_c=%.0e-N80.pdf' % (k_c)

        plt.savefig(filename)
        #subprocess.call('pdfcrop ' + filename, shell=True)
        shutil.copyfile(filename, '/home/adc/articles/own/2/graphics/' + filename)
        print(filename)
