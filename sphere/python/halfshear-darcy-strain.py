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

import seaborn as sns
#sns.set(style='ticks', palette='Set2')
#sns.set(style='ticks', palette='colorblind')
sns.set(style='ticks', palette='Set2')
sns.despine() # remove chartjunk

sigma0_list = [20000.0, 80000.0]
#cvals = ['dry', 1.0, 0.1, 0.01]
#cvals = ['dry', 3.5e-13, 3.5e-15]
cvals = ['dry', 3.5e-13, 3.5e-14, 3.5e-15]
#cvals = ['dry', 1.0]
#step = 1999

for sigma0 in sigma0_list:

    sim = sphere.sim('halfshear-sigma0=' + str(sigma0) + '-shear')
    sim.readfirst(verbose=False)


    # particle z positions
    zpos_p = [[], [], [], []]

    # cell midpoint cell positions
    zpos_c = [[], [], [], []]

    # particle x displacements
    xdisp = [[], [], [], []]
    xdisp_mean = [[], [], [], []]

    s = 0
    for c in cvals:

        if c == 'dry':
            fluid = False
            sid = 'halfshear-sigma0=' + str(sigma0) + '-shear'
        else:
            fluid = True
            sid = 'halfshear-darcy-sigma0=' + str(sigma0) + '-k_c=' + str(c) + \
            '-mu=1.797e-06-velfac=1.0-shear'

        sim = sphere.sim(sid, fluid=fluid)

        if os.path.isfile('../output/' + sid + '.status.dat'):

            sim.readlast(verbose=False)

            zpos_c[s] = numpy.zeros(sim.num[2]*2)
            dz = sim.L[2]/(sim.num[2]*2)
            for i in numpy.arange(sim.num[2]*2):
                zpos_c[s][i] = i*dz + 0.5*dz

            xdisp[s] = numpy.zeros(sim.np)
            xdisp_mean[s] = numpy.zeros(sim.num[2]*2)


            zpos_p[s][:] = sim.x[:,2]

            xdisp[s][:] = sim.xyzsum[:,0]

            #shear_strain[s] += sim.shearStrain()/nsteps_avg

            # calculate mean values of xdisp and f_pf
            for iz in numpy.arange(sim.num[2]*2):
                z_bot = iz*dz
                z_top = (iz+1)*dz
                I = numpy.nonzero((zpos_p[s][:] >= z_bot) & (zpos_p[s][:] < z_top))
                if len(I) > 0:
                    xdisp_mean[s][iz] = numpy.mean(xdisp[s][I])

            # normalize distance
            max_dist = numpy.nanmax(xdisp_mean[s])
            xdisp_mean[s] /= max_dist

        else:
            print(sid + ' not found')
        s += 1


    #fig = plt.figure(figsize=(8,4*(len(steps))+1))
    #fig = plt.figure(figsize=(8,5*(len(steps))+1))
    #fig = plt.figure(figsize=(8/2,6/2))
    fig = plt.figure(figsize=(3.74,3.47)) # 3.14 inch = 80 mm, 3.74 = 95 mm
    #fig = plt.figure(figsize=(8,6))

    ax = []
    #linetype = ['-', '--', '-.']
    #linetype = ['-', '-', '-', '-']
    linetype = ['-', '--', '-.', ':']
    #color = ['b','g','c','y']
    #color = ['k','g','c','y']
    color = ['y','g','c','k']
    #color = ['c','m','y','k']
    for s in numpy.arange(len(cvals)):
    #for s in numpy.arange(len(cvals)-1, -1, -1):

        ax.append(plt.subplot(111))
        #ax.append(plt.subplot(len(steps)*100 + 31 + s*3))
        #ax.append(plt.subplot(len(steps)*100 + 32 + s*3, sharey=ax[s*4+0]))
        #ax.append(plt.subplot(len(steps)*100 + 33 + s*3, sharey=ax[s*4+0]))
        #ax.append(ax[s*4+2].twiny())

        if cvals[s] == 'dry':
            legend = 'dry'
        elif cvals[s] == 3.5e-13:
            legend = 'wet, high permeability'
        elif cvals[s] == 3.5e-14:
            legend = 'wet, interm.\\ permeability'
        elif cvals[s] == 3.5e-15:
            legend = 'wet, low permeability'
        else:
            legend = 'wet, $k_c$ = ' + str(cvals[s]) + ' m$^2$'

        #ax[0].plot(xdisp[s], zpos_p[s], ',', color = '#888888')
        #ax[0].plot(xdisp[s], zpos_p[s], ',', color=color[s], alpha=0.5)
        ax[0].plot(xdisp_mean[s], zpos_c[s], linetype[s],
                label=legend)#,
                #color=color[s],
                #linewidth=2.0)

        ax[0].set_ylabel('Vertical position $z$ [m]')
        #ax[0].set_xlabel('$\\boldsymbol{x}^x_\\text{p}$ [m]')
        ax[0].set_xlabel('Normalized horizontal movement')

        #ax[s*4+0].get_xaxis().set_major_locator(MaxNLocator(nbins=5))
        #ax[s*4+1].get_xaxis().set_major_locator(MaxNLocator(nbins=5))
        #ax[s*4+2].get_xaxis().set_major_locator(MaxNLocator(nbins=5))

        #plt.setp(ax[s*4+0].xaxis.get_majorticklabels(), rotation=90)
        #plt.setp(ax[s*4+1].xaxis.get_majorticklabels(), rotation=90)
        #plt.setp(ax[s*4+2].xaxis.get_majorticklabels(), rotation=90)
        #plt.setp(ax[s*4+3].xaxis.get_majorticklabels(), rotation=90)

        #if s == 0:
            #y = 0.95
        #if s == 1:
            #y = 0.55

        #strain_str = 'Shear strain $\\gamma = %.3f$' % (shear_strain[s])
        #fig.text(0.1, y, strain_str, horizontalalignment='left', fontsize=22)
        #ax[s*4+0].annotate(strain_str, xytext=(0,1.1), textcoords='figure fraction',
                #horizontalalignment='left', fontsize=22)
        #plt.text(0.05, 1.06, strain_str, horizontalalignment='left', fontsize=22,
                #transform=ax[s*4+0].transAxes)
        #ax[s*4+0].set_title(strain_str)

        #ax[s*4+0].grid()
        #ax[s*4+1].grid()
        #ax[s*4+2].grid()
        #ax1.legend(loc='lower right', prop={'size':18})
        #ax2.legend(loc='lower right', prop={'size':18})

    # remove box at top and right
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    # remove ticks at top and right
    ax[0].get_xaxis().tick_bottom()
    ax[0].get_yaxis().tick_left()
    ax[0].get_xaxis().grid(False) # horizontal grid lines
    #ax[0].get_yaxis().grid(True, linestyle='--', linewidth=0.5) # vertical grid lines
    ax[0].get_xaxis().grid(True, linestyle=':', linewidth=0.5) # vertical grid lines
    ax[0].get_yaxis().grid(True, linestyle=':', linewidth=0.5) # vertical grid lines

    # reverse legend order
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].legend(handles[::-1], labels[::-1], loc='best')

    #legend_alpha=0.5
    #ax[0].legend(loc='lower right', prop={'size':18}, fancybox=True, framealpha=legend_alpha)
    #ax[0].legend(loc='best', prop={'size':18}, fancybox=True, framealpha=legend_alpha)
    #ax[0].legend(loc='best')
    #ax[0].grid()
    #ax[0].set_xlim([-0.05, 1.01])
    ax[0].set_xlim([-0.05, 1.05])
    #ax[0].set_ylim([0.0, 0.47])
    ax[0].set_ylim([0.20, 0.47])
    plt.tight_layout()
    plt.subplots_adjust(wspace = .05)
    plt.MaxNLocator(nbins=4)

    filename = 'halfshear-darcy-strain.pdf'
    if sigma0 == 80000.0:
        filename = 'halfshear-darcy-strain-N80.pdf'
    plt.savefig(filename)
    #shutil.copyfile(filename, '/Users/adc/articles/own/2/graphics/' + filename)
    shutil.copyfile(filename, '/home/adc/articles/own/2/graphics/' + filename)
    print(filename)
