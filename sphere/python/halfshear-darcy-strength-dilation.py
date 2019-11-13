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
smooth_window = 100
#smooth_window = 200

#sigma0_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])
sigma0_list = [20000.0, 80000.0]
#k_c_vals = [3.5e-13, 3.5e-15]
k_c = 3.5e-15

k_c_vals = ['dry', 3.5e-13, 3.5e-14, 3.5e-15]

mu_f = 1.797e-06

velfac = 1.0

# return a smoothed version of in. The returned array is smaller than the
# original input array
def smooth(x, window_len=10, window='hanning'):
#def smooth(x, window_len=10, window='flat'):
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


for sigma0 in sigma0_list:
    shear_strain = [[], [], [], []]
    friction = [[], [], [], []]
    dilation = [[], [], [], []]
    p_min = [[], [], [], []]
    p_mean = [[], [], [], []]
    p_max = [[], [], [], []]
    f_n_mean = [[], [], [], []]
    f_n_max  = [[], [], [], []]
    v_f_z_mean  = [[], [], [], []]

    fluid=True

    for c in numpy.arange(0,len(k_c_vals)):
        k_c = k_c_vals[c]

        if k_c == 'dry':
            sid = 'halfshear-sigma0=' + str(sigma0) + '-shear'
            fluid = False
        else:
            sid = 'halfshear-darcy-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
                    '-mu=' + str(mu_f) + '-velfac=' + str(velfac) + '-shear'
            fluid = True
        #sid = 'halfshear-sigma0=' + str(sigma0) + '-c_v=' + str(c_v) +\
                #'-c_a=0.0-velfac=1.0-shear'
        if os.path.isfile('../output/' + sid + '.status.dat'):

            sim = sphere.sim(sid, fluid=fluid)
            n = sim.status()
            #n = 20
            shear_strain[c] = numpy.zeros(n)
            friction[c] = numpy.zeros_like(shear_strain[c])
            dilation[c] = numpy.zeros_like(shear_strain[c])

            # fluid pressures and particle forces
            if fluid:
                p_mean[c]   = numpy.zeros_like(shear_strain[c])
                p_min[c]    = numpy.zeros_like(shear_strain[c])
                p_max[c]    = numpy.zeros_like(shear_strain[c])
            if contact_forces:
                f_n_mean[c] = numpy.zeros_like(shear_strain[c])
                f_n_max[c]  = numpy.zeros_like(shear_strain[c])

            for i in numpy.arange(n):

                sim.readstep(i, verbose=False)

                shear_strain[c][i] = sim.shearStrain()
                friction[c][i] = sim.shearStress('effective')/sim.currentNormalStress('defined')
                dilation[c][i] = sim.w_x[0]

                if fluid and pressures:
                    iz_top = int(sim.w_x[0]/(sim.L[2]/sim.num[2]))-1
                    p_mean[c][i] = numpy.mean(sim.p_f[:,:,0:iz_top])/1000
                    p_min[c][i]  = numpy.min(sim.p_f[:,:,0:iz_top])/1000
                    p_max[c][i]  = numpy.max(sim.p_f[:,:,0:iz_top])/1000

                if contact_forces:
                    sim.findNormalForces()
                    f_n_mean[c][i] = numpy.mean(sim.f_n_magn)
                    f_n_max[c][i]  = numpy.max(sim.f_n_magn)

            if fluid and zflow:
                v_f_z_mean[c] = numpy.zeros_like(shear_strain[c])
                for i in numpy.arange(n):
                        v_f_z_mean[c][i] = numpy.mean(sim.v_f[:,:,:,2])

            dilation[c] =\
                    (dilation[c] - dilation[c][0])/(numpy.mean(sim.radius)*2.0)

        else:
            print(sid + ' not found')

        # produce VTK files
        #for sid in sids:
            #sim = sphere.sim(sid, fluid=True)
            #sim.writeVTKall()


    if zflow or pressures:
        #fig = plt.figure(figsize=(8,10))
        #fig = plt.figure(figsize=(3.74, 2*3.74))
        fig = plt.figure(figsize=(2*3.74, 2*3.74))
    else:
        fig = plt.figure(figsize=(8,8)) # (w,h)
    #fig = plt.figure(figsize=(8,12))
    #fig = plt.figure(figsize=(8,16))

    #plt.subplot(3,1,1)
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    for c in numpy.arange(0,len(k_c_vals)):

        if zflow or pressures:
            ax1 = plt.subplot(3, len(k_c_vals), 1+c)
            ax2 = plt.subplot(3, len(k_c_vals), 5+c, sharex=ax1)
            if c > 0:
                ax3 = plt.subplot(3, len(k_c_vals), 9+c, sharex=ax1)
        else:
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212, sharex=ax1)
        #ax3 = plt.subplot(413, sharex=ax1)
        #ax4 = plt.subplot(414, sharex=ax1)
        #alpha = 0.5
        alpha = 1.0
        #ax1.plot(shear_strain[0], friction[0], label='dry', linewidth=1, alpha=alpha)
        #ax2.plot(shear_strain[0], dilation[0], label='dry', linewidth=1)
        #ax4.plot(shear_strain[0], f_n_mean[0], '-', label='dry', color='blue')
        #ax4.plot(shear_strain[0], f_n_max[0], '--', color='blue')

        #color = ['b','g','r','c']
        #color = ['g','r','c']
        color = sns.color_palette()
        #for c, mu_f in enumerate(mu_f_vals):
        #for c in numpy.arange(len(mu_f_vals)-1, -1, -1):
        k_c = k_c_vals[c]

        fluid = True
        if k_c == 'dry':
            label = 'dry'
            fluid = False
        elif numpy.isclose(k_c, 3.5e-13, atol=1.0e-16):
            label = 'high permeability'
        elif numpy.isclose(k_c, 3.5e-14, atol=1.0e-16):
            label = 'interm.\\ permeability'
        elif numpy.isclose(k_c, 3.5e-15, atol=1.0e-16):
            label = 'low permeability'
        else:
            label = str(k_c)

        # unsmoothed
        ax1.plot(shear_strain[c][1:], friction[c][1:], \
                label=label, linewidth=1,
                alpha=0.3, color=color[c], clip_on=False)
                #alpha=0.2, color='gray', clip_on=False)
                #alpha=alpha, color=color[c])

        # smoothed
        ax1.plot(shear_strain[c][1:-50], smooth(friction[c], smooth_window)[1:-50],\
                label=label, linewidth=1,
                alpha=alpha, color=color[c])


        ax2.plot(shear_strain[c], dilation[c], \
                label=label, linewidth=1,
                color=color[c])

        if zflow:
            ax3.plot(shear_strain[c], v_f_z_mean[c],
                label=label, linewidth=1)

        if fluid and pressures:
            ax3.plot(shear_strain[c], p_max[c], ':', color=color[c], alpha=0.5,
                     linewidth=0.5)

            ax3.plot(shear_strain[c], p_mean[c], '-', color=color[c], \
                    label=label, linewidth=1)

            ax3.plot(shear_strain[c], p_min[c], ':', color=color[c], alpha=0.5,
                     linewidth=0.5)


            #ax3.fill_between(shear_strain[c], p_min[c], p_max[c],
            #        where=p_min[c]<=p_max[c], facecolor=color[c], edgecolor='None',
            #        interpolate=True, alpha=0.3)

            #ax4.plot(shear_strain[c][1:], f_n_mean[c][1:], '-' + color[c],
                    #label='$c$ = %.2f' % (cvals[c-1]), linewidth=2)
            #ax4.plot(shear_strain[c][1:], f_n_max[c][1:], '--' + color[c])
                #label='$c$ = %.2f' % (cvals[c-1]), linewidth=2)



        #ax4.set_xlabel('Shear strain $\\gamma$ [-]')
        if fluid and (zflow or pressures):
            ax3.set_xlabel('Shear strain $\\gamma$ [-]')
        else:
            ax2.set_xlabel('Shear strain $\\gamma$ [-]')

        if c == 0:
            ax1.set_ylabel('Shear friction $\\tau/\\sigma_0$ [-]')
            #ax1.set_ylabel('Shear stress $\\tau$ [kPa]')
            ax2.set_ylabel('Dilation $\\Delta h/(2r)$ [-]')

        if c == 1:
            if zflow:
                ax3.set_ylabel('$\\boldsymbol{v}_\\text{f}^z h$ [ms$^{-1}$]')
            if pressures:
                ax3.set_ylabel('Fluid pressure $\\bar{p}_\\text{f}$ [kPa]')
            #ax4.set_ylabel('Particle contact force $||\\boldsymbol{f}_\\text{p}||$ [N]')

        #ax1.set_xlim([200,300])
        #ax3.set_ylim([595,608])

        plt.setp(ax1.get_xticklabels(), visible=False)
        if fluid and (zflow or pressures):
            plt.setp(ax2.get_xticklabels(), visible=False)
        #plt.setp(ax2.get_xticklabels(), visible=False)
        #plt.setp(ax3.get_xticklabels(), visible=False)

        '''
        ax1.grid()
        ax2.grid()
        if zflow or pressures:
            ax3.grid()
        #ax4.grid()
        '''

        if c == 0: # left
            # remove box at top and right
            ax1.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            #ax1.spines['left'].set_visible(True)
            # remove ticks at top and right
            ax1.get_xaxis().set_ticks_position('none')
            ax1.get_yaxis().set_ticks_position('none')
            ax1.get_yaxis().tick_left()

            # remove box at top and right
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(True)
            # remove ticks at top and right
            ax2.get_xaxis().set_ticks_position('none')
            ax2.get_yaxis().set_ticks_position('none')
            ax2.get_yaxis().tick_left()
            ax2.get_xaxis().tick_bottom()

            '''
            # remove box at top and right
            ax3.spines['top'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            ax3.spines['bottom'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            # remove ticks at top and right
            ax3.get_xaxis().set_ticks_position('none')
            ax3.get_yaxis().set_ticks_position('none')
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax3.get_yticklabels(), visible=False)
            '''

        elif c == len(k_c_vals)-1: # right
            # remove box at top and right
            ax1.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['right'].set_visible(True)
            ax1.spines['left'].set_visible(False)
            # remove ticks at top and right
            ax1.get_xaxis().set_ticks_position('none')
            ax1.get_yaxis().set_ticks_position('none')
            ax1.get_yaxis().tick_right()

            # remove box at top and right
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(True)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            # remove ticks at top and right
            ax2.get_xaxis().set_ticks_position('none')
            ax2.get_yaxis().set_ticks_position('none')
            #ax2.get_yaxis().tick_left()
            ax2.get_yaxis().tick_right()

            # remove box at top and right
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(True)
            ax3.spines['left'].set_visible(False)
            # remove ticks at top and right
            ax3.get_xaxis().set_ticks_position('none')
            ax3.get_yaxis().set_ticks_position('none')
            ax3.get_xaxis().tick_bottom()
            ax3.get_yaxis().tick_right()

        else: # middle
            # remove box at top and right
            ax1.spines['top'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['left'].set_visible(False)
            # remove ticks at top and right
            ax1.get_xaxis().set_ticks_position('none')
            ax1.get_yaxis().set_ticks_position('none')
            #ax1.get_yaxis().tick_left()
            plt.setp(ax1.get_yticklabels(), visible=False)

            # remove box at top and right
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.spines['bottom'].set_visible(False)
            ax2.spines['left'].set_visible(False)
            # remove ticks at top and right
            ax2.get_xaxis().set_ticks_position('none')
            ax2.get_yaxis().set_ticks_position('none')
            #ax2.get_yaxis().tick_left()
            plt.setp(ax2.get_yticklabels(), visible=False)

            # remove box at top and right
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            ax3.spines['left'].set_visible(False)
            # remove ticks at top and right
            ax3.get_xaxis().set_ticks_position('none')
            ax3.get_yaxis().set_ticks_position('none')
            ax3.get_xaxis().tick_bottom()
            #ax3.get_yaxis().tick_left()
            plt.setp(ax3.get_yticklabels(), visible=False)
            if c == 1:
                #ax3.get_yaxis().tick_left()
                ax3.spines['left'].set_visible(True)


        # vertical grid lines
        ax1.get_xaxis().grid(True, linestyle=':', linewidth=0.5)
        ax2.get_xaxis().grid(True, linestyle=':', linewidth=0.5)
        if fluid:
            ax3.get_xaxis().grid(True, linestyle=':', linewidth=0.5)


        # horizontal grid lines
        ax1.get_yaxis().grid(True, linestyle=':', linewidth=0.5)
        ax2.get_yaxis().grid(True, linestyle=':', linewidth=0.5)
        if fluid:
            ax3.get_yaxis().grid(True, linestyle=':', linewidth=0.5)

        ax1.set_title(label)
            #ax1.legend(loc='best')
        #legend_alpha=0.5
        #ax1.legend(loc='upper right', prop={'size':18}, fancybox=True,
                #framealpha=legend_alpha)
        #ax2.legend(loc='lower right', prop={'size':18}, fancybox=True,
                #framealpha=legend_alpha)
        #if zflow or pressures:
            #ax3.legend(loc='upper right', prop={'size':18}, fancybox=True,
                    #framealpha=legend_alpha)
        #ax4.legend(loc='best', prop={'size':18}, fancybox=True,
                #framealpha=legend_alpha)

        #ax1.set_xlim([0.0, 0.09])
        #ax2.set_xlim([0.0, 0.09])
        #ax2.set_xlim([0.0, 0.2])

        #ax1.set_ylim([-7, 45])
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.0])
        if fluid:
            ax3.set_ylim([-150., 150.])

        #ax1.set_ylim([0.0, 1.0])
        #if pressures:
            #ax3.set_ylim([-1400, 900])
            #ax3.set_ylim([-200, 200])
            #ax3.set_xlim([0.0, 0.09])

    #plt.tight_layout()
    #plt.subplots_adjust(hspace=0.05)
    plt.subplots_adjust(hspace=0.15)
    #filename = 'shear-' + str(int(sigma0/1000.0)) + 'kPa-stress-dilation.pdf'
    filename = 'halfshear-darcy.pdf'
    if sigma0 == 80000.0:
        filename = 'halfshear-darcy-N80.pdf'
    #print(os.getcwd() + '/' + filename)
    plt.savefig(filename)
    shutil.copyfile(filename, '/home/adc/articles/own/2/graphics/' + filename)
    plt.close()
    print(filename)
