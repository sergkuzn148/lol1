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

smoothed_results = False
contact_forces = False
pressures = False
zflow = False

#sigma0_list = numpy.array([1.0e3, 2.0e3, 4.0e3, 10.0e3, 20.0e3, 40.0e3])
#sigma0 = 10.0e3
sigma0 = float(sys.argv[1])
cvals = [1.0, 0.1]
#cvals = [1.0, 0.1, 0.01]
#cvals = [1.0]

# return a smoothed version of in. The returned array is smaller than the
# original input array
'''
def smooth(in_arr, plus_minus_steps):
    out_arr = numpy.zeros(in_arr.size - 2*plus_minus_steps + 1)
    s = 0
    for i in numpy.arange(in_arr.size):
        if i >= plus_minus_steps and i < plus_minus_steps:
            for i in numpy.arange(-plus_minus_steps, plus_minus_steps+1):
                out_arr[s] += in_arr[s+i]/(2.0*plus_minus_steps)
                s += 1
'''

def smooth(x, window_len=10, window='hanning'):
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


smooth_window = 10

shear_strain = [[], [], [], []]
friction = [[], [], [], []]
if smoothed_results:
    friction_smooth = [[], [], [], []]
dilation = [[], [], [], []]
p_min = [[], [], [], []]
p_mean = [[], [], [], []]
p_max = [[], [], [], []]
f_n_mean = [[], [], [], []]
f_n_max  = [[], [], [], []]
v_f_z_mean  = [[], [], [], []]

fluid=True

# dry shear
#sid = 'shear-sigma0=' + sys.argv[1] + '-hw'
sid = 'halfshear-sigma0=' + sys.argv[1] + '-shear'
sim = sphere.sim(sid)
sim.readlast(verbose=False)
sim.visualize('shear')
shear_strain[0] = sim.shear_strain
#shear_strain[0] = numpy.arange(sim.status()+1)
friction[0] = sim.tau/sim.sigma_eff
if smoothed_results:
    friction_smooth[0] = smooth(friction[0], smooth_window)
dilation[0] = sim.dilation

if contact_forces:
    f_n_mean[0] = numpy.zeros_like(shear_strain[0])
    f_n_max[0]  = numpy.zeros_like(shear_strain[0])
    for i in numpy.arange(sim.status()):
        sim.readstep(i, verbose=False)
        sim.findNormalForces()
        f_n_mean[0][i] = numpy.mean(sim.f_n_magn)
        f_n_max[0][i]  = numpy.max(sim.f_n_magn)

# wet shear
c = 1
for c in numpy.arange(1,len(cvals)+1):
    c_v = cvals[c-1]

    #sid = 'shear-sigma0=' + str(sigma0) + '-c_phi=' + \
                    #str(c_phi) + '-c_v=' + str(c_v) + \
                    #'-hi_mu-lo_visc-hw'
    sid = 'halfshear-sigma0=' + str(sigma0) + '-c=' + str(c_v) + '-shear'
    #sid = 'halfshear-sigma0=' + str(sigma0) + '-c_v=' + str(c_v) +\
            #'-c_a=0.0-velfac=1.0-shear'
    if os.path.isfile('../output/' + sid + '.status.dat'):

        sim = sphere.sim(sid, fluid=fluid)
        shear_strain[c] = numpy.zeros(sim.status())
        friction[c] = numpy.zeros_like(shear_strain[c])
        dilation[c] = numpy.zeros_like(shear_strain[c])
        if smoothed_results:
            friction_smooth[c] = numpy.zeros_like(shear_strain[c])

        sim.readlast(verbose=False)
        sim.visualize('shear')
        shear_strain[c] = sim.shear_strain
        #shear_strain[c] = numpy.arange(sim.status()+1)
        friction[c] = sim.tau/sim.sigma_eff
        dilation[c] = sim.dilation
        if smoothed_results:
            friction_smooth[c] = smooth(friction[c], smooth_window)

        # fluid pressures and particle forces
        if pressures or contact_forces:
            p_mean[c]   = numpy.zeros_like(shear_strain[c])
            p_min[c]    = numpy.zeros_like(shear_strain[c])
            p_max[c]    = numpy.zeros_like(shear_strain[c])
            f_n_mean[c] = numpy.zeros_like(shear_strain[c])
            f_n_max[c]  = numpy.zeros_like(shear_strain[c])
            for i in numpy.arange(sim.status()):
                if pressures:
                    sim.readstep(i, verbose=False)
                    iz_top = int(sim.w_x[0]/(sim.L[2]/sim.num[2]))-1
                    p_mean[c][i] = numpy.mean(sim.p_f[:,:,0:iz_top])/1000
                    p_min[c][i]  = numpy.min(sim.p_f[:,:,0:iz_top])/1000
                    p_max[c][i]  = numpy.max(sim.p_f[:,:,0:iz_top])/1000

                if contact_forces:
                    sim.findNormalForces()
                    f_n_mean[c][i] = numpy.mean(sim.f_n_magn)
                    f_n_max[c][i]  = numpy.max(sim.f_n_magn)

        if zflow:
            v_f_z_mean[c] = numpy.zeros_like(shear_strain[c])
            for i in numpy.arange(sim.status()):
                    v_f_z_mean[c][i] = numpy.mean(sim.v_f[:,:,:,2])

    else:
        print(sid + ' not found')

    # produce VTK files
    #for sid in sids:
        #sim = sphere.sim(sid, fluid=True)
        #sim.writeVTKall()
    c += 1


if zflow:
    fig = plt.figure(figsize=(8,10))
else:
    fig = plt.figure(figsize=(8,8)) # (w,h)
#fig = plt.figure(figsize=(8,12))
#fig = plt.figure(figsize=(8,16))
fig.subplots_adjust(hspace=0.0)

#plt.subplot(3,1,1)
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

if zflow:
    ax1 = plt.subplot(311)
    ax2 = plt.subplot(312, sharex=ax1)
    ax3 = plt.subplot(313, sharex=ax1)
else:
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212, sharex=ax1)
#ax3 = plt.subplot(413, sharex=ax1)
#ax4 = plt.subplot(414, sharex=ax1)
alpha = 0.5
if smoothed_results:
    x1.plot(shear_strain[0], friction_smooth[0], label='dry', linewidth=1,
            alpha=0.5)
else:
    ax1.plot(shear_strain[0], friction[0], label='dry', linewidth=1, alpha=0.5)
ax2.plot(shear_strain[0], dilation[0], label='dry', linewidth=1)
#ax4.plot(shear_strain[0], f_n_mean[0], '-', label='dry', color='blue')
#ax4.plot(shear_strain[0], f_n_max[0], '--', color='blue')

color = ['b','g','r','c']
for c in numpy.arange(1,len(cvals)+1):

    if smoothed_results:
        ax1.plot(shear_strain[c][1:], friction_smooth[c][1:], \
                label='$c$ = %.2f' % (cvals[c-1]), linewidth=1, alpha=0.5)
    else:
        ax1.plot(shear_strain[c][1:], friction[c][1:], \
                label='$c$ = %.2f' % (cvals[c-1]), linewidth=1, alpha=0.5)

    ax2.plot(shear_strain[c][1:], dilation[c][1:], \
            label='$c$ = %.2f' % (cvals[c-1]), linewidth=1)

    if zflow:
        ax3.plot(shear_strain[c][1:], v_f_z_mean[c][1:],
            label='$c$ = %.2f' % (cvals[c-1]), linewidth=1)


    '''
    alpha = 0.5
    ax3.plot(shear_strain[c][1:], p_max[c][1:], '-' + color[c], alpha=alpha)
    ax3.plot(shear_strain[c][1:], p_mean[c][1:], '-' + color[c], \
            label='$c$ = %.2f' % (cvals[c-1]), linewidth=2)
    ax3.plot(shear_strain[c][1:], p_min[c][1:], '-' + color[c], alpha=alpha)

    ax3.fill_between(shear_strain[c][1:], p_min[c][1:], p_max[c][1:], 
            where=p_min[c][1:]<=p_max[c][1:], facecolor=color[c],
            interpolate=True, alpha=alpha)

    ax4.plot(shear_strain[c][1:], f_n_mean[c][1:], '-' + color[c],
            label='$c$ = %.2f' % (cvals[c-1]), linewidth=2)
    ax4.plot(shear_strain[c][1:], f_n_max[c][1:], '--' + color[c])
            #label='$c$ = %.2f' % (cvals[c-1]), linewidth=2)
            '''

#ax4.set_xlabel('Shear strain $\\gamma$ [-]')

ax1.set_ylabel('Shear friction $\\tau/\\sigma\'$ [-]')
ax2.set_ylabel('Dilation $\\Delta h/(2r)$ [-]')
if zflow:
    ax3.set_ylabel('$\\boldsymbol{v}_\\text{f}^z h$ [ms$^{-1}$]')
#ax3.set_ylabel('Fluid pressure $p_\\text{f}$ [kPa]')
#ax4.set_ylabel('Particle contact force $||\\boldsymbol{f}_\\text{p}||$ [N]')

#ax1.set_xlim([200,300])
#ax3.set_ylim([595,608])

plt.setp(ax1.get_xticklabels(), visible=False)
#plt.setp(ax2.get_xticklabels(), visible=False)
#plt.setp(ax3.get_xticklabels(), visible=False)

ax1.grid()
ax2.grid()
if zflow:
    ax3.grid()
#ax4.grid()

legend_alpha=0.5
ax1.legend(loc='lower right', prop={'size':18}, fancybox=True,
        framealpha=legend_alpha)
ax2.legend(loc='lower right', prop={'size':18}, fancybox=True,
        framealpha=legend_alpha)
if zflow:
    ax3.legend(loc='lower right', prop={'size':18}, fancybox=True,
            framealpha=legend_alpha)
#ax4.legend(loc='best', prop={'size':18}, fancybox=True,
        #framealpha=legend_alpha)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
filename = 'shear-' + str(int(sigma0/1000.0)) + 'kPa-stress-dilation.pdf'
#print(os.getcwd() + '/' + filename)
plt.savefig(filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)
