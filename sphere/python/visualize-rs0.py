#!/usr/bin/env python
import sphere
import numpy
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
#import seaborn

jobname_prefix = 'rs0-'

verbose = True

# Visualization settings
outformat = 'pdf'
plotforcechains = False
calculateforcechains = False
# calculateforcechains = True
calculateforcechainhistory = False
legend_alpha = 0.7
linewidth = 0.5
smooth_friction = False
smooth_window = 10


# Simulation parameter values
effective_stresses = [10e3, 20e3, 100e3, 200e3, 1000e3, 2000e3]
velfacs = [0.1, 1.0, 10.0]
mu_s_vals = [0.5]
mu_d_vals = [0.5]

# return a smoothed version of in. The returned array is smaller than the
# original input array
def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming',
            'bartlett', 'blackman' flat window will produce a moving average
            smoothing.

    output:
        the smoothed signal

    example:

    import numpy as np
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead
    of a string
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(
            "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = numpy.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    # print(len(s))

    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = getattr(numpy, window)(window_len)
    y = numpy.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


def rateStatePlot(sid):

    matplotlib.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]


    rasterized = True  # rasterize colored areas (pcolormesh and colorbar)
    # izfactor = 4  # factor for vertical discretization in xvel
    izfactor = 1  # factor for vertical discretization in poros


    #############
    # DATA READ #
    #############

    sim = sphere.sim(sid, fluid=False)
    sim.readfirst()

    # nsteps = 2
    # nsteps = 10
    # nsteps = 400
    nsteps = sim.status()

    t = numpy.empty(nsteps)

    # Stress, pressure and friction
    sigma_def = numpy.empty_like(t)
    sigma_eff = numpy.empty_like(t)
    tau_def = numpy.empty_like(t)
    tau_eff = numpy.empty_like(t)
    p_f_bar = numpy.empty_like(t)
    p_f_top = numpy.empty_like(t)
    mu = numpy.empty_like(t)  # shear friction

    # shear velocity plot
    v = numpy.empty_like(t)  # velocity
    I = numpy.empty_like(t)  # inertia number

    # displacement and mean porosity plot
    xdisp = numpy.empty_like(t)
    phi_bar = numpy.empty_like(t)

    # mean horizontal porosity plot
    poros = numpy.empty((sim.num[2], nsteps))
    xvel = numpy.zeros((sim.num[2]*izfactor, nsteps))
    zpos_c = numpy.empty(sim.num[2]*izfactor)
    dz = sim.L[2]/(sim.num[2]*izfactor)
    for i in numpy.arange(sim.num[2]*izfactor):
        zpos_c[i] = i*dz + 0.5*dz

    # Contact statistics plot
    n = numpy.empty(nsteps)
    coordinationnumber = numpy.empty(nsteps)
    nkept = numpy.empty(nsteps)


    for i in numpy.arange(nsteps):

        sim.readstep(i+1, verbose=verbose)  # use step 1 to n

        t[i] = sim.currentTime()

        sigma_def[i] = sim.currentNormalStress('defined')
        sigma_eff[i] = sim.currentNormalStress('effective')
        tau_def[i] = sim.shearStress('defined')
        tau_eff[i] = sim.shearStress('effective')
        mu[i] = tau_eff[i]/sigma_eff[i]
        #mu[i] = tau_eff[i]/sigma_def[i]

        I[i] = sim.inertiaParameterPlanarShear()

        v[i] = sim.shearVelocity()
        xdisp[i] = sim.shearDisplacement()

        #poros[:, i] = numpy.average(numpy.average(sim.phi, axis=0), axis=0)

        # calculate mean values of xvel
        dz = sim.L[2]/(sim.num[2]*izfactor)
        for iz in numpy.arange(sim.num[2]*izfactor):
            z_bot = iz*dz
            z_top = (iz+1)*dz
            idx = numpy.nonzero((sim.x[:, 2] >= z_bot) & (sim.x[:, 2] < z_top))
            #import ipdb; ipdb.set_trace()
            if idx[0].size > 0:
                # xvel[iz,i] = numpy.mean(numpy.abs(sim.vel[I,0]))
                # xvel[iz, i] = numpy.mean(sim.vel[idx, 0])
                xvel[iz, i] = numpy.mean(sim.vel[idx, 0])/sim.shearVelocity()

        loaded_contacts = 0
        if calculateforcechains:
            if i > 0 and calculateforcechainhistory:
                loaded_contacts_prev = numpy.copy(loaded_contacts)
                pairs_prev = numpy.copy(sim.pairs)

            loaded_contacts = sim.findLoadedContacts(
                threshold=sim.currentNormalStress()*2.)
            # sim.currentNormalStress()/1000.)
            n[i] = numpy.size(loaded_contacts)

            sim.findCoordinationNumber()
            coordinationnumber[i] = sim.findMeanCoordinationNumber()

            if calculateforcechainhistory:
                nfound = 0
                if i > 0:
                    for a in loaded_contacts[0]:
                        for b in loaded_contacts_prev[0]:
                            if (sim.pairs[:, a] == pairs_prev[:, b]).all():
                                nfound += 1

                nkept[i] = nfound
                print nfound

            print coordinationnumber[i]


    if calculateforcechains:
        numpy.savetxt(sid + '-fc.txt', (n, nkept, coordinationnumber))
    else:
        if plotforcechains:
            n, nkept, coordinationnumber = numpy.loadtxt(sid + '-fc.txt')

    # Transform time from model time to real time [s]
    #t = t/t_DEM_to_t_real

    # integrate velocities to displacement along x (xdispint)
    #  Taylor two term expansion
    xdispint = numpy.zeros_like(t)
    v_limit = 2.78e-3  # 1 m/hour (WIP)
    dt = (t[1] - t[0])
    dt2 = dt*2.
    for i in numpy.arange(t.size):
        if i > 0 and i < t.size-1:
            acc = (numpy.min([v[i+1], v_limit]) - numpy.min([v[i-1], v_limit]))/dt2
            xdispint[i] = xdispint[i-1] +\
                numpy.min([v[i], v_limit])*dt + 0.5*acc*dt**2
        elif i == t.size-1:
            xdispint[i] = xdispint[i-1] + numpy.min([v[i], v_limit])*dt


    ############
    # PLOTTING #
    ############
    bbox_x = 0.03
    bbox_y = 0.96
    verticalalignment = 'top'
    horizontalalignment = 'left'
    fontweight = 'bold'
    bbox = {'facecolor': 'white', 'alpha': 1.0, 'pad': 3}

    # Time in days
    #t = t/(60.*60.*24.)

    nplots = 4
    fig = plt.figure(figsize=[3.5, 8.0/4.0*nplots])

    # ax1: v, ax2: I
    ax1 = plt.subplot(nplots, 1, 1)
    lns0 = ax1.plot(t, v, '-k', label="$v$",
                    linewidth=linewidth)
    # lns1 = ax1.plot(t, sigma_eff/1000., '-k', label="$\\sigma'$")
    # lns2 = ax1.plot(t, tau_def/1000., '-r', label="$\\tau$")
    # ns2 = ax1.plot(t, tau_def/1000., '-r')
    #lns3 = ax1.plot(t, tau_eff/1000., '-r', label="$\\tau'$", linewidth=linewidth)

    ax1.set_ylabel('Shear velocity $v$ [ms$^{-1}$]')

    ax2 = ax1.twinx()
    ax2color = 'blue'
    # lns4 = ax2.plot(t, p_f_top/1000.0 + 80.0, '-',
            # color=ax2color,
            # label='$p_\\text{f}^\\text{forcing}$')
    # lns5 = ax2.semilogy(t, I, ':',
    lns5 = ax2.semilogy(t, I, '--',
                        color=ax2color,
                        label='$I$', linewidth=linewidth)
    ax2.set_ylabel('Inertia number $I$ [-]')
    ax2.yaxis.label.set_color(ax2color)
    for tl in ax2.get_yticklabels():
        tl.set_color(ax2color)
        #ax2.legend(loc='upper right')
    #lns = lns0+lns1+lns2+lns3+lns4+lns5
    #lns = lns0+lns1+lns2+lns3+lns5
    #lns = lns1+lns3+lns5
    #lns = lns0+lns3+lns5
    lns = lns0+lns5
    labs = [l.get_label() for l in lns]
    # ax2.legend(lns, labs, loc='upper right', ncol=3,
            # fancybox=True, framealpha=legend_alpha)
    #ax1.set_ylim([-30, 200])
    #ax2.set_ylim([-115, 125])

    # ax1.text(bbox_x, bbox_y, 'A',
    ax1.text(bbox_x, bbox_y, 'a',
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            fontweight=fontweight, bbox=bbox,
            transform=ax1.transAxes)


    # ax3: mu, ax4: unused
    ax3 = plt.subplot(nplots, 1, 2, sharex=ax1)
    #ax3.semilogy(t, mu, 'k', linewidth=linewidth)
    alpha=1.0
    if smooth_friction:
        alpha=0.5
    ax3.plot(t, mu, 'k', alpha=alpha, linewidth=linewidth)
    if smooth_friction:
        # smoothed
        ax3.plot(t, smooth(mu, smooth_window), linewidth=2)
                # label='', linewidth=1,
                # alpha=alpha, color=color[c])
    ax3.set_ylabel('Bulk friction $\\mu = \\tau\'/N\'$ [-]')
    #ax3.set_ylabel('Bulk friction $\\mu = \\tau\'/N$ [-]')

    # ax3.text(bbox_x, bbox_y, 'B',
    ax3.text(bbox_x, bbox_y, 'b',
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            fontweight=fontweight, bbox=bbox,
            transform=ax3.transAxes)


    # ax7: n, ax8: unused
    ax7 = plt.subplot(nplots, 1, 3, sharex=ax1)
    if plotforcechains:
        ax7.plot(t[:n.size], coordinationnumber, 'k', linewidth=linewidth)
    ax7.set_ylabel('Coordination number $\\bar{n}$ [-]')
    #ax7.semilogy(t, n - nkept, 'b', label='$\Delta n_\\text{heavy}$')
    #ax7.set_ylim([1.0e1, 2.0e4])
    #ax7.set_ylim([-0.2, 9.8])
    ax7.set_ylim([-0.2, 5.2])

    # ax7.text(bbox_x, bbox_y, 'D',
    ax7.text(bbox_x, bbox_y, 'c',
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            fontweight=fontweight, bbox=bbox,
            transform=ax7.transAxes)


    # ax9: porosity or xvel, ax10: unused
    #ax9 = plt.subplot(nplots, 1, 5, sharex=ax1)
    ax9 = plt.subplot(nplots, 1, 4, sharex=ax1)
    #poros_min = 0.375
    #poros_max = 0.45
    poros[:, 0] = poros[:, 2]  # remove erroneous porosity increase
    #cmap = matplotlib.cm.get_cmap('Blues_r')
    cmap = matplotlib.cm.get_cmap('afmhot')
    # im9 = ax9.pcolormesh(t, zpos_c, poros,
    #zpos_c = zpos_c[:-1]

    xvel = xvel[:-1]
    # xvel[xvel < 0.0] = 0.0  # ignore negative velocities
    # im9 = ax9.pcolormesh(t, zpos_c, poros,
    im9 = ax9.pcolormesh(t, zpos_c, xvel,
                        cmap=cmap,
                        #vmin=poros_min, vmax=poros_max,
                        #norm=matplotlib.colors.LogNorm(vmin=1.0e-8, vmax=xvel.max()),
                        shading='goraud',
                        rasterized=rasterized)
    ax9.set_ylim([zpos_c[0], sim.w_x[0]])
    ax9.set_ylabel('Vertical position $z$ [m]')

    cbaxes = fig.add_axes([0.32, 0.1, 0.4, 0.01])  # x,y,w,h

    # ax9.add_patch(matplotlib.patches.Rectangle(
        # (3.0, 0.04), # x,y
        # 15., # dx
        # .15, # dy
        # fill=True,
        # linewidth=1,
        # facecolor='white'))
    # ax9.add_patch(matplotlib.patches.Rectangle(
    # (1.5, 0.04),  # x,y
    # 7.,  # dx
    # .15,  # dy
    #    fill=True,
    #    linewidth=1,
    #    facecolor='white',
    #    alpha=legend_alpha))

    cb9 = plt.colorbar(im9, cax=cbaxes,
                    #ticks=[poros_min, poros_min + 0.5*(poros_max-poros_min), poros_max],
                    #ticks=[xvel.min(), xvel.min() + 0.5*(xvel.max()-xvel.min()), xvel.max()],
                    orientation='horizontal',
                    # extend='min',
                    cmap=cmap)
    # cmap.set_under([8./255., 48./255., 107./255.]) # for poros
    # cmap.set_under([1.0e-3, 1.0e-3, 1.0e-3]) # for xvel
    # cb9.outline.set_color('w')
    cb9.outline.set_edgecolor('w')

    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb9.locator = tick_locator
    cb9.update_ticks()

    #cb9.set_label('Mean horizontal porosity [-]')
    cb9.set_label('Norm. avg. horiz. vel. [-]', color='w')
    '''
    ax9.text(0.5, 0.4, 'Mean horizontal porosity [-]\\\\',
            horizontalalignment='center',
            verticalalignment='center',
            bbox={'facecolor':'white', 'alpha':1.0, 'pad':3})
    '''
    cb9.solids.set_rasterized(rasterized)

    # change text color of colorbar to white
    #axes_obj = plt.getp(im9, 'axes')
    #plt.setp(plt.getp(axes_obj, 'yticklabels'), color='w')
    #plt.setp(plt.getp(axes_obj, 'xticklabels'), color='w')
    #plt.setp(plt.getp(cb9.ax.axes, 'yticklabels'), color='w')
    cb9.ax.yaxis.set_tick_params(color='w')
    # cb9.yaxis.label.set_color(ax2color)
    for tl in cb9.ax.get_xticklabels():
        tl.set_color('w')
    cb9.ax.yaxis.set_tick_params(color='w')

    # ax9.text(bbox_x, bbox_y, 'E',
    ax9.text(bbox_x, bbox_y, 'd',
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            fontweight=fontweight, bbox=bbox,
            transform=ax9.transAxes)


    plt.setp(ax1.get_xticklabels(), visible=False)
    #plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    #plt.setp(ax4.get_xticklabels(), visible=False)
    #plt.setp(ax5.get_xticklabels(), visible=False)
    #plt.setp(ax6.get_xticklabels(), visible=False)
    plt.setp(ax7.get_xticklabels(), visible=False)
    #plt.setp(ax8.get_xticklabels(), visible=False)

    ax1.set_xlim([numpy.min(t), numpy.max(t)])
    #ax2.set_ylim([1e-5, 1e-3])
    ax3.set_ylim([-0.2, 1.2])

    ax9.set_xlabel('Time [s]')
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05)

    filename = sid + '-combined.' + outformat
    plt.savefig(filename)
    plt.close()
    #shutil.copyfile(filename, '/home/adc/articles/own/3/graphics/' + filename)
    print(filename)

# Loop through parameter values
for effective_stress in effective_stresses:
    for velfac in velfacs:
        for mu_s in mu_s_vals:
            for mu_d in mu_s_vals:

                jobname = jobname_prefix + '{}Pa-v={}-mu_s={}-mu_d={}'.format(
                    effective_stress,
                    velfac,
                    mu_s,
                    mu_d)

                print(jobname)
                sim = sphere.sim(jobname)
                #sim.visualize('shear')
                rateStatePlot(jobname)
