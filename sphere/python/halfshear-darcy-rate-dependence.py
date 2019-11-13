#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
import shutil

import os
import sys
import numpy
import sphere
import matplotlib.pyplot as plt
import scipy.optimize

matplotlib.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

sids =\
        ['halfshear-darcy-sigma0=80000.0-k_c=3.5e-13-mu=1.04e-07-ss=10000.0-A=70000.0-f=0.2',
         'halfshear-darcy-sigma0=80000.0-k_c=3.5e-15-mu=1.04e-07-ss=10000.0-A=70000.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4000.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4125.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4250.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4375.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4500.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4625.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4750.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4875.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4000.0-f=0.1',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4500.0-f=0.1',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=3000.0-A=4000.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=3000.0-A=4250.0-f=0.2',
         'halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=3000.0-A=4500.0-f=0.2',
         ]
        #['halfshear-darcy-sigma0=80000.0-k_c=3.5e-13-mu=1.797e-06-ss=10000.0-A=70000.0-f=0.2']
sids = \
    ['halfshear-darcy-sigma0=80000.0-k_c=3.5e-13-mu=1.04e-07-ss=10000.0-A=70000.0-f=0.2']
outformat = 'pdf'
fluid = True
#threshold = 100.0 # [N]


def creep_rheology1(friction, n):
    ''' return strain rate from friction (tau/N) value '''
    return friction**n

def creep_rheology2(friction, n, A):
    ''' return strain rate from friction (tau/N) value '''
    return A*friction**n


for sid in sids:

    print sid
    sim = sphere.sim(sid, fluid=fluid)

    #nsteps = 2
    #nsteps = 10
    nsteps = sim.status()

    t = numpy.empty(nsteps)

    tau = numpy.empty(sim.status())
    N = numpy.empty(sim.status())
    #v = numpy.empty(sim.status())
    shearstrainrate = numpy.empty(sim.status())
    shearstrain = numpy.empty(sim.status())
    dilation = numpy.empty(sim.status())
    for i in numpy.arange(sim.status()):
        sim.readstep(i+1, verbose=False)
        t_DEM_to_t_real = sim.mu[0]/1.787e-3
        #tau = sim.shearStress()
        tau[i] = sim.w_tau_x # defined shear stress
        #tau[i] = sim.shearStress()[0] # measured shear stress along x
        N[i] = sim.currentNormalStress() # defined normal stress
        #v[i] = sim.shearVel()
        shearstrainrate[i] = sim.shearStrainRate()*t_DEM_to_t_real
        shearstrain[i] = sim.shearStrain()
        t[i] = sim.currentTime()

        if i == 0:
            initial_height = sim.w_x[0]

        dilation[i] = sim.w_x[0] - initial_height

    # remove nonzero sliding velocities and their associated values
    #idx = numpy.nonzero(v)
    idx = numpy.nonzero(shearstrainrate)
    #v_nonzero = v[idx]
    shearstrainrate_nonzero = shearstrainrate[idx]
    tau_nonzero = tau[idx]
    N_nonzero = N[idx]
    shearstrain_nonzero = shearstrain[idx]

    t_nonzero = t[idx]

    ### "Critical" state fit
    # The algorithm uses the Levenberg-Marquardt algorithm through leastsq
    idxfit = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1*t_DEM_to_t_real) &
            (((t_nonzero >  6.0) & (t_nonzero <  8.0)) |
            ((t_nonzero > 11.0) & (t_nonzero < 13.0)) |
            ((t_nonzero > 16.0) & (t_nonzero < 18.0)) |
            ((t_nonzero > 21.0) & (t_nonzero < 23.0)) |
            ((t_nonzero > 26.0) & (t_nonzero < 28.0)) |
            ((t_nonzero > 31.0) & (t_nonzero < 33.0)) |
            ((t_nonzero > 36.0) & (t_nonzero < 38.0)) |
            ((t_nonzero > 41.0) & (t_nonzero < 43.0)) |
            ((t_nonzero > 46.0) & (t_nonzero < 48.0)) |
            ((t_nonzero > 51.0) & (t_nonzero < 53.0)) |
            ((t_nonzero > 56.0) & (t_nonzero < 58.0))))

    #popt, pvoc = scipy.optimize.curve_fit(
            #creep_rheology, tau/N, shearstrainrate)
    popt, pvoc = scipy.optimize.curve_fit(
            creep_rheology1, tau_nonzero[idxfit]/N_nonzero[idxfit],
            shearstrainrate_nonzero[idxfit])
    popt2, pvoc2 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit]/N_nonzero[idxfit],
            shearstrainrate_nonzero[idxfit])
    print '# Creep rheology 1'
    print popt
    print pvoc
    n = popt[0] # stress exponent
    #A = popt[1] # stress exponent
    A = 1.

    print '# Creep rheology 2'
    print popt2
    print pvoc2
    n = popt2[0] # stress exponent
    A = popt2[1] # stress exponent
    #A = 1.


    '''
    # Investigate n and A evolution from cycle to cycle
    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero >  1.0) & (t_nonzero <  3.0))))
    popt_0, pvoc_0 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero >  6.0) & (t_nonzero <  8.0))))
    popt_1, pvoc_1 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 11.0) & (t_nonzero < 13.0))))
    popt_2, pvoc_2 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 16.0) & (t_nonzero < 18.0))))
    popt_3, pvoc_3 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 21.0) & (t_nonzero < 23.0))))
    popt_4, pvoc_4 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 26.0) & (t_nonzero < 28.0))))
    popt_5, pvoc_5 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 31.0) & (t_nonzero < 33.0))))
    popt_6, pvoc_6 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 36.0) & (t_nonzero < 38.0))))
    popt_7, pvoc_7 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 41.0) & (t_nonzero < 43.0))))
    popt_8, pvoc_8 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 46.0) & (t_nonzero < 48.0))))
    popt_9, pvoc_9 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 51.0) & (t_nonzero < 53.0))))
    popt_10, pvoc_10 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    idxfit_ = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            (shearstrainrate_nonzero < 0.1) &
            (((t_nonzero > 56.0) & (t_nonzero < 58.0))))
    popt_11, pvoc_11 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit_]/N_nonzero[idxfit_],
            shearstrainrate_nonzero[idxfit_])

    n_list = [
        popt_0[0],
        popt_1[0],
        popt_2[0],
        popt_3[0],
        popt_4[0],
        popt_5[0],
        popt_6[0],
        popt_7[0],
        popt_8[0],
        popt_9[0],
        popt_10[0],
        popt_11[0]]

    A_list = [
        popt_0[1],
        popt_1[1],
        popt_2[1],
        popt_3[1],
        popt_4[1],
        popt_5[1],
        popt_6[1],
        popt_7[1],
        popt_8[1],
        popt_9[1],
        popt_10[1],
        popt_11[1]]
    cycle = numpy.arange(1, len(n_list)+1)

    print(n_list)
    print(A_list)
    '''


    friction = tau_nonzero/N_nonzero
    x_min = numpy.floor(numpy.min(friction))
    x_max = numpy.ceil(numpy.max(friction)) + 0.05
    friction_fit =\
            numpy.linspace(numpy.min(tau_nonzero[idxfit]/N_nonzero[idxfit]),
                    numpy.max(tau_nonzero[idxfit]/N_nonzero[idxfit]),
                    100)
    strainrate_fit = A*friction_fit**n
    '''
    ### Consolidated state fit
    #idxfit2 = numpy.nonzero((tau_nonzero/N_nonzero < 0.38) &
            #(shearstrainrate_nonzero < 0.1) &
            #((t_nonzero > 0.0) & (t_nonzero < 2.5)))
    idxfit2 = numpy.nonzero((shearstrain_nonzero < 0.1) &
            (tau_nonzero/N_nonzero < 0.38))

    popt2, pvoc2 = scipy.optimize.curve_fit(
            creep_rheology2, tau_nonzero[idxfit2]/N_nonzero[idxfit2],
            shearstrainrate_nonzero[idxfit2])
    print '# Consolidated state'
    print popt2
    print pvoc2
    n2 = popt2[0] # stress exponent
    A2 = popt2[1] # prefactor

    friction_fit2 =\
            numpy.linspace(numpy.min(tau_nonzero[idxfit2]/N_nonzero[idxfit2]),
                    numpy.max(tau_nonzero[idxfit2]/N_nonzero[idxfit2]),
                    100)
    strainrate_fit2 = A2*friction_fit2**n2
    '''


    ### Plotting
    fig = plt.figure(figsize=(3.5,2.5))
    ax1 = plt.subplot(111)
    #ax1.semilogy(N/1000., v)
    #ax1.semilogy(tau_nonzero/N_nonzero, v_nonzero, '+k')
    #ax1.plot(tau/N, v, '.')
    #CS = ax1.scatter(friction, v_nonzero, c=shearstrain_nonzero,
            #linewidth=0)
    CS = ax1.scatter(friction, shearstrainrate_nonzero,
            #c=t_nonzero, linewidth=0.2,
            c=shearstrain_nonzero, linewidth=0.05,
            cmap=matplotlib.cm.get_cmap('afmhot'))
    #CS = ax1.scatter(friction[idxfit], shearstrainrate_nonzero[idxfit],
            #c=shearstrain_nonzero[idxfit], linewidth=0.2,
            #cmap=matplotlib.cm.get_cmap('afmhot'))

    ## plastic limit
    x = [0.26, 0.26]
    y = ax1.get_ylim()
    limitcolor = '#333333'
    ax1.plot(x, y, '--', linewidth=2, color=limitcolor)
    ax1.text(x[0]+0.03, 2.0e-4*t_DEM_to_t_real,
            'Yield strength', rotation=90, color=limitcolor,
            bbox={'fc':'#ffffff', 'pad':3, 'alpha':0.7})


    ## Fit
    ax1.plot(friction_fit, strainrate_fit)
    #ax1.plot(friction_fit2, strainrate_fit2)
    ax1.annotate('$\\dot{{\\gamma}}$ = ' + '{:.2e} s$^{{-1}}$'.format(A) +
            '$(\\tau/N)^{{{:.2f}}}$'.format(n),
            xy = (friction_fit[40], strainrate_fit[40]),
            xytext = (0.31+0.05, 2.0e-9*t_DEM_to_t_real),
            arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0.1,
                width=1, headwidth=4, frac=0.2),)
            #xytext = (friction_fit[50]+0.15, strainrate_fit[50]-1.0e-5))#,
            #arrowprops=dict(facecolor='black', shrink=0.05),)

    ax1.set_yscale('log')
    ax1.set_xlim([x_min, x_max])
    #y_min = numpy.min(v_nonzero)*0.5
    #y_max = numpy.max(v_nonzero)*2.0
    y_min = numpy.min(shearstrainrate_nonzero)*0.5
    y_max = numpy.max(shearstrainrate_nonzero)*2.0
    ax1.set_ylim([y_min, y_max])
    #ax1.set_xlim([friction_fit[0], friction_fit[-1]])
    #ax1.set_ylim([strainrate_fit[0], strainrate_fit[-1]])

    cb = plt.colorbar(CS)
    cb.set_label('Shear strain $\\gamma$ [-]')

    #ax1.set_xlabel('Effective normal stress [kPa]')
    ax1.set_xlabel('Friction $\\tau/N$ [-]')
    #ax1.set_ylabel('Shear velocity [m/s]')
    ax1.set_ylabel('Shear strain rate $\\dot{\\gamma}$ [s$^{-1}$]')

    plt.tight_layout()
    filename = sid + '-rate-dependence.' + outformat
    plt.savefig(filename)
    print(filename)
    plt.close()
    #shutil.copyfile(filename, '/home/adc/articles/own/3/graphics/' + filename)

    ## dilation vs. rate
    '''
    fig = plt.figure(figsize=(3.5,2.5))
    ax1 = plt.subplot(111)
    CS = ax1.scatter(friction, dilation[idx],
            #tau_nonzero[idxfit2]/N_nonzero[idxfit2],
            shearstrainrate_nonzero[idxfit2],
            c=shearstrain_nonzero[idxfit2], linewidth=0.05,
            #c=shearstrain_nonzero, linewidth=0.05,
            cmap=matplotlib.cm.get_cmap('afmhot'))

    ## plastic limit
    x = [0.3, 0.3]
    y = ax1.get_ylim()
    limitcolor = '#333333'
    ax1.plot(x, y, '--', linewidth=2, color=limitcolor)
    ax1.text(x[0]+0.03, 2.0e-4,
            'Yield strength', rotation=90, color=limitcolor,
            bbox={'fc':'#ffffff', 'pad':3, 'alpha':0.7})
    '''

    ## Fit
    ax1.plot(friction_fit, strainrate_fit)
    #ax1.plot(friction_fit2, strainrate_fit2)
    ax1.annotate('$\\dot{\\gamma} = (\\tau/N)^{6.4}$',
            xy = (friction_fit[40], strainrate_fit[40]),
            xytext = (0.32+0.05, 2.0e-9),
            arrowprops=dict(facecolor='blue', edgecolor='blue', shrink=0.1,
                width=1, headwidth=4, frac=0.2),)
            #xytext = (friction_fit[50]+0.15, strainrate_fit[50]-1.0e-5))#,
            #arrowprops=dict(facecolor='black', shrink=0.05),)
    '''
    '''

    ax1.set_yscale('log')
    ax1.set_xlim([x_min, x_max])
    y_min = numpy.min(shearstrainrate_nonzero)*0.5
    y_max = numpy.max(shearstrainrate_nonzero)*2.0
    ax1.set_ylim([y_min, y_max])

    cb = plt.colorbar(CS)
    cb.set_label('Shear strain $\\gamma$ [-]')

    #ax1.set_xlabel('Effective normal stress [kPa]')
    ax1.set_xlabel('Friction $\\tau/N$ [-]')
    #ax1.set_ylabel('Shear velocity [m/s]')
    ax1.set_ylabel('Dilation $\\Delta h$ [m]')

    plt.tight_layout()
    filename = sid + '-rate-dependence-dilation.' + outformat
    plt.savefig(filename)
    plt.close()
    shutil.copyfile(filename, '/home/adc/articles/own/3/graphics/' + filename)
    print(filename)



    fig = plt.figure(figsize=(3.5,2.5))
    ax1 = plt.subplot(111)
    #ax1.semilogy(N/1000., v)
    #ax1.semilogy(tau_nonzero/N_nonzero, v_nonzero, '+k')
    #ax1.plot(tau/N, v, '.')
    #CS = ax1.scatter(friction, v_nonzero, c=shearstrain_nonzero,
            #linewidth=0)
    CS = ax1.plot(cycle, A_list, '-k')

    ax1.set_xlabel('Forcing cycle [-]')
    #ax1.set_ylabel('Shear velocity [m/s]')
    ax1.set_ylabel('$A$ [s$^{-1}$]')

    ax2 = ax1.twinx()
    ax2color = 'blue'
    lns = ax2.plot(cycle, n_list, ':', color=ax2color)
    ax2.set_ylabel('$n$ [-]')
    for tl in ax2.get_yticklabels():
        tl.set_color(ax2color)

    plt.tight_layout()
    filename = sid + '-rate-dependence-A-n.' + outformat
    plt.savefig(filename)
    print(filename)
    plt.close()
    #shutil.copyfile(filename, '/home/adc/articles/own/3/graphics/' + filename)
