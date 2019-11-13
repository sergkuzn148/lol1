#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
#matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
matplotlib.rcParams.update({'font.size': 7, 'font.family': 'sans-serif'})
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
import shutil

import os
import sys
import numpy
import sphere
import matplotlib.pyplot as plt

sids =\
        ['halfshear-darcy-sigma0=80000.0-k_c=3.5e-13-mu=1.04e-07-ss=10000.0-A=70000.0-f=0.2']
        #['halfshear-darcy-sigma0=80000.0-k_c=3.5e-13-mu=1.797e-06-ss=10000.0-A=70000.0-f=0.2']
outformat = 'pdf'
fluid = True
#threshold = 100.0 # [N]



for sid in sids:

    sim = sphere.sim(sid, fluid=fluid)

    #nsteps = 2
    #nsteps = 10
    nsteps = sim.status()

    t = numpy.empty(nsteps)
    n = numpy.empty(nsteps)
    coordinationnumber = numpy.empty(nsteps)
    nkept = numpy.empty(nsteps)

    for i in numpy.arange(nsteps):
        sim.readstep(i+1)
        t[i] = sim.currentTime()
        #n[i] = countLoadedContacts(sim, threshold)

        if i > 0:
            loaded_contacts_prev = numpy.copy(loaded_contacts)
            pairs_prev = numpy.copy(sim.pairs)

        loaded_contacts = sim.findLoadedContacts(
                threshold = sim.currentNormalStress()*2.)
                #sim.currentNormalStress()/1000.)
        n[i] = numpy.size(loaded_contacts)

        sim.findCoordinationNumber()
        coordinationnumber[i] = sim.findMeanCoordinationNumber()

        nfound = 0
        if i > 0:
            for a in loaded_contacts[0]:
                for b in loaded_contacts_prev[0]:
                    if (sim.pairs[:,a] == pairs_prev[:,b]).all():
                        nfound += 1;

        nkept[i] = nfound

        print coordinationnumber[i]
        print nfound

    #fig = plt.figure(figsize=[8,8])
    fig = plt.figure(figsize=[3.5,7])

    ax1 = plt.subplot(3,1,1)
    #ax1.plot(t, n)
    ax1.semilogy(t, n, 'k')
    ax1.set_xlabel('Time $t$ [s]')
    #ax1.set_ylabel(\
            #'Heavily loaded contacts $||\\boldsymbol{{f}}_n|| \\geq {}$ N'.format(
                #threshold))
    ax1.set_ylabel(\
            'Heavily loaded contacts $\sigma_c \\geq \\sigma_0\\times 4$ Pa')
            #'Heavily loaded contacts $||\\boldsymbol{{\sigma}}_c|| \\geq \\sigma_0\\times 4$ Pa')

    ax2 = plt.subplot(3,1,2)
    ax2.semilogy(t, n - nkept, 'k')
    ax2.set_xlabel('Time $t$ [s]')
    ax2.set_ylabel(\
            'New heavily loaded contacts')
            #'Heavily loaded contacts $||\\boldsymbol{{\sigma}}_c|| \\geq \\sigma_0\\times 4$ Pa')

    ax3 = plt.subplot(3,1,3)
    #ax3.semilogy(t, coordinationnumber)
    ax3.plot(t, coordinationnumber, 'k')
    ax3.set_xlabel('Time $t$ [s]')
    ax3.set_ylabel('Coordination number $z$')



    plt.savefig(sid + '-nloaded.' + outformat)
