#!/usr/bin/env python

import sphere
import numpy
import matplotlib
matplotlib.rcParams.update({'font.size': 12})
import matplotlib.pyplot as plt
import subprocess

dpdz_values = [0.0, -50.0, -100.0]
slope_angle_values = [5.0, 10.0, 15.0, 20.0]

scatter_color = '#666666'
scatter_alpha = 0.6

for dpdz in dpdz_values:
    outfiles = ''
    for slope_angle in slope_angle_values:

        sim = sphere.sim("supraglacial-slope{}-dpdz{}".format(slope_angle, dpdz),
                         fluid=True)
        print('### ' + sim.id())
        print('Last output file: ' + str(sim.status()))
        sim.readTime(4.99)

        fig = plt.figure()

        title = 'slope = ' + str(slope_angle) + '$^\circ$, ' + \
                '$dp/dz$ = -' + str(dpdz) + ' Pa/m'

        z = numpy.zeros(20)
        v_x_space_avg = numpy.empty_like(z)
        xsum_space_avg = numpy.empty_like(z)
        dz = numpy.max(sim.x[:,2])/len(z)
        for i in range(len(z)):
            z[i] = i*dz + 0.5*dz
            I = numpy.nonzero((sim.x[:,2] >= i*dz) & (sim.x[:,2] < (i+1)*dz))
            v_x_space_avg[i] = numpy.mean(sim.vel[I,0])
            xsum_space_avg[i] = numpy.mean(sim.xyzsum[I,0])

        plt.plot(sim.vel[:,0], sim.x[:,2], '.',
                 color=scatter_color, alpha=scatter_alpha)
        plt.plot(v_x_space_avg, z, '+-k')
        plt.title(title)
        plt.xlabel('Horizontal particle velocity $v_x$ [m/s]')
        plt.ylabel('Vertical position $z$ [m]')
        plt.savefig(sim.id() + '-vel.pdf')
        plt.savefig(sim.id() + '-vel.png')
        plt.clf()

        plt.plot(sim.xyzsum[:,0]/sim.time_current, sim.x[:,2], '.',
                 color=scatter_color, alpha=scatter_alpha)
        plt.plot(xsum_space_avg/sim.time_current, z, '+-k')
        plt.title(title)
        plt.xlabel('Average horizontal particle velocity $\\bar{v}_x$ [m/s]')
        plt.ylabel('Vertical position $z$ [m]')
        plt.savefig(sim.id() + '-avg_vel.pdf')
        plt.savefig(sim.id() + '-avg_vel.png')
        plt.clf()

        plt.close()
        outfiles += sim.id() + '-avg_vel.png '

    subprocess.call('montage ' + outfiles +
                    '-geometry +0+0 ' +
                    'supraglacial-avg_vel-dpdz_' + str(dpdz) + '.png ',
                     shell=True)
