#!/usr/bin/env python

# This script simulates the effect of capillary cohesion on a sand pile put on a
# desk.

# start with
# $ python capillary-cohesion.py <DEVICE> <COHESION> <GRAVITY>
# where DEVICE specifies the index of the GPU (0 is the most common value).
# COHESION should have the value of 0 or 1. 0 denotes a dry simulation without
# cohesion, 1 denotes a wet simulation with capillary cohesion.
# GRAVITY toggles gravitational acceleration. Without it, the particles are
# placed in the middle of a volume. With it enabled, the particles are put on
# top of a flat wall.

import sphere
#import numpy
import sys

device = int(sys.argv[1])
cohesion = int(sys.argv[2])
gravity = int(sys.argv[3])

# Create packing
sim = sphere.sim('cap-cohesion=' + str(cohesion) + '-init-grav=' \
        + str(gravity), np=2000)
#sim.mu_s[0] = 0.0
#sim.mu_d[0] = 0.0
#sim.k_n[0] = 1.0e7
#sim.k_t[0] = 1.0e7
sim.generateRadii(psd='uni', mean=1.0e-3, variance=1.0e-4)
sim.contactModel(1)
sim.initRandomGridPos(gridnum=[24, 24, 10000], padding=1.4)
sim.defaultParams(gamma_t = 1.0e3, capillaryCohesion=1)
sim.initTemporal(5.0, file_dt=0.01, epsilon=0.07)
#I = numpy.nonzero(sim.x[:,2] < sim.L[2]*0.5)
#sim.vel[I[0], 2] =  0.01  # add a instability seeding perturbation
#I = numpy.nonzero(sim.x[:,2] > sim.L[2]*0.5)
#sim.vel[I[0], 2] = -0.01  # add a instability seeding perturbation
if gravity == 1:
    sim.g[2] = -10.0
sim.run(dry=True)
sim.run(device=device)
sim.writeVTKall()

# if gravity is enabled, read last output file, place the sediment in a large
# box, and restart time
if gravity == 1:
    sim.readlast()
    sim.sid = 'cap-cohesion=' + str(cohesion)
    sim.defaultParams(capillaryCohesion=cohesion)
    sim.adjustUpperWall()
    init_lx = sim.L[0]
    init_ly = sim.L[1]
    sim.L[0] *= 5
    sim.L[1] *= 5
    sim.num[0] *= 5
    sim.num[1] *= 5
    sim.x[:,0] += 0.5*sim.L[0] - 0.5*init_lx
    sim.x[:,1] += 0.5*sim.L[1] - 0.5*init_ly

    sim.initTemporal(2.0, file_dt=0.01, epsilon=0.07)
    sim.run(dry=True)
    sim.run(device=device)
    sim.writeVTKall()
