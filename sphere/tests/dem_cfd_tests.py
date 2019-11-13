#!/usr/bin/env python
from pytestutils import *

import sphere
import sys
import numpy
import matplotlib.pyplot as plt

print('### DEM/CFD tests - Dirichlet/Neumann BCs and a single particle ###')

print('# No gravity')
orig = sphere.sim('dem_cfd', fluid = True)
cleanup(orig)
orig.defaultParams(mu_s = 0.4, mu_d = 0.4)
orig.addParticle([0.2, 0.2, 0.6], 0.05)
orig.defineWorldBoundaries([0.4, 0.4, 1.0], dx = 0.1)
orig.initFluid(mu = 8.9e-4)
orig.initTemporal(total = 0.5, file_dt = 0.05, dt = 1.0e-4)
py = sphere.sim(sid = orig.sid, fluid = True)
orig.bc_bot[0] = 1      # No-flow BC at bottom (Neumann)
#orig.run(dry=True)
orig.run(verbose=False)
#orig.writeVTKall()
py.readlast(verbose = False)
ones = numpy.ones((orig.num))
zeros = numpy.zeros((orig.num[0], orig.num[1], orig.num[2], 3))
compareNumpyArraysClose(ones, py.p_f, 'Conservation of pressure:',
        tolerance = 1.0e-1)
compareNumpyArraysClose([0,0,0], py.vel[0], 'Particle velocity:\t',
        tolerance = 1.0e-5)
compareNumpyArraysClose(zeros, py.v_f, 'Fluid velocities:\t',
        tolerance = 1.0e-4)

print('# Gravity')
orig = sphere.sim('dem_cfd', fluid = True)
cleanup(orig)
orig.defaultParams(mu_s = 0.4, mu_d = 0.4)
orig.addParticle([0.2, 0.2, 0.6], 0.02)
orig.defineWorldBoundaries([0.4, 0.4, 1], dx = 0.04)
orig.initFluid(mu = 8.9e-4)
#orig.initTemporal(total = 0.5, file_dt = 0.01)
orig.initTemporal(total = 1.0e-4, file_dt = 1.0e-5)
py = sphere.sim(sid = orig.sid, fluid = True)
orig.g[2] = -10.0
orig.bc_bot[0] = 1      # No-flow BC at bottom (Neumann)
orig.setTolerance(1.0e-3)
orig.setMaxIterations(2e4)
orig.run(dry=True)
orig.run(verbose=True)
orig.writeVTKall()
py.readlast(verbose = False)
ones = numpy.ones((orig.num))
zeros = numpy.zeros((orig.num[0], orig.num[1], orig.num[2], 3))
