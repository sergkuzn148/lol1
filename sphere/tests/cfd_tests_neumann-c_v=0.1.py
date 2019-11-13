#!/usr/bin/env python
from pytestutils import *

import sphere
import sys
import numpy
import matplotlib.pyplot as plt

print('### CFD tests - Dirichlet/Neumann BCs ###')

print('''# Neumann bottom, Dirichlet top BC.
# No gravity, no pressure gradients => no flow''')
orig = sphere.sim("neumann", fluid = True)
cleanup(orig)
orig.defaultParams(mu_s = 0.4, mu_d = 0.4)
orig.defineWorldBoundaries([0.4, 0.4, 1], dx = 0.1)
orig.initFluid(mu = 8.9e-4)
#orig.initFluid(mu = 0.0)
orig.initTemporal(total = 0.05, file_dt = 0.005, dt = 1.0e-4)
orig.c_v[0] = 0.1
#orig.c_phi[0] = 0.1
py = sphere.sim(sid = orig.sid, fluid = True)
orig.bc_bot[0] = 1      # No-flow BC at bottom (Neumann)
#orig.run(dry=True)
orig.run(verbose=False)
#orig.run(device=2)
#orig.writeVTKall()
py.readlast(verbose = False)
zeros = numpy.zeros((orig.num))
py.readlast(verbose = False)
compareNumpyArraysClose(zeros, py.p_f, "Conservation of pressure:",
        tolerance = 1.0e-5)

# Fluid flow along z should be very small
if ((numpy.abs(py.v_f[:,:,:,:]) < 1.0e-6).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())
    print(numpy.min(py.v_f))
    print(numpy.mean(py.v_f))
    print(numpy.max(py.v_f))
    raise Exception("Failed")

print('''# Neumann bottom, Dirichlet top BC.
# Gravity, pressure gradients => transient flow''')
orig = sphere.sim("neumann", fluid = True)
orig.defaultParams(mu_s = 0.4, mu_d = 0.4)
orig.defineWorldBoundaries([0.4, 0.4, 1], dx = 0.1)
orig.initFluid(mu = 8.9e-4)
#orig.initTemporal(total = 0.05, file_dt = 0.005, dt = 1.0e-4)
#orig.initTemporal(total = 0.05, file_dt = 0.005, dt = 0.001)
orig.initTemporal(total = 0.5, file_dt = 0.05, dt = 1.0e-4)
py = sphere.sim(sid = orig.sid, fluid = True)
orig.g[2] = -10.0
orig.c_v[0] = 0.1
orig.bc_bot[0] = 1      # No-flow BC at bottom (Neumann)
#orig.run(dry=True)
orig.run(verbose=False)
#orig.run(device=2)
#orig.writeVTKall()
py.readlast(verbose = False)
#ideal_grad_p_z = numpy.linspace(
#        orig.p_f[0,0,0] + orig.L[2]*orig.rho_f*numpy.abs(orig.g[2]),
#        orig.p_f[0,0,-1], orig.num[2])
ideal_grad_p_z = numpy.linspace(
        orig.p_f[0,0,0] + (orig.L[2]-orig.L[2]/orig.num[2])*orig.rho_f*numpy.abs(orig.g[2]),
        orig.p_f[0,0,-1], orig.num[2])
compareNumpyArraysClose(ideal_grad_p_z, py.p_f[0,0,:],
        "Pressure gradient:\t", tolerance=1.0)

# Fluid flow along z should be very small
if ((numpy.abs(py.v_f[:,:,:,2]) < 1.0e-4).all()):
    print("Flow field:\t\t" + passed())
else:
    print("Flow field:\t\t" + failed())
    raise Exception("Failed")

#orig.cleanup()
