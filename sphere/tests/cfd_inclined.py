#!/usr/bin/env python
import sphere
from pytestutils import *

orig = sphere.sim('cfd_incl', fluid=True)
orig.cleanup()
#orig.defineWorldBoundaries([0.3, 0.3, 0.3], dx = 0.1)
orig.defineWorldBoundaries([0.3, 0.3, 0.3], dx = 0.06)
orig.initFluid(mu=8.9e-4) # inviscid "fluids" (mu=0) won't work!
#orig.initTemporal(total = 0.5, file_dt = 0.05, dt = 1.0e-4)
orig.initTemporal(total = 1.0e-0, file_dt = 1.0e-1, dt = 1.0e-3)
orig.bc_bot[0] = 1 # No-flow, free slip BC at bottom (Neumann)
#orig.bc_bot[0] = 2 # No-flow, no slip BC at bottom (Neumann)
#orig.bc_top[0] = 1 # No-flow, free slip BC at top (Neumann)

angle = 10.0 # slab inclination in degrees
g_magnitude = 10.0
orig.g[0] =  numpy.sin(numpy.radians(angle))*g_magnitude
orig.g[2] = -numpy.cos(numpy.radians(angle))*g_magnitude

tau_d = orig.g * orig.rho_f * orig.L[2] # analytical driving stress
v_sur = tau_d * orig.L[2] / orig.mu     # analytical surface velocity

# increase the max iterations for first step
orig.setMaxIterations(1e5)

# Homogeneous pressure, no gravity
orig.run(verbose=False)
orig.writeVTKall()

py = sphere.sim(sid=orig.sid, fluid=True)
py.readlast(verbose = False)
ones = numpy.ones((orig.num))
zeros = numpy.zeros((orig.num[0], orig.num[1], orig.num[2], 3))
#compareNumpyArraysClose(ones, py.p_f, "Conservation of pressure:",
        #tolerance = 1.0e-5)
#compareNumpyArraysClose(zeros, py.v_f, "Flow field:              ",
        #tolerance = 1.0e-5)
#ideal_grad_p_z = numpy.linspace(
        #orig.p_f[0,0,0] + orig.L[2]*orig.rho_f*numpy.abs(orig.g[2]),
        #orig.p_f[0,0,-1], orig.num[2])
#compareNumpyArraysClose(ideal_grad_p_z, py.p_f[0,0,:],
        #"Pressure gradient:\t", tolerance=1.0e2)
#orig.cleanup()
