#!/usr/bin/env python
import sphere
from pytestutils import *

orig = sphere.sim('cfd_simple', fluid=True)
orig.cleanup()
#orig.defineWorldBoundaries([0.3, 0.3, 0.3], dx = 0.1)
orig.defineWorldBoundaries([0.3, 0.3, 0.3], dx = 0.06)
#orig.initFluid(mu=0.0)
orig.initFluid(mu=8.9e-4)
#orig.initTemporal(total = 0.5, file_dt = 0.05, dt = 1.0e-4)
orig.initTemporal(total = 1.0e-0, file_dt = 1.0e-1, dt = 1.0e-3)
orig.bc_bot[0] = 1      # No-flow BC at bottom (Neumann)
orig.g[2] = -10.0

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
ideal_grad_p_z = numpy.linspace(
        orig.p_f[0,0,0] + orig.L[2]*orig.rho_f*numpy.abs(orig.g[2]),
        orig.p_f[0,0,-1], orig.num[2])
compareNumpyArraysClose(ideal_grad_p_z, py.p_f[0,0,:],
        "Pressure gradient:\t", tolerance=1.0e2)
#orig.cleanup()
