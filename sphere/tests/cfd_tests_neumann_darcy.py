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
orig.initFluid(cfd_solver = 1)
orig.initTemporal(total = 0.05, file_dt = 0.005, dt = 1.0e-4)
py = sphere.sim(sid = orig.sid, fluid = True)
orig.bc_bot[0] = 1      # No-flow BC at bottom (Neumann)
#orig.run(dry=True)
orig.run(verbose=False)
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

orig.cleanup()
