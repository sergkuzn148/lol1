#!/usr/bin/env python
from pytestutils import *

import sphere
import sys
import numpy
import matplotlib.pyplot as plt

print("### Stokes test - single sphere sedimentation ###")
## Stokes drag
orig = sphere.sim(sid = "stokes_law_set2", fluid = True)
cleanup(orig)
orig.defaultParams()
orig.addParticle([0.5,0.5,1.46], 0.05)
#orig.defineWorldBoundaries([1.0,1.0,5.0], dx=0.1)
#orig.defineWorldBoundaries([1.0,1.0,5.0])
orig.defineWorldBoundaries([1.0,1.0,2.0])
orig.initFluid(mu = 8.9e-4)
#orig.initTemporal(total = 1.0, file_dt = 0.01)
#orig.initTemporal(total = 1.0e-1, file_dt = 5.0e-3)
#orig.initTemporal(total = 5.0, file_dt = 1.0e-2)
orig.initTemporal(total = 1.0, file_dt = 1.0e-2)
#orig.time_file_dt = orig.time_dt
#orig.time_total = orig.time_dt*200
#orig.time_file_dt = orig.time_dt*10
#orig.time_total = orig.time_dt*1000
#orig.g[2] = -10.0
#orig.bc_bot[0] = 1      # No-flow BC at bottom
#orig.setGamma(0.0)
orig.vel[0,2] = -0.1
#orig.vel[0,2] = -0.001
#orig.setBeta(0.5)
orig.setTolerance(1.0e-4)
orig.setDEMstepsPerCFDstep(100)
orig.run(dry=True)
orig.run(verbose=True)
py = sphere.sim(sid = orig.sid, fluid = True)

ones = numpy.ones((orig.num))
py.readlast(verbose = False)
py.plotConvergence()
py.writeVTKall()
#compareNumpyArrays(ones, py.p_f, "Conservation of pressure:")

it = numpy.loadtxt("../output/" + orig.sid + "-conv.log")
#test((it[:,1] < 2000).all(), "Convergence rate:\t\t")

t = numpy.empty(py.status())
acc = numpy.empty(py.status())
vel = numpy.empty(py.status())
pos = numpy.empty(py.status())
for i in range(py.status()):
    py.readstep(i+1, verbose=False)
    t[i] = py.time_current[0]
    acc[i] = py.force[0,2]/(V_sphere(py.radius[0])*py.rho[0]) + py.g[2]
    vel[i] = py.vel[0,2]
    pos[i] = py.x[0,2]

fig = plt.figure()
#plt.title('Convergence evolution in CFD solver in "' + self.sid + '"')
plt.xlabel('Time [s]')
plt.ylabel('$z$ value')
plt.plot(t, acc, label='Acceleration')
plt.plot(t, vel, label='Velocity')
plt.plot(t, pos, label='Position')
plt.grid()
plt.legend()
format = 'png'
plt.savefig('./' + py.sid + '-stokes.' + format)
plt.clf()
plt.close(fig)
#cleanup(orig)

# Print 
print('Final z-acceleration of particle: ' + str(acc[-1]) + ' m/s^2')
print('Final z-velocity of particle:     ' + str(py.vel[0,2]) + ' m/s')
print('Lowest fluid pressure:  ' + str(numpy.min(py.p_f)) + ' Pa')
print('Highest fluid pressure: ' + str(numpy.max(py.p_f)) + ' Pa')
