#!/usr/bin/env python
import sphere
from pytestutils import *

sim = sphere.sim('fluid_particle_interaction', fluid=True)
sim.cleanup()

sim.defineWorldBoundaries([1.0, 1.0, 1.0], dx = 0.1)
#sim.defineWorldBoundaries([1.0, 1.0, 1.0], dx = 0.25)
sim.initFluid(cfd_solver = 1)


# No gravity, pressure gradient enforced by Dirichlet boundaries.
# The particle should be sucked towards the low pressure
print('# Test 1: Test pressure gradient force')
sim.p_f[:,:,-1] = 1.0
#sim.addParticle([0.5, 0.5, 0.5], 0.01)
sim.addParticle([0.55, 0.55, 0.55], 0.05)
#sim.vel[0,2] = 1.0e-2
sim.initTemporal(total=0.001, file_dt=0.0001)
#sim.time_file_dt[0] = sim.time_dt[0]*1
#sim.time_total[0] = sim.time_dt[0]*100

#sim.g[2] = -10.
sim.run(verbose=False)
#sim.run()
#sim.run(dry=True)
#sim.run(cudamemcheck=True)
#sim.writeVTKall()

sim.readlast(verbose=False)
test(sim.vel[0,2] < 0.0, 'Particle velocity:')

sim.cleanup()


# Gravity, pressure gradient enforced by Dirichlet boundaries.
# The particle should be sucked towards the low pressure
print('# Test 2: Test pressure gradient force from buoyancy')

sim = sphere.sim('fluid_particle_interaction', fluid=True)
sim.defineWorldBoundaries([1.0, 1.0, 1.0], dx = 0.1)
sim.initFluid(cfd_solver = 1)
sim.p_f[:,:,-1] = 0.0
sim.addParticle([0.5, 0.5, 0.5], 0.01)
sim.initTemporal(total=0.001, file_dt=0.0001)
#sim.time_file_dt[0] = sim.time_dt[0]
#sim.time_total[0] = sim.time_dt[0]

sim.g[2] = -10.
sim.run(verbose=False)
#sim.run()
#sim.run(dry=True)
#sim.run(cudamemcheck=True)
#sim.writeVTKall()

sim.readlast(verbose=False)
test(sim.vel[0,2] < 0.0, 'Particle velocity:')
