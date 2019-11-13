#!/usr/bin/env python
import sphere
import numpy

sid = 'diffusivity'

## Initialization from loose packing to a gravitationally collapsed state
## without fluids
#sim = sphere.sim(sid + '-init', np = 2400, fluid = False)
sim = sphere.sim(sid + '-init', np = 10000, fluid = False)
#sim.cleanup()
sim.radius[:] = 0.01
#sim.initRandomGridPos(gridnum = [12, 12, 9000])
sim.initRandomGridPos(gridnum = [24, 24, 9000])
sim.initTemporal(total = 5.0, file_dt = 0.05, epsilon=0.07)
sim.g[2] = -9.81
sim.periodicBoundariesXY()
#sim.run(dry=True)
#sim.run()
#sim.writeVTKall()

# Stack two init assemblages on top of each other
sim.readlast()
max_z = numpy.max(sim.x[:,2] + sim.radius[:])
for i in numpy.arange(sim.np):
    sim.addParticle(
            [sim.x[i,0], sim.x[i,1], sim.x[i,2] + max_z],
            radius=sim.radius[i])

cellsize_min = 2.1*numpy.max(sim.radius)
sim.L[2] = numpy.max(sim.x[:,2] + sim.radius[:])
#sim.num[0] = numpy.ceil((sim.L[0]-sim.origo[0])/cellsize_min)
#sim.num[1] = numpy.ceil((sim.L[1]-sim.origo[1])/cellsize_min)
#sim.num[2] = numpy.ceil((sim.L[2]-sim.origo[2])/cellsize_min)
sim.initGrid()

## Relaxation step
sim.sid = sid + '-relax'
sim.initTemporal(total = 2.0, file_dt = 0.05, epsilon=0.07)
sim.mu_s[0] = 0.3
sim.mu_d[0] = 0.3
sim.zeroKinematics()
sim.periodicBoundariesXY()
#sim.run(dry=True)
#sim.run()
#sim.writeVTKall()

## Consolidation from a top wall with fluids
sim.readlast()
sim.sid = sid + '-cons-wet'

# Checkerboard colors
x_min = numpy.min(sim.x[:,0])
x_max = numpy.max(sim.x[:,0])
y_min = numpy.min(sim.x[:,1])
y_max = numpy.max(sim.x[:,1])
z_min = numpy.min(sim.x[:,2])
z_max = numpy.max(sim.x[:,2])
color_nx = 6
color_ny = 6
color_nz = 6
for i in range(sim.np):
    ix = numpy.floor((sim.x[i,0] - x_min)/(x_max/color_nx))
    iy = numpy.floor((sim.x[i,1] - y_min)/(y_max/color_ny))
    iz = numpy.floor((sim.x[i,2] - z_min)/(z_max/color_nz))
    sim.color[i] = (-1)**ix + (-1)**iy + (-1)**iz

sim.cleanup()
sim.adjustUpperWall()
sim.zeroKinematics()
sim.consolidate(normal_stress = 10.0e3)
sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)  # mu = water at 0 deg C
#sim.initFluid(mu = 8.9e-4, p = 1.0e5, hydrostatic = True)  # mu = water at 20 deg C
sim.setFluidBottomNoFlow()
sim.setFluidTopFixedPressure()
#sim.setDEMstepsPerCFDstep(100)
sim.setDEMstepsPerCFDstep(10)
sim.initTemporal(total = 5.0, file_dt = 0.01, epsilon=0.07)
sim.run(dry=True)
sim.run()
sim.writeVTKall()
sim.visualize('walls')
sim.visualize('fluid-pressure')
