#!/usr/bin/env python

import sphere

sid = 'fluidshear'

## Initialization from loose packing to a gravitationally collapsed state
## without fluids
sim = sphere.sim(sid + '-init', np = 24000, fluid = False)
#sim.cleanup()
sim.radius[:] = 0.05
sim.periodicBoundariesXY()
sim.initRandomGridPos(gridnum = [20, 20, 9000])
sim.initTemporal(total = 10.0, file_dt = 0.05)
sim.g[2] = -9.81
sim.run(dry=True)
sim.run()
sim.writeVTKall()

## Consolidation from a top wall without fluids
sim.readlast()
sim.sid = sid + '-cons'
sim.adjustUpperWall()
sim.consolidate(normal_stress = 10.0e3)
#sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)  # mu = water at 0 deg C
sim.initTemporal(total = 1.0, file_dt = 0.01)
sim.run(dry=True)
sim.run()
sim.writeVTKall()
sim.visualize('walls')

## Shear with fluids
sim.readlast()
sim.sid = sid + '-shear'
sim.shear()
sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)
sim.bc_bot[0] = 1  # Neumann BC
sim.setDEMstepsPerCFDstep(100)
sim.initTemporal(total = 1.0, file_dt = 0.01)
sim.run(dry=True)
sim.run()
sim.writeVTKall()
sim.visualize('shear')
