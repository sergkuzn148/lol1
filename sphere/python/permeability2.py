#!/usr/bin/env python
import sphere
import numpy


for dp in [4.0e3, 10.0e3]:
    # Read initial configuration
    sim = sphere.sim('diffusivity-relax')
    sim.readlast()

    sim.sid = 'permeability-dp=' + str(dp)
    sim.cleanup()

    sim.g[2] = 0.0
    sim.nw = 0
    sim.initGrid()
    sim.zeroKinematics()
    sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic=True)

    # Initialize to linear hydraulic gradient
    p_bottom = 10.0
    p_top = p_bottom + dp
    dz = sim.L[2]/sim.num[2]
    for iz in range(sim.num[2]-1):
        #z = dz*iz + 0.5*dz # cell-center z-coordinate
        z = dz*iz
        sim.p_f[:,:,iz] = p_bottom + dp/sim.L[2] * z

    sim.setFluidTopFixedPressure()
    sim.setFluidBottomFixedPressure()
    sim.p_f[:,:,-1] = p_top
    sim.setDEMstepsPerCFDstep(10)
    sim.initTemporal(total = 2.0, file_dt = 0.01, epsilon=0.07)

    sim.run(dry=True)
    sim.run(device=1)
    #sim.writeVTKall()
