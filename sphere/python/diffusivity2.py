#!/usr/bin/env python
import sphere
import numpy

for sigma0 in [40.0e3, 60.0e3]:

    sim = sphere.sim('diffusivity-relax')
    sim.readlast()

    sim.sid = 'diffusivity-sigma0=' + str(sigma0)

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
    sim.consolidate(normal_stress = sigma0)
    sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)
    sim.setFluidBottomNoFlow()
    sim.setFluidTopFixedPressure()
    sim.setDEMstepsPerCFDstep(10)
    sim.setMaxIterations(2e5)
    sim.initTemporal(total = 5.0, file_dt = 0.01, epsilon=0.07)
    sim.run(dry=True)
    sim.run(device=1)
    #sim.writeVTKall()
    sim.visualize('walls')
    sim.visualize('fluid-pressure')
