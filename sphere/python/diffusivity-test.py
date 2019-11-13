#!/usr/bin/env python
import sphere
import numpy
import sys

# launch with:
# $ python diffusivity-test <DEVICE> <C_PHI> <C_GRAD_P>

# Unique simulation parameters
device = int(sys.argv[1])
c_phi = float(sys.argv[2])
c_grad_p = float(sys.argv[3])

# Unconsolidated input
sim = sphere.sim('diffusivity-relax')
sim.readlast()

# Load sequence as per Bowles 1992, p. 135, units: Pa. dp/p = 1
sigma0_list = numpy.array([5.0e3, 10.0e3, 20.0e3, 40.0e3, 80.0e3, 160.0e3])

i = 0
for sigma0 in sigma0_list:

    if (i == 0):
        i += 1
        continue

    # Read previous output if not first load test
    if (i > 0):
        sim.sid = 'cons-sigma0=' + str(sigma0_list[i-1]) + '-c_phi=' + \
                str(c_phi) + '-c_grad_p=' + str(c_grad_p)
        sim.readlast()

    sim.sid = 'cons-sigma0=' + str(sigma0) + '-c_phi=' + str(c_phi) + \
            '-c_grad_p=' + str(c_grad_p)
    print('\n###### ' + sim.sid + ' ######')

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
    for n in range(sim.np):
        ix = numpy.floor((sim.x[n,0] - x_min)/(x_max/color_nx))
        iy = numpy.floor((sim.x[n,1] - y_min)/(y_max/color_ny))
        iz = numpy.floor((sim.x[n,2] - z_min)/(z_max/color_nz))
        sim.color[n] = (-1)**ix + (-1)**iy + (-1)**iz

    sim.cleanup()
    sim.adjustUpperWall()
    sim.zeroKinematics()

    sim.consolidate(normal_stress = sigma0)

    sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)
    sim.setFluidBottomNoFlow()
    sim.setFluidTopFixedPressure()
    sim.setDEMstepsPerCFDstep(10)
    sim.setMaxIterations(2e5)
    sim.initTemporal(total = 20.0, file_dt = 0.01, epsilon=0.07)
    sim.c_grad_p[0] = c_grad_p
    sim.c_phi[0] = c_phi

    # Fix lowermost particles
    dz = sim.L[2]/sim.num[2]
    I = numpy.nonzero(sim.x[:,2] < 1.5*dz)
    sim.fixvel[I] = 1
    
    sim.run(dry=True)
    sim.run(device=device)
    #sim.writeVTKall()
    sim.visualize('walls')
    sim.visualize('fluid-pressure')

    i += 1
