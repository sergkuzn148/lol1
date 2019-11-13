#!/usr/bin/env python
import sphere
import numpy
import sys

# launch with:
# $ python diffusivity-starter <DEVICE> <C_PHI> <C_GRAD_P> <DP_1, DP_2, ...>

for sigma0_str in sys.argv[4:]:

    sigma0 = float(sigma0_str)
    device = int(sys.argv[1])
    c_phi = float(sys.argv[2])
    c_grad_p = float(sys.argv[3])

    sim = sphere.sim('diffusivity-relax')
    sim.readlast()

    sim.sid = 'diffusivity-sigma0=' + str(sigma0) + '-c_phi=' + str(c_phi) + \
            '-c_grad_p=' + str(c_grad_p) + '-tall'
    print(sim.sid)

    sim.checkerboardColors()

    sim.cleanup()
    sim.adjustUpperWall()
    sim.zeroKinematics()

    sim.consolidate(normal_stress = sigma0)

    # Increase height of grid
    sim.L[2] *= 2.0
    sim.num[2] *= 2

    sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)
    sim.setFluidBottomNoFlow()
    sim.setFluidTopFixedPressure()
    sim.setDEMstepsPerCFDstep(10)
    sim.setMaxIterations(2e5)
    sim.initTemporal(total = 10.0, file_dt = 0.01, epsilon=0.07)

    # Fix lowermost particles
    dz = sim.L[2]/sim.num[2]
    I = numpy.nonzero(sim.x[:,2] < 1.5*dz)
    sim.fixvel[I] = 1
    
    sim.run(dry=True)
    sim.run(device=device)
    #sim.writeVTKall()
    sim.visualize('walls')
    sim.visualize('fluid-pressure')
