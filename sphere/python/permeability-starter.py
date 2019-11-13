#!/usr/bin/env python
import sphere
import numpy
import sys

# launch with:
# $ python permeability-starter <DEVICE> <C_PHI> <C_GRAD_P> <DP_1, DP_2, ...>

for dp_str in sys.argv[4:]:

    dp = float(dp_str)
    device = int(sys.argv[1])
    c_phi = float(sys.argv[2])
    c_grad_p = float(sys.argv[3])

    # Read initial configuration
    sim = sphere.sim('diffusivity-relax')
    sim.readlast()

    sim.sid = 'permeability-dp=' + str(dp) + '-c_phi=' + str(c_phi) + \
            '-c_grad_p=' + str(c_grad_p)
    print(sim.sid)
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

    # Fix lowermost particles
    I = numpy.nonzero(sim.x[:,2] < 1.5*dz)
    sim.fixvel[I] = 1
    
    sim.setFluidTopFixedPressure()
    sim.setFluidBottomFixedPressure()
    sim.p_f[:,:,-1] = p_top
    sim.setDEMstepsPerCFDstep(10)
    sim.initTemporal(total = 4.0, file_dt = 0.01, epsilon=0.07)
    sim.c_phi[0] = c_phi
    sim.c_grad_p[0] = c_grad_p

    sim.run(dry=True)
    sim.run(device=device)
    #sim.writeVTKall()
