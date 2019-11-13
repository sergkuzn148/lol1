#!/usr/bin/env python
import sphere
import numpy
import sys

# launch with:
# $ python consolidation-curves <DEVICE> <C_PHI> <C_GRAD_P>

# Unique simulation parameters
device = int(sys.argv[1])
c_phi = float(sys.argv[2])
c_grad_p = float(sys.argv[3])

sim = sphere.sim('cons-sigma0=' + str(5.0e3) + '-c_phi=' + \
            str(c_phi) + '-c_grad_p=' + str(c_grad_p), fluid=True)
sim.readlast()

sigma0 = 10.0e3
sim.sid = 'cons-sigma0=' + str(sigma0) + '-c_phi=' + str(c_phi) + \
        '-c_grad_p=' + str(c_grad_p) + '-tall'
print('\n###### ' + sim.sid + ' ######')

# Checkerboard colors
sim.checkerboardColors()
sim.cleanup()
#sim.adjustUpperWall()
sim.zeroKinematics()

#sim.consolidate(normal_stress = sigma0)
sim.w_sigma0[0] = sigma0

sim.L[2] *= 2.0
sim.num[2] *= 2
sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)
#sim.setFluidBottomNoFlow()
#sim.setFluidTopFixedPressure()
sim.setDEMstepsPerCFDstep(10)
sim.setMaxIterations(2e5)
sim.initTemporal(total = 10.0, file_dt = 0.01, epsilon=0.07)
sim.c_grad_p[0] = c_grad_p
sim.c_phi[0] = c_phi

# Fix lowermost particles
#dz = sim.L[2]/sim.num[2]
#I = numpy.nonzero(sim.x[:,2] < 1.5*dz)
#sim.fixvel[I] = 1

sim.run(dry=True)
sim.run(device=device)
#sim.writeVTKall()
#sim.visualize('walls')
#sim.visualize('fluid-pressure')
