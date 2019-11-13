#!/usr/bin/env python
import sphere
import numpy
import sys

# launch with:
# $ python halfshear-starter-rate.py <DEVICE> <FLUID> <C_PHI> <C_V> <SIGMA_0> <VELOCITY FACTOR>

device = int(sys.argv[1])
wet = int(sys.argv[2])
c_phi = float(sys.argv[3])
c_v = float(sys.argv[4])
sigma0 = float(sys.argv[5])
velfac = float(sys.argv[6])

#sim = sphere.sim('diffusivity-sigma0=' + str(sigma0) + '-c_phi=' + \
#        str(c_phi) + '-c_grad_p=' + str(c_grad_p), fluid=True)
if wet == 1:
    fluid = True
else:
    fluid = False
    
#sim = sphere.sim('diffusivity-sigma0=' + str(sigma0) +'-c_phi=1.0-c_grad_p=1.0',
#        fluid=True)
sim = sphere.sim('halfshear-sigma0=' + str(sigma0), fluid=False)
print('Input: ' + sim.sid)
sim.readlast()

sim.fluid = fluid
if fluid:
    sim.id('halfshear-sigma0=' + str(sigma0) + '-c_v=' + str(c_v) + \
            '-velfac=' + str(velfac) + '-shear')
else:
    sim.id('halfshear-sigma0=' + str(sigma0) + \
            '-velfac=' + str(velfac) + '-shear')

sim.checkerboardColors(nx=6,ny=3,nz=6)
sim.cleanup()
sim.adjustUpperWall()
sim.zeroKinematics()

sim.shear(1.0/20.0 * velfac)

if fluid:
    #sim.num[2] *= 2
    #sim.L[2] *= 2.0
    sim.initFluid(mu = 1.787e-6, p = 600.0e3, hydrostatic = True)
    #sim.initFluid(mu = 17.87e-4, p = 1.0e5, hydrostatic = True)
    sim.setFluidBottomNoFlow()
    sim.setFluidTopFixedPressure()
    sim.setDEMstepsPerCFDstep(100)
    sim.setMaxIterations(2e5)
    sim.c_phi[0] = c_phi
    sim.c_v[0] = c_v

sim.initTemporal(total = 20.0/velfac, file_dt = 0.01/velfac, epsilon=0.07)
sim.w_sigma0[0] = sigma0
sim.w_m[0] = numpy.abs(sigma0*sim.L[0]*sim.L[1]/sim.g[2])
sim.mu_s[0] = 0.5
sim.mu_d[0] = 0.5
sim.setDampingNormal(0.0)
sim.setDampingTangential(0.0)

# Fix lowermost particles
#dz = sim.L[2]/sim.num[2]
#I = numpy.nonzero(sim.x[:,2] < 1.5*dz)
#sim.fixvel[I] = 1

sim.run(dry=True)
sim.run(device=device)
#sim.writeVTKall()
#sim.visualize('walls')
#sim.visualize('fluid-pressure')
