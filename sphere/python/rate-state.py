#!/usr/bin/env python
import sphere
import numpy
#import sys

# launch with:
# $ ipython sigma-sim1-starter.py <device> <fluid> <c_phi> <k_c> <sigma_0> <mu> <velfac>

# start with
# ipython sigma-sim1-starter.py 0 1 1.0 2.0e-16 10000.0 2.080e-7 1.0

sid_prefix = 'ratestate3'

# device = int(sys.argv[1])
# wet = int(sys.argv[2])
# c_phi = float(sys.argv[3])
# k_c = float(sys.argv[4])
# sigma0 = float(sys.argv[5])
# mu = float(sys.argv[6])
# velfac = float(sys.argv[7])
device = 0
wet = 0
c_phi = 1.0
k_c = 3.5e-13
#sigma0 = 80000.0
sigma0 = 40000.0
mu = 2.080e-7
velfac = 1.0

start_from_beginning = False

if wet == 1:
    fluid = True
else:
    fluid = False

# load consolidated granular assemblage
#sim = sphere.sim('halfshear-sigma0=' + str(sigma0), fluid=False)
sim = sphere.sim(fluid=False)
if start_from_beginning:
    sim = sphere.sim('shear-sigma0=' + str(sigma0), fluid=False)
    sim.readlast()
else:
    if fluid:
        sim.id('ratestate2-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
                '-mu=' + str(mu) + '-shear')
    else:
        sim.id('ratestate2-sigma0=' + str(sigma0) + '-shear')

    sim.readTime(4.9)

    if fluid:
        sim.id(sid_prefix + '-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
                '-mu=' + str(mu) + '-shear')
    else:
        sim.id(sid_prefix + '-sigma0=' + str(sigma0) + '-shear')

#sim.readbin('../input/shear-sigma0=10000.0-new.bin')
#sim.scaleSize(0.01) # from 1 cm to 0.01 cm = 100 micro m (fine sand)


sim.fluid = fluid
if fluid:
    sim.id(sid_prefix + '-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
            '-mu=' + str(mu) + '-shear')
else:
    sim.id(sid_prefix + '-sigma0=' + str(sigma0) + '-shear')

if start_from_beginning:
    sim.checkerboardColors(nx=6,ny=3,nz=6)
    #sim.checkerboardColors(nx=6,ny=6,nz=6)
    sim.cleanup()
    sim.adjustUpperWall()
    sim.zeroKinematics()

    sim.shear(1.0/20.0 * velfac)
    K_q_real = 36.4e9
    K_w_real =  2.2e9


    K_q_sim  = 1.16e9
    #K_q_sim = 1.16e6
    sim.setStiffnessNormal(K_q_sim)
    sim.setStiffnessTangential(K_q_sim)
    K_w_sim  = K_w_real/K_q_real * K_q_sim


    if fluid:
        #sim.num[2] *= 2
        sim.num[:] /= 2
        #sim.L[2] *= 2.0
        #sim.initFluid(mu = 1.787e-6, p = 600.0e3, cfd_solver = 1)
        sim.initFluid(mu = mu, p = 0.0, cfd_solver = 1)
        sim.setFluidTopFixedPressure()
        #sim.setFluidTopFixedFlow()
        sim.setFluidBottomNoFlow()
        #sim.setFluidBottomFixedPressure()
        #sim.setDEMstepsPerCFDstep(10)
        sim.setMaxIterations(2e5)
        sim.setPermeabilityPrefactor(k_c)
        sim.setFluidCompressibility(1.0/K_w_sim)

    sim.w_sigma0[0] = sigma0
    sim.w_m[0] = numpy.abs(sigma0*sim.L[0]*sim.L[1]/sim.g[2])

    #sim.setStiffnessNormal(36.4e9 * 0.1 / 2.0)
    #sim.setStiffnessTangential(36.4e9/3.0 * 0.1 / 2.0)
    sim.setStiffnessNormal(K_q_sim)
    sim.setStiffnessTangential(K_q_sim)
    sim.mu_s[0] = 0.5
    sim.mu_d[0] = 0.5
    sim.setDampingNormal(0.0)
    sim.setDampingTangential(0.0)

    sim.initTemporal(total = 20.0/velfac, file_dt = 0.01/velfac, epsilon=0.07)


I = numpy.nonzero(sim.fixvel > 0)
sim.fixvel[I] = 8.0 # step-wise velocity change when fixvel in ]5.0; 10.0[

sim.run(dry=True)

sim.run(device=device)
sim.writeVTKall()
sim.visualize('shear')
