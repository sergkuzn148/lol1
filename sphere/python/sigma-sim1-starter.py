#!/usr/bin/env python
import sphere
import numpy
import sys

# launch with:
# $ ipython sigma-sim1-starter.py <device> <fluid> <c_phi> <k_c> <sigma_0> <mu> <velfac>

# start with
# ipython sigma-sim1-starter.py 0 1 1.0 2.0e-16 10000.0 2.080e-7 1.0

device = int(sys.argv[1])
wet = int(sys.argv[2])
c_phi = float(sys.argv[3])
k_c = float(sys.argv[4])
sigma0 = float(sys.argv[5])
mu = float(sys.argv[6])
velfac = float(sys.argv[7])

if wet == 1:
    fluid = True
else:
    fluid = False

#sim = sphere.sim('halfshear-sigma0=' + str(sigma0), fluid=False)
sim = sphere.sim('shear-sigma0=' + str(sigma0), fluid=False)
sim.readlast()
#sim.readbin('../input/shear-sigma0=10000.0-new.bin')
#sim.scaleSize(0.01) # from 1 cm to 0.01 cm = 100 micro m (fine sand)

sim.fluid = fluid
if fluid:
    #sim.id('halfshear-darcy-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
            #'-mu=' + str(mu) + '-velfac=' + str(velfac) + '-shear')
    sim.id('s1-darcy-sigma0=' + str(sigma0) + '-k_c=' + str(k_c) + \
            '-mu=' + str(mu) + '-velfac=' + str(velfac) + '-noflux-shear')
else:
    sim.id('s1-darcy-sigma0=' + str(sigma0) + '-velfac=' + str(velfac) + \
            '-noflux-shear')

#sim.checkerboardColors(nx=6,ny=3,nz=6)
sim.checkerboardColors(nx=6,ny=6,nz=6)
sim.cleanup()
sim.adjustUpperWall()
sim.zeroKinematics()

# customized shear function for linear velocity gradient
def shearVelocityGradient(sim, shear_strain_rate = 1.0, shear_stress = False):
    '''
    Setup shear experiment either by a constant shear rate or a constant
    shear stress.  The shear strain rate is the shear velocity divided by
    the initial height per second. The shear movement is along the positive
    x axis. The function zeroes the tangential wall viscosity (gamma_wt) and
    the wall friction coefficients (mu_ws, mu_wn).

    :param shear_strain_rate: The shear strain rate [-] to use if
        shear_stress isn't False.
    :type shear_strain_rate: float
    :param shear_stress: The shear stress value to use [Pa].
    :type shear_stress: float or bool
    '''

    sim.nw = 1

    # Find lowest and heighest point
    z_min = numpy.min(sim.x[:,2] - sim.radius)
    z_max = numpy.max(sim.x[:,2] + sim.radius)

    # the grid cell size is equal to the max. particle diameter
    cellsize = sim.L[0] / sim.num[0]

    # make grid one cell heigher to allow dilation
    sim.num[2] += 1
    sim.L[2] = sim.num[2] * cellsize

    # zero kinematics
    sim.zeroKinematics()

    # Adjust grid and placement of upper wall
    sim.wmode = numpy.array([1])

    # Fix horizontal velocity to 0.0 of lowermost particles
    d_max_below = numpy.max(sim.radius[numpy.nonzero(sim.x[:,2] <
        (z_max-z_min)*0.3)])*2.0
    I = numpy.nonzero(sim.x[:,2] < (z_min + d_max_below))
    sim.fixvel[I] = 1
    sim.angvel[I,0] = 0.0
    sim.angvel[I,1] = 0.0
    sim.angvel[I,2] = 0.0
    sim.vel[I,0] = 0.0 # x-dim
    sim.vel[I,1] = 0.0 # y-dim
    sim.color[I] = -1

    # Fix horizontal velocity to specific value of uppermost particles
    d_max_top = numpy.max(sim.radius[numpy.nonzero(sim.x[:,2] >
        (z_max-z_min)*0.7)])*2.0
    I = numpy.nonzero(sim.x[:,2] > (z_max - d_max_top))
    sim.fixvel[I] = 1
    sim.angvel[I,0] = 0.0
    sim.angvel[I,1] = 0.0
    sim.angvel[I,2] = 0.0
    if shear_stress == False:
        prefactor = sim.x[I,1]/sim.L[1]
        sim.vel[I,0] = prefactor*(z_max-z_min)*shear_strain_rate
    else:
        sim.vel[I,0] = 0.0
        sim.wmode[0] = 3
        sim.w_tau_x[0] = float(shear_stress)
    sim.vel[I,1] = 0.0 # y-dim
    sim.color[I] = -1

    # Set wall tangential viscosity to zero
    sim.gamma_wt[0] = 0.0

    # Set wall friction coefficients to zero
    sim.mu_ws[0] = 0.0
    sim.mu_wd[0] = 0.0
    return sim

sim = shearVelocityGradient(sim, 1.0/20.0 * velfac)
K_q_real = 36.4e9
K_w_real =  2.2e9


#K_q_sim  = 1.16e9
K_q_sim = 1.16e6
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


# frictionless side boundaries
sim.periodicBoundariesX()

# rearrange particle assemblage to accomodate frictionless side boundaries
sim.x[:,1] += numpy.abs(numpy.min(sim.x[:,1] - sim.radius[:]))
sim.L[1] = numpy.max(sim.x[:,1] + sim.radius[:])


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
#sim.deleteAllParticles()
#sim.fixvel[:] = -1.0

sim.initTemporal(total = 20.0/velfac, file_dt = 0.01/velfac, epsilon=0.07)
#sim.time_dt[0] *= 1.0e-2
#sim.initTemporal(total = 1.0e-4, file_dt = 1.0e-5, epsilon=0.07)

# Fix lowermost particles
#dz = sim.L[2]/sim.num[2]
#I = numpy.nonzero(sim.x[:,2] < 1.5*dz)
#sim.fixvel[I] = 1

sim.run(dry=True)

sim.run(device=device)
sim.writeVTKall()
