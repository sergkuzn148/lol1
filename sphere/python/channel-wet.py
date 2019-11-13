#!/usr/bin/env python
import sphere
import numpy

relaxation = False
consolidation = True
#water = False
water = True

id_prefix = 'chan'

#N = 2.5e3
#N = 5e3
#N = 7.5e3
N = 10e3
#N = 15e3
#N = 20e3
#N = 25e3
#N = 30e3
#N = 40e3

#dpdx = 10  # fluid-pressure gradient in Pa/m along x
#dpdx = 100  # fluid-pressure gradient in Pa/m along x
#dpdx = 200  # fluid-pressure gradient in Pa/m along x
dpdx = 1000
#dpdx = 5e3
#dpdx = 10e3
#dpdx = 20e3
#dpdx = 40e3

sim = sphere.sim(id_prefix + '-relax', nw=0)

if relaxation:
    cube = sphere.sim('cube-init')
    cube.readlast()
    cube.adjustUpperWall(z_adjust=1.0)

    # Fill out grid with cubic packages
    grid = numpy.array((
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ))

    # World dimensions and cube grid
    nx = grid.shape[1]    # horizontal cubes
    ny = 1                # horizontal (thickness) cubes
    nz = grid.shape[0]    # vertical cubes
    dx = cube.L[0]
    dy = cube.L[1]
    dz = cube.L[2]
    Lx = dx*nx
    Ly = dy*ny
    Lz = dz*nz

    # insert particles into each cube
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):

                if (grid[z, x] == 0):
                    continue  # skip to next iteration

                for i in range(cube.np):
                    pos = [cube.x[i, 0] + x*dx,
                        cube.x[i, 1] + y*dy,
                        cube.x[i, 2] + (nz-z)*dz]

                    sim.addParticle(pos, radius=cube.radius[i], color=grid[z, x])

    # move to x=0
    min_x = numpy.min(sim.x[:, 0] - sim.radius[:])
    sim.x[:, 0] = sim.x[:, 0] - min_x

    # move to y=0
    min_y = numpy.min(sim.x[:, 1] - sim.radius[:])
    sim.x[:, 1] = sim.x[:, 1] - min_y

    # move to z=0
    min_z = numpy.min(sim.x[:, 2] - sim.radius[:])
    sim.x[:, 2] = sim.x[:, 2] - min_z

    sim.defineWorldBoundaries(L=[numpy.max(sim.x[:, 0] + sim.radius[:]),
                                numpy.max(sim.x[:, 1] + sim.radius[:]),
                                numpy.max(sim.x[:, 2] + sim.radius[:])*1.2])
                                #numpy.max(sim.x[:, 2] + sim.radius[:])*10.2])
    sim.k_t[0] = 2.0/3.0*sim.k_n[0]

    # sim.cleanup()
    sim.writeVTK()
    print("Number of particles: " + str(sim.np))

    # Set grain contact properties
    #sim.setStiffnessNormal(1.16e7)
    #sim.setStiffnessTangential(1.16e7)
    sim.setYoungsModulus(70e7)
    sim.setStaticFriction(0.5)
    sim.setDynamicFriction(0.5)
    sim.setDampingNormal(0.0)
    sim.setDampingTangential(0.0)

    # Set wall properties
    sim.gamma_wn[0] = 0.0
    sim.gamma_wt[0] = 0.0


    # Relaxation

    # Add gravitational acceleration
    sim.g[0] = 0.0
    sim.g[1] = 0.0
    sim.g[2] = -9.81

    sim.normalBoundariesXY()
    # sim.consolidate(normal_stress=0.0)

    # assign automatic colors, overwriting values from grid array
    sim.checkerboardColors(nx=grid.shape[1], ny=2, nz=grid.shape[0]/4)

    sim.contactmodel[0] = 2
    sim.mu_s[0] = 0.5
    sim.mu_d[0] = 0.5

    # Set duration of simulation, automatically determine timestep, etc.
    sim.initTemporal(total=8.0, file_dt=0.01, epsilon=0.07)
    #sim.time_dt[0] = 1.0e-20
    #sim.time_file_dt = sim.time_dt
    #sim.time_total = sim.time_file_dt*5.
    sim.zeroKinematics()

    sim.run(dry=True)
    sim.run()
    sim.writeVTKall()


# Consolidation under constant normal stress
if consolidation:
    sim.readlast()
    sim.id(id_prefix + '-' + str(int(N/1000.)) + 'kPa')
    #sim.cleanup()
    sim.initTemporal(current=0.0, total=10.0, file_dt=0.01, epsilon=0.07)

    # fix horizontal movement of lowest plane of particles
    I = numpy.nonzero(sim.x[:, 2] < 1.5*numpy.mean(sim.radius))
    sim.fixvel[I] = 1
    sim.color[I] = 0

    # fix horizontal movement of uppermost plane of particles
    z_min = numpy.min(sim.x[:,2] - sim.radius)
    z_max = numpy.max(sim.x[:,2] + sim.radius)
    d_max_top = numpy.max(sim.radius[numpy.nonzero(sim.x[:,2] >
                                                    (z_max-z_min)*0.7)])*2.0
    I = numpy.nonzero(sim.x[:,2] > (z_max - 4.0*d_max_top))
    sim.fixvel[I] = 1
    sim.color[I] = 0

    sim.zeroKinematics()

    # Wall parameters
    sim.mu_ws[0] = 0.5
    sim.mu_wd[0] = 0.5
    #sim.gamma_wn[0] = 1.0e2
    #sim.gamma_wt[0] = 1.0e2
    sim.gamma_wn[0] = 0.0
    sim.gamma_wt[0] = 0.0

    # Particle parameters
    sim.setYoungsModulus(70e7)
    sim.mu_s[0] = 0.5
    sim.mu_d[0] = 0.5
    sim.gamma_n[0] = 0.0
    sim.gamma_t[0] = 0.0

    # apply effective normal stress from upper wall
    sim.consolidate(normal_stress=N)


    if water:

        # read last output from previous dry experiment
        sim.readlast()
        sim.zeroKinematics()
        sim.initTemporal(current=0.0, total=10.0, file_dt=0.01, epsilon=0.07)

        # initialize fluid
        sim.num = sim.num/2
        sim.initFluid(mu=1.797e-6, p=0.0, rho=1000.0, cfd_solver=1) # water at 0 C / 1000
        sim.setFluidCompressibility(1.426e-8) # water at 0 C
        sim.setMaxIterations(2e5)
        #sim.setPermeabilityGrainSize()
        sim.setPermeabilityPrefactor(3.5e-13)
        #sim.setPermeabilityPrefactor(3.5e-11)
        #sim.setPermeabilityPrefactor(3.5e-15)

        # initialize linear fluid pressure gradient along x
        dx = sim.L[0]/sim.num[0]
        for ix in numpy.arange(sim.num[0]):
            x = dx*ix + 0.5*dx
            sim.p_f[ix,:,:] = (dpdx*sim.L[0]) - x*dpdx

        ## Fluid phase boundary conditions

        # set initial pressure to zero (hydrostatic pres. distr.) everywhere
        #sim.p_f[:,:,:] = 0.

        # x
        sim.bc_xn[0] = 0  # -x boundary: fixed pressure

        # set higher fluid pressure at x-boundary away from the channel
        #sim.p_f[0,:,:] = dpdx/sim.L[0]

        sim.id(sim.id() + '-dpdx=' + str(dpdx))

        sim.bc_xp[0] = 1  # +x boundary: no flow

        # pressure held constant in cells at (x=nx-1, z=nz-1)
        #sim.p_f_constant[30:,:,-2:] = 1
        #sim.p_f_constant[40:,:,-3] = 1

        # y: Can't prescribe Y pressures without affecting pressure field from
        # -x to +x
        #sim.setFluidYFixedPressure()
        sim.setFluidYNoFlow()

        # z
        #sim.setFluidTopNoFlow()  # ignore small contribution by IBI melting
        sim.setFluidTopFixedPressure()
        #sim.setFluidTopFixedFlux(specific_flux=??)
        sim.setFluidBottomNoFlow()

        # Adjust fluid grid size dynamically
        sim.adaptiveGrid()
        #sim.id(sim.id() + '-dpdx=' + str(dpdx) + '-newBC')


    #sim.time_file_dt = sim.time_dt
    #sim.time_total = sim.time_file_dt * 10

    sim.run(dry=True)
    sim.run()
    sim.writeVTKall()
