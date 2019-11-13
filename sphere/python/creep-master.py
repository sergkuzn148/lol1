#!/usr/bin/env python

# Import sphere functionality
import sphere
import numpy

### EXPERIMENT SETUP ###
initialization = False
consolidation  = False
shearing       = False
creeping       = True
rendering      = False
plots          = True

# Common simulation id
sim_id = "creep2"

# Fluid-pressure gradient [Pa/m]
dpdx = -100.0

# Deviatoric stress [Pa]
N = 100e3

# Grain density
rho_g = 1000.0

# Fluid density
rho_f = 1000.0

# Gravitational acceleration
g = 10.0

# Number of particles
np = 1e4


### INITIALIZATION ###

# New class
#init = sphere.sim(np = np, nd = 3, nw = 0, sid = sim_id + "-init")
init = sphere.sim(np = np, nd = 3, nw = 0, sid = 'creep1' + "-init")

# Uniform radii from 0.8 cm to 1.2 cm
init.generateRadii(psd = 'uni', mean = 0.005, variance = 0.001)

# Use default params
init.defaultParams(gamma_n = 100.0, mu_s = 0.6, mu_d = 0.6)
init.setYoungsModulus(1e8)

# Add gravity
init.g[2] = -g

# Periodic x and y boundaries
init.periodicBoundariesXY()

# Initialize positions in random grid (also sets world size)
hcells = np**(1.0/3.0)
init.initRandomGridPos(gridnum = [hcells, hcells, 1e9])

# Set duration of simulation
init.initTemporal(total = 5.0)

if (initialization == True):

    # Run sphere
    init.run(dry = True)
    init.run()

    if (plots == True):
        # Make a graph of energies
        init.visualize('energy')

    init.writeVTKall()

    if (rendering == True):
        # Render images with raytracer
        init.render(method = "angvel", max_val = 0.3, verbose = False)


### CONSOLIDATION ###

# New class
cons = sphere.sim(np = init.np, nw = 1, sid = sim_id +
                    "-cons-N{}".format(N))

# Read last output file of initialization step
lastf = init.status()
#cons.readbin("../output/" + sim_id + "-init.output{:0=5}.bin".format(lastf), verbose=False)
cons.readbin("../output/" + 'creep1' + "-init.output{:0=5}.bin".format(lastf),
             verbose=False)

# Periodic x and y boundaries
cons.periodicBoundariesXY()

# Setup consolidation experiment
cons.consolidate(normal_stress = N)

# Disable wall viscosities
cons.gamma_wn[0] = 0.0
cons.gamma_wt[0] = 0.0

cons.rho[0] = rho_g
cons.g[2] = -g

# Set duration of simulation
cons.initTemporal(total = 1.5)

if (consolidation == True):

    # Run sphere
    cons.run(dry = True) # show values, don't run
    cons.run() # run

    if (plots == True):
        # Make a graph of energies
        cons.visualize('energy')
        cons.visualize('walls')

    cons.writeVTKall()

    if (rendering == True):
        # Render images with raytracer
        cons.render(method = "pres", max_val = 2.0*N, verbose = False)


### SHEARING ###

# New class
shear = sphere.sim(np = cons.np, nw = cons.nw, sid = sim_id +
                    "-shear-N{}".format(N))

# Read last output file of initialization step
lastf = cons.status()
shear.readbin("../output/" + sim_id +
                "-cons-N{}.output{:0=5}.bin".format(N, lastf), verbose =
                False)

# Periodic x and y boundaries
shear.periodicBoundariesXY()

shear.rho[0] = rho_g
shear.g[2] = -g

# Disable particle viscosities
shear.gamma_n[0] = 0.0
shear.gamma_t[0] = 0.0

# Setup shear experiment
shear.shear(shear_strain_rate = 0.1)

# Set duration of simulation
shear.initTemporal(total = 10.0)

if (shearing == True):

    # Run sphere
    shear.run(dry = True)
    shear.run()

    if (plots == True):
        # Make a graph of energies
        shear.visualize('energy')
        shear.visualize('shear')

    shear.writeVTKall()

    if (rendering == True):
        # Render images with raytracer
        shear.render(method = "pres", max_val = 2.0*N, verbose = False)


### CREEP ###

# New class
creep = sphere.sim(np = cons.np, nw = cons.nw, sid = sim_id +
                    "-N{}-dpdx{}".format(N, dpdx))

# Read last output file of initialization step
lastf = shear.status()
creep.readbin("../output/" + sim_id +
                "-shear-N{}.output{:0=5}.bin".format(N, lastf), verbose =
                False)

# Periodic x and y boundaries
creep.periodicBoundariesXY()

# set fluid and solver properties
creep.initFluid(mu=8.9e-4, p=0.0, rho=rho_f, cfd_solver=1)  # water at 25 C
creep.setMaxIterations(2e5)
creep.setPermeabilityGrainSize()
creep.setFluidCompressibility(4.6e-10) # water at 25 C

# set fluid BCs
creep.setFluidTopNoFlow()
creep.setFluidBottomNoFlow()
creep.setFluidXFixedPressure()
creep.adaptiveGrid()

# set fluid pressures at the boundaries and internally
dx = creep.L[0]/creep.num[0]
for ix in range(creep.num[0]):
    x = ix + 0.5*dx
    creep.p_f[ix,:,:] = numpy.abs(creep.L[0]*dpdx) + x*dpdx

creep.zeroKinematics()

# Remove fixvel constraint from uppermost grains
creep.fixvel[numpy.nonzero(creep.x[:,2] > 0.5*creep.L[2])] = 0

# Produce regular coloring pattern
creep.checkerboardColors(creep.num[0], creep.num[1], creep.num[2])
creep.color[numpy.nonzero(creep.fixvel == 1)] == -1

# Adapt grid size during progressive deformation
creep.adaptiveGrid()

# Set duration of simulation
creep.initTemporal(total = 20.0)

if (creeping == True):

    # Run sphere
    creep.run(dry = True)
    creep.run()

    if (plots == True):
        # Make a graph of energies
        creep.visualize('energy')

    creep.writeVTKall()

    if (rendering == True):
        # Render images with raytracer
        creep.render(method = "pres", max_val = 2.0*N, verbose = False)
