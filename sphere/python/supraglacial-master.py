#!/usr/bin/env python

# sphere grain/fluid simulation: https://src.adamsgaard.dk/sphere
import sphere
import numpy

### EXPERIMENT SETUP ###
initialization = False
creeping       = True
plots          = True

# Common simulation id
sim_id = "supraglacial"

# Fluid-pressure gradient [Pa/m]
dpdz = 0.0

# Grain density
rho_g = 3600.0

# Fluid density
rho_f = 1000.0

# Gravitational acceleration
g = 9.8

# Slope
slope_angle = 20.0

# Number of particles
np = 1e4

device = 0  # automatically choose best GPU


### INITIALIZATION ###

# New class
init = sphere.sim(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Uniform diameters from 0.3 cm to 0.7 cm
init.generateRadii(psd = 'uni', mean = 0.0025, variance = 0.001)

# Use default params
init.defaultParams(gamma_n = 100.0, mu_s = 0.6, mu_d = 0.6)
init.setYoungsModulus(1e8)

# Disable wall viscosities
init.gamma_wn[0] = 0.0
init.gamma_wt[0] = 0.0

# Add gravity
init.g[2] = -g

# Periodic x and y boundaries
init.periodicBoundariesXY()

# Initialize positions in random grid (also sets world size)
hcells = np**(1.0/3.0)
init.initRandomGridPos(gridnum = [hcells-2, hcells-2, 1e9])

# Set duration of simulation
init.initTemporal(total = 5.0)

if (initialization == True):

    # Run sphere
    init.run(dry = True)
    init.run(device=device)

    if (plots == True):
        # Make a graph of energies
        init.visualize('energy')

    init.writeVTKall()


### CREEP ###

# New class
creep = sphere.sim(np = init.np,
                   sid = sim_id + "-slope{}-dpdz{}".format(slope_angle, dpdz))

# Read last output file of initialization step
creep.readbin("../output/" + sim_id + "-init.output{:0=5}.bin"
              .format(init.status()))

# Tilt gravity
creep.g[2] = -g*numpy.cos(numpy.deg2rad(slope_angle))
creep.g[0] = g*numpy.sin(numpy.deg2rad(slope_angle))

# Disable particle contact viscosities
creep.gamma_n[0] = 0.0
creep.gamma_t[0] = 0.0

# zero all velocities and accelerations
creep.zeroKinematics()

# Periodic x and y boundaries
creep.periodicBoundariesXY()

# Fit grid to grains
creep.adjustUpperWall(z_adjust=1.2)
creep.nw = 0   # no dynamic wall on top

# Fix bottom grains
z_min = numpy.min(creep.x[:,2] - creep.radius)
z_max = numpy.max(creep.x[:,2] + creep.radius)
d_max_below = numpy.max(creep.radius[numpy.nonzero(creep.x[:,2] <
    (z_max-z_min)*0.3)])*2.0
I = numpy.nonzero(creep.x[:,2] < (z_min + d_max_below))
creep.fixvel[I] = 1

# set fluid and solver properties
creep.initFluid(mu=8.9e-4, p=0.0, rho=rho_f, cfd_solver=1)  # water at 25 C
creep.setMaxIterations(2e5)
creep.setPermeabilityGrainSize()
creep.setFluidCompressibility(4.6e-10) # water at 25 C

# set fluid BCs
# creep.setFluidTopNoFlow()
# creep.setFluidBottomNoFlow()
# creep.setFluidXFixedPressure()
# creep.adaptiveGrid()
creep.setFluidTopFixedPressure()
creep.setFluidBottomFixedPressure()
creep.setFluidXPeriodic()
creep.setFluidYPeriodic()

# set fluid pressures at the boundaries and internally
dz = creep.L[2]/creep.num[2]
for iz in range(creep.num[2]):
    z = iz + 0.5*dz
    creep.p_f[:,:,iz] = numpy.abs(creep.L[2]*dpdz) + z*dpdz

# Remove fixvel constraint from uppermost grains
#creep.fixvel[numpy.nonzero(creep.x[:,2] > 0.5*creep.L[2])] = 0

# Produce regular coloring pattern
creep.checkerboardColors(creep.num[0], creep.num[1], creep.num[2])
creep.color[numpy.nonzero(creep.fixvel == 1)] == -1

# Adapt grid size during progressive deformation
#creep.adaptiveGrid()

# Set duration of simulation
creep.initTemporal(total=5.0, file_dt=0.01)

if (creeping == True):

    # Run sphere
    creep.run(dry = True)
    creep.run(device=device)

    if (plots == True):
        # Make a graph of energies
        creep.visualize('energy')

    creep.writeVTKall()
