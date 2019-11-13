#!/usr/bin/env python

# Import sphere functionality
import sphere

# EXPERIMENT SETUP #
initialization = True
consolidation = True
plots = True

# CUDA device to use
device = 0

# Number of particles
np = 6634

# Common simulation id
sim_id = "alejandro"

# Consolidation stress [Pa]
N = 10e3

# INITIALIZATION #

# New class
init = sphere.sim(np=np, nd=3, nw=0, sid=sim_id + "-init")

# Save radii
init.generateRadii(mean=0.01)

# Add viscous damping to quickly dissipate kinetic energy
init.defaultParams(k_n=1.16e7, k_t=1.16e7, gamma_n=100.0, mu_s=0.5, mu_d=0.5)
init.gamma_wn[0] = 10000.0

# Add gravity
init.g[2] = -9.81

# Periodic x and y boundaries
init.periodicBoundariesX()

# Initialize positions in random grid (also sets world size)
init.initRandomGridPos(gridnum=[24, 24, 1e9])

# Set duration of simulation
init.initTemporal(total=10.0, epsilon=0.07)

if (initialization):

    # Run sphere
    init.run(dry=True)
    init.run(device=device)

    if (plots):
        # Make a graph of energies
        init.visualize('energy')

    init.writeVTKall()


# CONSOLIDATION #

# New class
cons = sphere.sim(
    np=init.np,
    nw=1,
    sid=sim_id +
    "-cons-N={}".format(N))

# Read last output file of initialization step
lastf = sphere.status(sim_id + "-init")
cons.readbin(
    "../output/" +
    sim_id +
    "-init.output{:0=5}.bin".format(lastf),
    verbose=False)

cons.periodicBoundariesX()

# Setup consolidation experiment
cons.consolidate(normal_stress=N)
cons.w_m[0] = cons.totalMass()

# Disable all viscosities
cons.gamma_n[0] = 0.0
cons.gamma_t[0] = 0.0
cons.gamma_wn[0] = 0.0
cons.gamma_wt[0] = 0.0

# Set duration of simulation
cons.initTemporal(total=5.0, epsilon=0.07)

if (consolidation):

    # Run sphere
    cons.run(dry=True)  # show values, don't run
    cons.run(device=device)  # run

    if (plots):
        # Make a graph of energies
        cons.visualize('energy')
        cons.visualize('walls')

    cons.writeVTKall()
