#!/usr/bin/env python

# Import sphere functionality
import sphere

initialization = True
plots = True

# CUDA device to use
device = 0

# Number of particles
np = 1e4

# Common simulation id
sim_id = 'init-1e4'

init = sphere.sim(np=np, nd=3, nw=0, sid=sim_id)

# Save radii
init.generateRadii(mean=0.01)

# Use default params
init.defaultParams(gamma_n=100.0, mu_s=0.6, mu_d=0.6)
init.setStiffnessNormal(1.16e7)
init.setStiffnessTangential(1.16e7)

# Add gravity
init.g[2] = -9.81

# Periodic x and y boundaries
init.periodicBoundariesXY()

# Initialize positions in random grid (also sets world size)
hcells = np**(1.0/3.0)
init.initRandomGridPos(gridnum=[hcells, hcells, 1e9])

# Set duration of simulation
init.initTemporal(total=10.0, epsilon=0.07)

if (initialization):

    # Run sphere
    init.run(dry=True)
    init.run(device=device)

    if (plots):
        init.visualize('energy')

    init.writeVTKall()
