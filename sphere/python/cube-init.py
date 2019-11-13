#!/usr/bin/env python
import sphere

init = sphere.sim('cube-init', np=1e3)

init.generateRadii(psd='uni', mean=0.01, variance=0.002)

init.periodicBoundariesXY()

# Initialize positions in random grid (also sets world size)
init.initRandomGridPos(gridnum=(7, 7, 1e12))

# Disable friction to dissipate energy fast
init.k_n[0] = 1.0e8
init.mu_s[0] = 0.5
init.mu_d[0] = 0.5

# Choose the tangential contact model
# 1) Visco-frictional (somewhat incorrect, fast computations)
# 2) Elastic-viscous-frictional (more correct, slow computations in dense
# packings)
init.contactmodel[0] = 2

# Add gravitational acceleration
init.g[2] = -10.0

# Set duration of simulation, automatically determine timestep, etc.
init.initTemporal(total=6.0, file_dt=0.2)
print(init.num)

init.run(dry=True)
init.run()
init.writeVTKall()
