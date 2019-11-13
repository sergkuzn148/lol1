#!/usr/bin/env python

# Import sphere functionality
import sphere

### EXPERIMENT SETUP ###
initialization = False
consolidation  = True
shearing       = True
fluid          = True
rendering      = False
plots          = True

# Number of particles
np = 1e4

# Common simulation id
sim_id = "adapt-grid"

# Deviatoric stress [Pa]
devslist = [80e3]
#devs = 0

### INITIALIZATION ###

# New class
init = sphere.sim(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Save radii
init.generateRadii(mean = 0.02)

# Use default params
init.defaultParams(gamma_n = 100.0, mu_s = 0.6, mu_d = 0.6)
init.setYoungsModulus(7.0e9)

# Add gravity
init.g[2] = -9.81

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



# For each normal stress, consolidate and subsequently shear the material
for devs in devslist:

    ### CONSOLIDATION ###

    # New class
    cons = sphere.sim(np = init.np, nw = 1, sid = sim_id + "-cons-devs{}".format(devs))

    # Read last output file of initialization step
    lastf = init.status()
    cons.readbin("../output/" + sim_id + "-init.output{:0=5}.bin".format(lastf), verbose=False)

    # Periodic x and y boundaries
    cons.periodicBoundariesXY()

    if fluid:
        # set fluid and solver properties
        cons.initFluid(mu=2.080e-7, p=0.0, cfd_solver=1)
        cons.setFluidTopFixedPressure()
        cons.setFluidBottomNoFlow()
        cons.setMaxIterations(2e5)
        cons.setPermeabilityPrefactor(2.0e-16)
        cons.setFluidCompressibility(1/2.2e9)

    # Setup consolidation experiment
    cons.consolidate(normal_stress = devs)
    cons.adaptiveGrid()

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
            cons.render(method = "pres", max_val = 2.0*devs, verbose = False)


    ### SHEARING ###

    # New class
    shear = sphere.sim(np = cons.np, nw = cons.nw, sid = sim_id + "-shear-devs{}".format(devs))

    # Read last output file of initialization step
    lastf = cons.status()
    shear.readbin("../output/" + sim_id + "-cons-devs{}.output{:0=5}.bin".format(devs, lastf), verbose = False)

    # Periodic x and y boundaries
    shear.periodicBoundariesXY()

    if fluid:
        # set fluid and solver properties
        shear.initFluid(mu=2.080e-7, p=0.0, cfd_solver=1)
        shear.setFluidTopFixedPressure()
        shear.setFluidBottomNoFlow()
        shear.setMaxIterations(2e5)
        shear.setPermeabilityPrefactor(2.0e-16)
        shear.setFluidCompressibility(1/2.2e9)

    # Setup shear experiment
    shear.shear(shear_strain_rate = 0.05)
    shear.adaptiveGrid()

    # Set duration of simulation
    shear.initTemporal(total = 20.0)

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
            shear.render(method = "pres", max_val = 2.0*devs, verbose = False)
