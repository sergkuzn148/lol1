#!/usr/bin/env python

# Import sphere functionality
import sphere

### EXPERIMENT SETUP ###
initialization = True
consolidation  = True
shearing       = True
rendering      = True
plots          = True

# Number of particles
np = 1e4

# Common simulation id
sim_id = "shear-test"

# Deviatoric stress [Pa]
Nlist = [80e3]

### INITIALIZATION ###

# New class
init = sphere.sim(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Save radii
init.generateRadii(mean = 0.02)

# Use default params
init.defaultParams(gamma_n = 100.0, mu_s = 0.6, mu_d = 0.6)

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
for N in Nlist:

    ### CONSOLIDATION ###

    # New class
    cons = sphere.sim(np = init.np, nw = 1, sid = sim_id +
                      "-cons-N{}".format(N))

    # Read last output file of initialization step
    lastf = status(sim_id + "-init")
    cons.readbin("../output/" + sim_id + "-init.output{:0=5}.bin".format(lastf), verbose=False)

    # Periodic x and y boundaries
    cons.periodicBoundariesXY()

    # Setup consolidation experiment
    cons.consolidate(normal_stress = N, periodic = init.periodic)
    cons.adaptiveGrid()


    # Set duration of simulation
    cons.initTemporal(total = 1.5)

    """
    cons.w_m[0] *= 0.001
    cons.mu_s[0] = 0.0
    cons.mu_d[0] = 0.0
    cons.gamma_wn[0] = 1e4
    cons.gamma_wt[0] = 1e4
    cons.contactmodel[0] = 1
    """

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
    lastf = status(sim_id + "-cons-N{}".format(N))
    shear.readbin("../output/" + sim_id +
                  "-cons-N{}.output{:0=5}.bin".format(N, lastf),
                  verbose = False)

    # Periodic x and y boundaries
    shear.periodicBoundariesXY()

    # Setup shear experiment
    shear.shear(shear_strain_rate = 0.05, periodic = init.periodic)
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
            shear.render(method = "pres", max_val = 2.0*N, verbose = False)
