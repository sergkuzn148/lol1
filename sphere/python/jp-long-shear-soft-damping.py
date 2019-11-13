#!/usr/bin/env python

# Import sphere functionality
import sphere

### EXPERIMENT SETUP ###
initialization = False
consolidation = False
shearing = True
rendering = False
plots = True

# CUDA device to use
device = 0

# Number of particles
np = 1e4

# Common simulation id
sim_id = "jp-long-shear-soft"

# Deviatoric stress [Pa]
devslist = [100e3]

### INITIALIZATION ###

# New class
init = sphere.sim(np=np, nd=3, nw=0, sid=sim_id + "-init")

# Save radii
init.generateRadii(mean=0.01)

# Use default params
init.defaultParams(k_n=1.16e7, k_t=1.16e7, gamma_n=100.0, mu_s=0.6, mu_d=0.6)

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
        # Make a graph of energies
        init.visualize('energy')

    init.writeVTKall()

    if (rendering):
        # Render images with raytracer
        init.render(method="angvel", max_val=0.3, verbose=False)


# For each normal stress, consolidate and subsequently shear the material
for devs in devslist:

    ### CONSOLIDATION ###

    # New class
    cons = sphere.sim(
        np=init.np,
        nw=1,
        sid=sim_id +
        "-cons-devs{}".format(devs))

    # Read last output file of initialization step
    lastf = sphere.status(sim_id + "-init")
    cons.readbin(
        "../output/" +
        sim_id +
        "-init.output{:0=5}.bin".format(lastf),
        verbose=False)

    # Periodic x and y boundaries
    cons.periodicBoundariesXY()

    # Setup consolidation experiment
    cons.consolidate(normal_stress=devs)

    # Set duration of simulation
    cons.initTemporal(total=3.0, epsilon=0.07)

    """
    cons.w_m[0] *= 0.001
    cons.mu_s[0] = 0.0
    cons.mu_d[0] = 0.0
    cons.gamma_wn[0] = 1e4
    cons.gamma_wt[0] = 1e4
    cons.contactmodel[0] = 1
    """

    if (consolidation):

        # Run sphere
        cons.run(dry=True)  # show values, don't run
        cons.run(device=device)  # run

        if (plots):
            # Make a graph of energies
            cons.visualize('energy')
            cons.visualize('walls')

        cons.writeVTKall()

        if (rendering):
            # Render images with raytracer
            cons.render(method="pres", max_val=2.0*devs, verbose=False)

    ### SHEARING ###

    # New class
    shear = sphere.sim(
        np=cons.np,
        nw=cons.nw,
        sid=sim_id +
        "-shear-devs{}".format(devs))

    # Read last output file of initialization step
    lastf = sphere.status(sim_id + "-cons-devs{}".format(devs))
    shear.readbin(
        "../output/" + sim_id + "-cons-devs{}.output{:0=5}.bin".format(devs,
                                                                       lastf), verbose=False)

    # Periodic x and y boundaries
    shear.periodicBoundariesXY()

    # Setup shear experiment
    shear.shear(shear_strain_rate=0.05)

    shear.gamma_wn[0] = 1.0e4
    shear.w_m[0] = shear.totalMass()

    # Set duration of simulation
    shear.initTemporal(total=240.0, epsilon=0.07)

    if (shearing):

        # Run sphere
        shear.run(dry=True)
        shear.run(device=device)

        if (plots):
            # Make a graph of energies
            shear.visualize('energy')
            shear.visualize('shear')

        shear.writeVTKall()

        if (rendering):
            # Render images with raytracer
            shear.render(method="pres", max_val=2.0*devs, verbose=False)
