#!/usr/bin/env python

# Import sphere functionality
import sphere

### EXPERIMENT SETUP ###
initialization = True
consolidation  = True
relaxation     = True
shearing       = True
rendering      = False
plots          = True

# Number of particles
np = 2e4

# Common simulation id
sim_id = "shear-test-ocr"

# Effective normal stresses during consolidation [Pa]
Nlist = [10e3, 25e3, 50e3, 100e3, 250e3, 500e3]

# Effective normal stresses during relaxation and shear [Pa]
Nshear = 10e3

### INITIALIZATION ###

# New class
init = sphere.sim(np = np, nd = 3, nw = 0, sid = sim_id + "-init")

# Save radii with uniform size distribution
init.generateRadii(psd = 'uni', mean = 1e-2, variance = 2e-3, histogram = True)

# Set mechanical parameters
init.setYoungsModulus(7e8)
init.setStaticFriction(0.5)
init.setDynamicFriction(0.5)
init.setDampingNormal(5e1)
init.setDampingTangential(0.0)

# Add gravitational acceleration
init.g[0] = 0.0
init.g[1] = 0.0
init.g[2] = -9.81

# Periodic x and y boundaries
init.periodicBoundariesXY()

# Initialize positions in random grid (also sets world size)
hcells = np**(1.0/3.0)
init.initRandomGridPos(gridnum = [hcells, hcells, 1e9])

init.checkerboardColors(nx=init.num[0]/2, ny=init.num[1]/2, nz=init.num[2]/2)

# Set duration of simulation
init.initTemporal(total = 10.0, epsilon = 0.07)

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
    lastf = sphere.status(sim_id + "-init")
    cons.readbin("../output/" + sim_id + "-init.output{:0=5}.bin".format(lastf), verbose=False)
    cons.setDampingNormal(0.0)

    # Periodic x and y boundaries
    cons.periodicBoundariesXY()

    # Setup consolidation experiment
    cons.consolidate(normal_stress = N)
    cons.adaptiveGrid()
    cons.checkerboardColors(nx=cons.num[0]/2, ny=cons.num[1]/2, nz=cons.num[2]/2)

    # Set duration of simulation
    cons.initTemporal(total = 4.0)

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


    ### RELAXATION at Nshear ###
    relax = sphere.sim(np = cons.np, nw = cons.nw, sid = sim_id +
                       "-relax-from-N{}".format(N))
    lastf = sphere.status(sim_id + "-cons-N{}".format(N))
    relax.readbin("../output/" + sim_id +
                  "-cons-N{}.output{:0=5}.bin".format(N, lastf))

    relax.periodicBoundariesXY()

    # Setup relaxation experiment
    relax.consolidate(normal_stress = Nshear)
    relax.adaptiveGrid()
    relax.checkerboardColors(nx=relax.num[0]/2, ny=relax.num[1]/2,
            nz=relax.num[2]/2)

    # Set duration of simulation
    relax.initTemporal(total = 3.0)

    if (relaxation == True):

        # Run sphere
        relax.run(dry = True) # show values, don't run
        relax.run() # run

        if (plots == True):
            # Make a graph of energies
            relax.visualize('energy')
            relax.visualize('walls')

        relax.writeVTKall()

        if (rendering == True):
            # Render images with raytracer
            relax.render(method = "pres", max_val = 2.0*Nshear, verbose = False)

    ### SHEARING ###

    # New class
    shear = sphere.sim(np = relax.np, nw = relax.nw, sid = sim_id +
                       "-shear-N{}-OCR{}".format(Nshear, N/Nshear))

    # Read last output file of initialization step
    lastf = sphere.status(sim_id + "-relax-from-N{}".format(N))
    shear.readbin("../output/" + sim_id +
                  "-relax-from-N{}.output{:0=5}.bin".format(N, lastf),
                  verbose = False)

    # Periodic x and y boundaries
    shear.periodicBoundariesXY()

    # Setup shear experiment
    shear.shear(shear_strain_rate = 0.05)
    shear.adaptiveGrid()
    shear.checkerboardColors(nx=shear.num[0]/2, ny=shear.num[1]/2,
            nz=shear.num[2]/2)

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
