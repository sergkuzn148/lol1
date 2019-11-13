#!/usr/bin/env python
import sphere
import numpy

relaxation = True
consolidation = False
water = False

id_prefix = 'channel3'
N = 10e3

cube = sphere.sim('cube-init')
cube.readlast()
cube.adjustUpperWall(z_adjust=1.0)

# Fill out grid with cubic packages
grid = numpy.array((
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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

sim = sphere.sim(id_prefix + '-relaxation', nw=0)

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
                             numpy.max(sim.x[:, 2] + sim.radius[:])*10.2])
                             #numpy.max(sim.x[:, 2] + sim.radius[:])*1.2])
sim.k_t[0] = 2.0/3.0*sim.k_n[0]

# sim.cleanup()
sim.writeVTK()
print("Number of particles: " + str(sim.np))


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
sim.initTemporal(total=3.0, file_dt=0.01, epsilon=0.07)
sim.time_dt[0] = 1.0e-20
sim.time_file_dt = sim.time_dt
sim.time_total = sim.time_file_dt*5.
sim.zeroKinematics()

if relaxation:
    sim.run(dry=True)
    sim.run()
    sim.writeVTKall()

exit()

# Consolidation under constant normal stress
if consolidation:
    sim.readlast()
    sim.id(id_prefix + '-' + str(int(N/1000.)) + 'kPa')
    sim.cleanup()
    sim.initTemporal(current=0.0, total=10.0, file_dt=0.01, epsilon=0.07)

    # fix lowest plane of particles
    I = numpy.nonzero(sim.x[:, 2] < 1.5*numpy.mean(sim.radius))
    sim.fixvel[I] = -1
    sim.color[I] = 0

    sim.zeroKinematics()

    # Wall parameters
    sim.mu_ws[0] = 0.5
    sim.mu_wd[0] = 0.5
    sim.gamma_wn[0] = 1.0e2
    sim.gamma_wt[0] = 1.0e2
    # sim.gamma_wn[0] = 0.0
    # sim.gamma_wt[0] = 0.0

    # Particle parameters
    sim.mu_s[0] = 0.5
    sim.mu_d[0] = 0.5
    sim.gamma_n[0] = 0.0
    sim.gamma_t[0] = 0.0

    # apply effective normal stress from upper wall
    sim.consolidate(normal_stress=N)

    sim.run(dry=True)
    #sim.run()
    #sim.writeVTKall()

## Add water
#if water:
    #sim.readlast()
    #sim.id(id_prefix + '-wet')
    #sim.wet()
    #sim.initTemporal(total=3.0, file_dt=0.01, epsilon=0.07)
#
    #sim.run(dry=True)
    #sim.run()
    #sim.writeVTKall()
