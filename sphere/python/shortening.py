#!/usr/bin/env python
import sphere
import numpy

cube = sphere.sim('cube-init')
cube.readlast()
cube.adjustUpperWall(z_adjust=1.0)

# Fill out grid with cubic packages
grid = numpy.array((
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [ 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ))

# World dimensions and cube grid
nx = 1                # horizontal (thickness) cubes
ny = grid.shape[1]    # horizontal cubes
nz = grid.shape[0]    # vertical cubes
dx = cube.L[0]
dy = cube.L[1]
dz = cube.L[2]
Lx = dx*nx
Ly = dy*ny
Lz = dz*nz

sim = sphere.sim('shortening-relaxation', nw=0)

# insert particles into each cube in 90 degree CCW rotated coordinate system
# around y
for z in range(nz):
    for y in range(ny):
        for x in range(nx):

            if (grid[z,y] == 0):
                continue # skip to next iteration

            for i in range(cube.np):
                # x=x, y=Ly-z, z=y
                pos = [ cube.x[i,0] + x*dx,
                        Ly - ((dz - cube.x[i,2]) + z*dz),
                        cube.x[i,1] + y*dy ]
                sim.addParticle(pos, radius=cube.radius[i], color=grid[z,y])

# move to x=0
min_x = numpy.min(sim.x[:,0] - sim.radius[:])
sim.x[:,0] = sim.x[:,0] - min_x

# move to y=0
min_y = numpy.min(sim.x[:,1] - sim.radius[:])
sim.x[:,1] = sim.x[:,1] - min_y

# move to z=0
min_z = numpy.min(sim.x[:,2] - sim.radius[:])
sim.x[:,2] = sim.x[:,2] - min_z

#sim.defineWorldBoundaries(L=[Lx, Lz*3, Ly])
sim.defineWorldBoundaries(L=[numpy.max(sim.x[:,0] + sim.radius[:]), Lz*3, Ly])
sim.k_t[0] = 2.0/3.0*sim.k_n[0]

#sim.cleanup()
sim.writeVTK()
print(sim.np)


## Relaxation

# Add gravitational acceleration
# Flip geometry so the upper wall pushes downwards
sim.g[0] = 0
sim.g[1] = -9.81
sim.g[2] = 0

sim.setDampingNormal(5.0e1)
sim.setDampingTangential(1.0e1)

#sim.periodicBoundariesX()
sim.normalBoundariesXY()
sim.uniaxialStrainRate(wvel = 0.0)

# Set duration of simulation, automatically determine timestep, etc.
sim.initTemporal(total=3.0, file_dt = 0.01)
sim.zeroKinematics()

sim.run(dry=True)
sim.run()
sim.writeVTKall()


## Shortening
sim = sphere.sim('shortening-relaxation', nw=1)
sim.readlast()
sim.sid = 'shortening'
sim.cleanup()
sim.initTemporal(current=0.0, total=20.0, file_dt = 0.01, epsilon=0.07)

# set colors again
y_min = numpy.min(sim.x[:,1])
y_max = numpy.max(sim.x[:,1])
z_min = numpy.min(sim.x[:,2])
z_max = numpy.max(sim.x[:,2])
color_ny = 6
color_nz = int((z_max - z_min)/(y_max - y_min)*color_ny)
color_dy = y_max/color_ny
color_dz = z_max/color_nz
color_y = numpy.arange(0.0, y_max, ny)
color_z = numpy.arange(0.0, z_max, nz)

# 1 or 2 in horizontal layers
#for i in range(ny-1):
    #I = numpy.nonzero((sim.x[:,1] >= color_y[i]) & (sim.x[:,1] <= color_y[i+1]))
    #sim.color[I] = i%2 + 1

# 1 or 3 in checkerboard
for i in range(sim.np):
    iy = numpy.floor((sim.x[i,1] - y_min)/(y_max/color_ny))
    iz = numpy.floor((sim.x[i,2] - z_min)/(z_max/color_nz))
    sim.color[i] = (-1)**iy + (-1)**iz + 1

# fix lowest plane of particles
I = numpy.nonzero(sim.x[:,1] < 1.5*numpy.mean(sim.radius))
sim.fixvel[I] = -1
sim.color[I] = 0
sim.x[I,1] = 0.0 # move into center into lower wall to avoid stuck particles

# fix left-most plane of particles
I = numpy.nonzero(sim.x[:,2] < 1.5*numpy.mean(sim.radius))
sim.fixvel[I] = -1
sim.color[I] = 0

# fix right-most plane of particles
I = numpy.nonzero((sim.x[:,2] > z_max - 1.5*numpy.mean(sim.radius)) &
        (sim.x[:,1] > 10.0*numpy.mean(sim.radius)))
sim.fixvel[I] = -1
sim.color[I] = 0

#sim.normalBoundariesXY()
#sim.periodicBoundariesX()
sim.zeroKinematics()

# Wall parameters
sim.mu_ws[0] = 0.5
sim.mu_wd[0] = 0.5
#sim.gamma_wn[0] = 1.0e2
#sim.gamma_wt[0] = 1.0e2
sim.gamma_wn[0] = 0.0
sim.gamma_wt[0] = 0.0

# Particle parameters
sim.mu_s[0] = 0.5
sim.mu_d[0] = 0.5
sim.gamma_n[0] = 1000.0
sim.gamma_t[0] = 1000.0

# push down upper wall
compressional_strain = 0.5
wall_velocity = -compressional_strain*(z_max - z_min)/sim.time_total[0]
sim.uniaxialStrainRate(wvel = wall_velocity)
sim.vel[I,2] = wall_velocity

sim.run(dry=True)
sim.run()
sim.writeVTKall()
