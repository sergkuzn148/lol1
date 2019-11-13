#!/usr/bin/env python
'''
Validate the implemented contact models by observing the behavior of two
particles.
'''

import sphere
import numpy
import pytestutils

### Particle-particle interaction ##############################################

## Linear elastic collisions

# Normal impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
#orig.radius[:] = [1.0, 2.0]
orig.radius[:] = [1.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 4.05]
orig.setYoungsModulus(7.0e9)
v_orig = 1
orig.vel[0,2] = v_orig
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(dry=True)
orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.vel[0,2], after.vel[1,2],\
        "Elastic normal collision (1/4):")
#print(orig.totalKineticEnergy())
#print(after.totalKineticEnergy())
pytestutils.compareFloats(orig.totalKineticEnergy(), after.totalKineticEnergy(),\
        "Elastic normal collision (2/4):")

# Normal impact with different sizes: Check for conservation of momentum
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
orig.radius[:] = [2.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 5.05]
orig.setYoungsModulus(7.0e9)
orig.vel[0,2] = 1.0
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.totalKineticEnergy(), after.totalKineticEnergy(),\
        "Elastic normal collision (3/4):")

# Normal impact with different sizes: Check for conservation of momentum
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
orig.radius[:] = [1.0, 2.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 5.05]
orig.setYoungsModulus(7.0e9)
orig.vel[0,2] = 1.0
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.totalKineticEnergy(), after.totalKineticEnergy(),\
        "Elastic normal collision (4/4):")


## Linear viscous-elastic collisions

# Normal impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
orig.radius[:] = [1.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 4.05]
orig.setYoungsModulus(7.0e9)
v_orig = 1
orig.vel[0,2] = v_orig
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)
orig.gamma_n[0] = 1.0e6

orig.run(verbose=False)
after.readlast(verbose=False)
#print(orig.totalKineticEnergy())
#print(after.totalKineticEnergy())
#print(after.totalViscousEnergy())
pytestutils.test(orig.vel[0,2] > after.vel[1,2],\
        "Viscoelastic normal collision (1/4):")
pytestutils.compareFloats(orig.totalKineticEnergy(),
                          after.totalKineticEnergy()
                          + after.totalViscousEnergy(),
        "Viscoelastic normal collision (2/4):", tolerance=0.05)

# Normal impact with different sizes: Check for conservation of momentum
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
orig.radius[:] = [2.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 5.05]
orig.setYoungsModulus(7.0e9)
orig.vel[0,2] = 1.0
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)
orig.gamma_n[0] = 1.0e6

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.totalKineticEnergy(),
                          after.totalKineticEnergy()
                          + after.totalViscousEnergy(),
        "Viscoelastic normal collision (3/4):", tolerance=0.05)

# Normal impact with different sizes: Check for conservation of momentum
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
orig.radius[:] = [1.0, 2.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 5.05]
orig.setYoungsModulus(7.0e9)
orig.vel[0,2] = 1.0
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)
orig.gamma_n[0] = 1.0e6

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.totalKineticEnergy(),
                          after.totalKineticEnergy()
                          + after.totalViscousEnergy(),
        "Viscoelastic normal collision (4/4):", tolerance=0.05)



## Oblique elastic collisions

# Normal impact, low angle, no slip
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
#orig.radius[:] = [1.0, 2.0]
orig.radius[:] = [1.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 4.05]
orig.setYoungsModulus(7.0e9)
orig.vel[0,2] = 1
orig.vel[0,0] = 1
orig.mu_s[0] = 1e9 # no slip
orig.mu_d[0] = 1e9 # no slip
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.test((after.angvel[:,1] < 0.0).all(),
                 "Oblique normal collision (1/8):")
pytestutils.compareFloats(orig.totalKineticEnergy(),
                          after.totalKineticEnergy()
                          + after.totalRotationalEnergy(),
                          "Oblique normal collision (2/8):", tolerance=0.05)

# Normal impact, low angle, slip
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
#orig.radius[:] = [1.0, 2.0]
orig.radius[:] = [1.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 4.05]
orig.setYoungsModulus(7.0e9)
orig.vel[0,2] = 1
orig.vel[0,0] = 1
orig.mu_s[0] = 0.3
orig.mu_d[0] = 0.3
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(verbose=False)
after.readlast(verbose=False)
pytestutils.compareFloats(orig.totalKineticEnergy(),
                          after.totalKineticEnergy()
                          + after.totalRotationalEnergy()
                          + after.totalFrictionalEnergy(),
                          "Oblique normal collision (3/8):", tolerance=0.05)
pytestutils.test((after.angvel[:,1] < 0.0).all(),
                 "Oblique normal collision (4/8):")
pytestutils.test(after.totalFrictionalEnergy() > 0.0,
                 "Oblique normal collision (5/8):")

# Normal impact, low angle, slip, viscous damping tangentially
orig = sphere.sim(np=2, sid='contactmodeltest')
after = sphere.sim(np=2, sid='contactmodeltest')
sphere.cleanup(orig)
#orig.radius[:] = [1.0, 2.0]
orig.radius[:] = [1.0, 1.0]
orig.x[0,:] = [5.0, 5.0, 2.0]
orig.x[1,:] = [5.0, 5.0, 4.05]
orig.setYoungsModulus(7.0e9)
orig.vel[0,2] = 1
orig.vel[0,0] = 1
orig.mu_s[0] = 0.3
orig.mu_d[0] = 0.3
orig.gamma_t[0] = 1.0e3
orig.defineWorldBoundaries(L=[10,10,10])
orig.initTemporal(total = 0.1, file_dt = 0.01)

orig.run(verbose=False)
after.readlast(verbose=False)
print(after.totalViscousEnergy())
pytestutils.compareFloats(orig.totalKineticEnergy(),
                          after.totalKineticEnergy()
                          + after.totalRotationalEnergy()
                          + after.totalFrictionalEnergy()
                          + after.totalViscousEnergy(),
                          "Oblique normal collision (6/8):", tolerance=0.05)
pytestutils.test((after.angvel[:,1] < 0.0).all(),
                 "Oblique normal collision (7/8):")
pytestutils.test(after.totalFrictionalEnergy() > 0.0,
                 "Oblique normal collision (8/8):")
pytestutils.test(after.totalFrictionalEnergy() > 0.0,
                 "Oblique normal collision (8/8):")

orig.cleanup()
