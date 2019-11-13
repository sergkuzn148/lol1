#!/usr/bin/env python
'''
Validate the implemented contact models by observing the behavior of one or two
particles.
'''

import sphere
import numpy
import pytestutils

### Wall-particle interaction ##################################################

## Linear elastic collisions

# Normal impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.sim(np=1, nw=0, sid='contactmodeltest')
sphere.cleanup(orig)
orig.radius[:] = 1.0
orig.x[0,:] = [5.0, 5.0, 1.05]
orig.vel[0,2] = -0.1
orig.defineWorldBoundaries(L=[10,10,10])
orig.gamma_wn[0] = 0.0  # Disable wall viscosity
orig.gamma_wt[0] = 0.0  # Disable wall viscosity
orig.initTemporal(total = 1.0, file_dt = 0.01)
#orig.time_dt = orig.time_dt*0.1
moment_before = orig.totalKineticEnergy()
orig.run(verbose=False)
#orig.writeVTKall()
orig.readlast(verbose=False)
pytestutils.compareFloats(orig.vel[0,2], 0.1,\
        "Elastic normal wall collision (1/2):")
moment_after = orig.totalKineticEnergy()
#print(moment_before)
#print(moment_after)
#print("time step: " + str(orig.time_dt[0]))
#print(str((moment_after[0]-moment_before[0])/moment_before[0]*100.0) + " %")
pytestutils.compareFloats(moment_before, moment_after,\
        "Elastic normal wall collision (2/2):")

# Oblique impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.sim(np=1, sid='contactmodeltest')
orig.radius[:] = 1.0
orig.x[0,:] = [5.0, 5.0, 1.05]
orig.vel[0,2] = -0.1
orig.vel[0,0] =  0.1
orig.defineWorldBoundaries(L=[10,10,10])
orig.gamma_wn[0] = 0.0  # Disable wall viscosity
orig.gamma_wt[0] = 0.0  # Disable wall viscosity
orig.initTemporal(total = 0.3, file_dt = 0.01)
moment_before = orig.totalKineticEnergy()
orig.run(verbose=False)
#orig.writeVTKall()
orig.readlast(verbose=False)
moment_after = orig.totalKineticEnergy()
pytestutils.compareFloats(moment_before, moment_after,\
        "       45 deg. wall collision:\t")

## Visco-elastic collisions

# Normal impact with normal viscous damping. Test that the lost kinetic energy
# is saved as dissipated viscous energy
orig = sphere.sim(np=1, sid='contactmodeltest')
orig.radius[:] = 1.0
orig.x[0,:] = [5.0, 5.0, 1.05]
orig.vel[0,2] = -0.1
orig.defineWorldBoundaries(L=[10,10,10])
orig.gamma_wn[0] = 1.0e6
orig.gamma_wt[0] = 0.0
orig.initTemporal(total = 1.0, file_dt = 0.01)
Ekin_before = orig.energy('kin')
orig.run(verbose=False)
#orig.writeVTKall()
orig.readlast(verbose=False)
Ekin_after = orig.energy('kin')
Ev_after = orig.energy('visc_n')
#print("Ekin_before = " + str(Ekin_before) + " J")
#print("Ekin_after  = " + str(Ekin_after) + " J")
pytestutils.test(Ekin_before > Ekin_after,
        "Viscoelastic normal wall collision (1/2):")
pytestutils.compareFloats(Ekin_before, Ekin_after+Ev_after,\
        "Viscoelastic normal wall collision (2/2):", tolerance=0.05)

# Oblique impact: Check for conservation of momentum (sum(v_i*m_i))
orig = sphere.sim(np=1, sid='contactmodeltest')
orig.radius[:] = 1.0
orig.x[0,:] = [5.0, 5.0, 1.05]
orig.vel[0,2] = -0.1
orig.vel[0,0] =  0.1
orig.defineWorldBoundaries(L=[10,10,10])
orig.gamma_wn[0] = 1.0e6
orig.gamma_wt[0] = 1.0e6
orig.initTemporal(total = 1.0, file_dt = 0.01)
E_kin_before = orig.energy('kin')
orig.run(verbose=False)
#orig.writeVTKall()
orig.readlast(verbose=False)
#Ekin_after = orig.energy('kin')
#Erot_after = orig.energy('rot')
#Es_after = orig.energy('shear')
#pytestutils.compareFloats(Ekin_before,\
        #Ekin_after+Erot_after+Es_after,\
        #"            45 deg. wall collision:", tolerance=0.03)
pytestutils.test(Ekin_before > Ekin_after,
        "            45 deg. wall collision (1/2):")
pytestutils.test((orig.angvel[0,0] == 0.0 and orig.angvel[0,1] > 0.0 \
        and orig.angvel[0,2] == 0.0),
        "            45 deg. wall collision (2/2):")



sphere.cleanup(orig)
