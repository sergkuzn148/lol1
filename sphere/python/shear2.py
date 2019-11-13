#!/usr/bin/env python
import sphere

fluid = True

sim = sphere.sim('cons2-20kPa')
sim.readlast()
c = 1.0
sim.id('shear2-20kPa-c=' + str(c))
sim.shear(1.0/20.0)

if fluid:
    #sim.num[2] *= 2
    #sim.L[2] *= 2.0
    sim.initFluid(mu=1.787e-6, p=600.0e3, hydrostatic=True)
    sim.setFluidBottomNoFlow()
    sim.setFluidTopFixedPressure()
    sim.setDEMstepsPerCFDstep(100)
    sim.setMaxIterations(2e5)
    sim.c_grad_p[0] = c

sim.checkerboardColors()
sim.initTemporal(20.0, epsilon=0.07)
sim.run(device=1)
sim.writeVTKall()
