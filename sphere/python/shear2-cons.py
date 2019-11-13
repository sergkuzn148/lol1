#!/usr/bin/env python
import sphere

sim = sphere.sim('init2', np=10000)
sim.readlast()
sim.consolidate(20.0e3)
sim.id('cons2-20kPa')
sim.defaultParams()
sim.initTemporal(5.0, epsilon=0.07)
sim.run()
sim.writeVTKall()
