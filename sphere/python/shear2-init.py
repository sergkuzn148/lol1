#!/usr/bin/env python
import sphere

sim = sphere.sim('init2', np=10000)
sim.generateRadii(psd='uni', mean=0.02, variance=0.01)
sim.initRandomGridPos([12, 12, 1000])
sim.initTemporal(10.0, file_dt=0.05, epsilon=0.07)
sim.gamma_n[0] = 1000.0
sim.gamma_wn[0] = 1000.0
sim.periodicBoundariesXY()
sim.g[2] = -9.81
sim.run()
sim.writeVTKall()
