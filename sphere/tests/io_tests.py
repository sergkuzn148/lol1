#!/usr/bin/env python
from pytestutils import *
import sphere

#### Input/output tests ####
print("### Input/output tests ###")

# Generate data in python
orig = sphere.sim(np=100, nw=1, sid="test-initgrid")
orig.generateRadii(histogram=False)
orig.defaultParams()
orig.g[2] = 0.0
orig.initRandomGridPos()
orig.initTemporal(current=0.0, total=0.0)
orig.time_total=2.0*orig.time_dt
orig.time_file_dt = orig.time_dt
orig.writebin(verbose=False)

# Test the test
compare(orig, orig, "Comparison:")

# Test Python IO routines
py = sphere.sim()
py.readbin("../input/" + orig.sid + ".bin", verbose=False)
compare(orig, py, "Python IO:")

# Test C++ IO routines
#orig.run(verbose=True, hideinputfile=True)
orig.run(dry=True)
#orig.run(valgrind=True)
orig.run()
cpp = sphere.sim()
cpp.readbin("../output/" + orig.sid + ".output00000.bin", verbose=False)
compare(orig, cpp, "C++ IO:   ")

# Test CUDA IO routines
cuda = sphere.sim()
cuda.readbin("../output/" + orig.sid + ".output00001.bin", verbose=False)
cuda.time_current = orig.time_current
cuda.time_step_count = orig.time_step_count
compare(orig, cuda, "CUDA IO:  ")

# Remove temporary files
cleanup(orig)
