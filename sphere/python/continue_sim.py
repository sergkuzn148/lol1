#!/usr/bin/env python
import sphere
import sys

def print_usage():
    print('Usage: ' + sys.argv[0]
            + ' <simulation id> <fluid> <device> [duration]')
    print('where "simulation id" is a string and "fluid" is either 0 or 1.')
    print('"device" is the number of the GPU device.')
    print('The total simulation can optionally be defined to continue from the '
            'current time and "duration" seconds more.')

if len(sys.argv) < 2:
    print_usage()
    sys.exit(1)

else:
    sim = sphere.sim(sys.argv[1], fluid = int(sys.argv[2]))
    sim.readlast()
    if len(sys.argv) == 5:
        sim.time_total[0] = sim.time_current[0] + float(sys.argv[4])
    sim.run(device=sys.argv[3])
