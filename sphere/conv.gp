#!/usr/bin/env gnuplot

set terminal dumb
#set terminal pngcairo
#set out 'output/'.sid.'-conv.png'

set title "Convergence evolution in CFD solver (".sid.")"
set xlabel "Time step"
set ylabel "Jacobi iterations"

plot 'output/'.sid.'-conv.log' with linespoints notitle
