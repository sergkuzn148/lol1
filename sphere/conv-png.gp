#!/usr/bin/env gnuplot

#set terminal pngcairo
set terminal pngcairo background "#002b36"
set border linecolor rgbcolor "#657b83"
set key textcolor rgbcolor "#657b83"
set out 'output/'.sid.'-conv.png'

set title "Convergence evolution in CFD solver (".sid.")" textcolor rgb "#93a1a1"
set xlabel "Time step" textcolor rgb "#657b83"
set ylabel "Jacobi iterations" textcolor rgb "#657b83"
set grid

plot 'output/'.sid.'-conv.log' with linespoints notitle linecolor rgb "#657b83"
