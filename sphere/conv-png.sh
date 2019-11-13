#!/bin/bash
INTERVAL=10
INTERVALDIV2=`expr $INTERVAL / 2`
if [[ "$1" == "" ]]; then
    echo "Usage: $0 <simulation id>"
    exit 1
else 
    feh --reload $INTERVALDIV2 --auto-zoom --image-bg black --scale-down output/${1}-conv.png &
    watch --interval $INTERVAL --no-title gnuplot -e \"sid=\'$1\'\" conv-png.gp
fi
