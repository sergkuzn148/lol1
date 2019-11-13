#!/bin/bash
if [[ "$1" == "" ]]; then
    echo "Usage: $0 <simulation id>"
    exit 1
else 
    watch --interval 5 --no-title gnuplot -e \"sid=\'$1\'\" conv.gp
fi
