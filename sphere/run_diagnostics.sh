#!/bin/sh

OUTFILE=diagnostics.txt

report() {
    echo >> $OUTFILE
    echo "#### $@ ###" >> $OUTFILE
    $@ >> $OUTFILE 2>&1
}

echo "Running diagnostics, this will take a few minutes. Please wait..."
date > $OUTFILE
report ls -lh .
report ls -lhR src
report ls -lhR python
report ls -lhR input
report ls -lhR Testing
report git show
report git status
report lspci
report uname -a
report cmake --version
report nvcc --version
report gcc --version
report cmake .
report make
report make test
report cat Testing/Temporary/LastTestsFailed.log
report cat Testing/Temporary/LastTest.log

rm -f $OUTFILE.gz
gzip $OUTFILE
echo "Diagnostics complete."
echo "Report bugs and unusual behavior to anders.damsgaard@geo.au.dk."
echo "Please attach the file $OUTFILE.gz to the mail."
