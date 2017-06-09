#!/usr/bin/env bash

if [ $# -ne 3 ] ; then
  echo "Usage: $(basename "$0") <count> <breadth> <depth>"
  exit
fi

echo Catching $1 copies of $3layers$2each
for i in `seq 1 $1`; do echo $i; python MLOOPQuickTest.py $2 $3 >> 3_3_$3layers$2each.dat; done
