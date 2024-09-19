#!/bin/bash -l
#module load lang/Python
#source ../../../python3venvs/optfl_env/bin/activate

INPUTFILE=$(pwd)/../src/nsga2.py 
echo ${@: 2}
time python $INPUTFILE ${@: 2}
