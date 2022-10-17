#!/usr/bin/env bash

# Grab home directory to give as absolute path
BASEPATH="/Users/pmannix/Desktop/Nice_CASTOR/SphereManOpt_Proj/SphereManOpt";

# Files
SUBFOLD1=$BASEPATH$"/FWD_Solve_Poiseuille.py";
SUBFOLD2=$BASEPATH$"/plot_figure_Poiseuille.py";

FOLD="Test"

rm -rf $FOLD;
mkdir -p $FOLD"/";
cd "./"$FOLD;

mpiexec -np 4 python3 $SUBFOLD1 # Value of epsilon
mpiexec -np 1 python3 $SUBFOLD2 # Value of epsilon
