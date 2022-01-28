#!/usr/bin/env bash

#mpiexec -np 1 python3 MHD_Couette_WORK_BPRESS.py

#mpiexec -np 1 python3 MHD_Couette_SETUP.py

#declare -i eps = 100 # Parameter divided by 10^3 


# Grab home directory to give as absolute path
BASEPATH="/Users/pmannix/Desktop/Nice_CASTOR/Discrete_Adjoint/";

# Files
SUBFOLD1=$BASEPATH$"/FWD_Solve_PBox_IND_MHD.py";
SUBFOLD2=$BASEPATH$"/plot_figure_PBox_FULL.py";

FOLD="Willis_Test_Rm1_T1"

rm -rf $FOLD;
mkdir -p $FOLD"/";
cd "./"$FOLD;

mpiexec -np 1 python3 $SUBFOLD1 # Value of epsilon
mpiexec -np 1 python3 $SUBFOLD2 # Value of epsilon

#mpiexec -np 1 python3 -m dedalus merge_procs snapshots --cleanup
#mpiexec -np 1 python3 -m dedalus merge_procs radial_profiles --cleanup
#mpiexec -np 1 python3 -m dedalus merge_procs scalar_data --cleanup
