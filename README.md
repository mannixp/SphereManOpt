# Gradient descent on a spherical manifold(s)


Optimisation code ....


**Example 1**

------------------

The following examples require an Anaconda environment with the parallelised spectral code [Dedalus installed](https://dedalus-project.org). Having installed Dedalus and activated the relevant conda environment both examples can be ran by executing the following commands.

**Example 2**


**Example 3**

Optimisation code to search the "*best*" kinematic dynamo in a triply periodic box following [(A.P. Willis, PRL, 2012)](https://doi.org/10.1103/PhysRevLett.109.251101). In contrast to the common approach of deriving the adjoint of the continuous forward equations, this code uses the adjoint of the discrete forward equations & objective function. 

The default arguments contained in FWD_Solve_PBox_IND_MHD.py are

`Rm=1, T=1, Npts = 24, dt = 0.001, Noise = True`

corresponding to the magentic Reynolds number, optimisation time window, the number of Fourier modes, the time-step and wether to initialise the velocity field with noise (True) or the analytical solution (False) given in [(A.P. Willis, PRL, 2012)](https://doi.org/10.1103/PhysRevLett.109.251101). Executing this script with 

`mpiexec -np 4 python3 FWD_Solve_PBox_IND_MHD.py && python3 plot_figure_PBox_FULL.py'

optimises both the velocity and magnetic fields using 4 cores and plots the resulting solution. *add comment about the gradient*
