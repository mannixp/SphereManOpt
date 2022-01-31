# KinematicDynamoDiscrete
Optimisation code to search the "best" kinematic dynamo in a periodic box following [(A.P. Willis, PRL, 2012)](https://doi.org/10.1103/PhysRevLett.109.251101). In contrast to the common approach of deriving the adjoint of the continuous forward equations, this code uses the adjoint of the discrete forward equations & objective function. 

Combing this approach with an efficient line-search and both Wolfe conditions superior convergence is achieved, reaching a relative max/min of the objective function with a smaller residual in less iterations. 


Running the code requires an anaconda environment with the parallelised spectral code [Dedalus installed](https://dedalus-project.org). Having installed Dedalus and activated the relevant conda environment the optimisation code can be run and the results plotted by executing:

`./run_PBox_problem.sh`

the default arguments contained in FWD_Solve_PBox_IND_MHD.py are

`Rm=1, T=1, Npts = 24, dt = 0.002, Noise = True`

and correspond to the magentic Reynolds number, optimisation time window, the number of Fourier modes, the time-step and wether to initialise the velocity field with noise or the analytical solution given in [(A.P. Willis, PRL, 2012)](https://doi.org/10.1103/PhysRevLett.109.251101). Executing this script with 

`DAL_LOOP(X0,[M_0],*args)`

optimises the magnetic field only, while

`DAL_LOOP(X0,[M_0,E_0],*args)`

optimises both the velocity and magnetic fields. To test the gradient approximation the function

'Adjoint_Gradient_Test(X0,dBx0,*args)`

is used. Additional options of interest contained in FWD_Solve_PBox_IND_MHD.py are

`Adjoint_type = "Discrete"/"Continuous"`

which chooses the gradient approximation used allowing for a comparison of performance.

