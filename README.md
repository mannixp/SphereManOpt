# SphereManOpt: Gradient descent on a spherical manifold(s)

Optimisation code to solve the minimisation problem:

$\underset{\boldsymbol{X} \in \mathcal{S}}{\text{min}} \quad J(\boldsymbol{X}) \quad$  where  $\quad \mathcal{S} = \[ \boldsymbol{X} \quad | \quad \langle \boldsymbol{X}, \boldsymbol{X} \rangle = E \]$,

is the spherical manifold of radius $E$. 

Given a (python) list of (numpy) vectors $X_i$

$X = \[X_0,X_1, ..., X_n\]$

a corresponding list of constraint amplitudes (float) $E_i$

$E = \[E_0, E_1, ...., E_n\]$

the accompanying routines 

`f,Grad_f,Inner_Product` 

which calculate the object-function $J(\boldsymbol{X})$, its Euclidean gradient $\nabla J(\boldsymbol{X})$ and the inner-product $\langle \boldsymbol{f}, \boldsymbol{g} \rangle$ respectively. Calling the routine

`RESIDUAL, FUNCT, X_opt = Optimise_On_Multi_Sphere(X,E,f,Grad_f,Inner_Product,args_f,args_IP)`

returns the optimal vector `X_opt` (as a list of numpy vectors) the residual errors `RESIDUAL` (as a list of floats) and the objective function evaluations `FUNCT` (as a list of floats) during the iterative optimisation procedure. The arguments `args_f = (), args_IP=()` are tuples which can be used to supply the necessary arguments to `f,Grad_f` and `Inner_Product` respectively. 

In addition to the above the remaining *optional arguments*, whose default values are

`err_tol = 1e-06, max_iters = 200, alpha_k = 1., LS = 'LS_wolfe', CG = True, callback=None`

specifiy the termination conditions `err_tol,max_iters`, the maximum/initial step-size `alpha_k`, the line-search routine ('Armijo' or 'LS_wolfe'), the gradient descent rotuine 'CG=True' (use a conjuagte-gradient update) or 'CG=False' (use a steepest-descent update) and provide the utility of a callback (function/callable) which takes the current iteration $k$ as its sole argument and allows the user to save or perform calculations on the current iterate's information.

**Example 1**

Optimisation code to search the largest principle component of a symmetric matrix M. The parameter `DIM` controls the dimension of the random symmetric matrix generated. Executing this script with 

`python3 PCA_example.py`

finds the principle component, using steepest-descent (SD) and conjugate-gradient (CG) methods, plots the residual error of each and compares `X_opt` with the solution found using numpy's built-in eigen-vector solver.

------------------  <>  ------------------  <>  ------------------  <>  ------------------  <>  ------------------  <>  ------------------ 

The following examples require an Anaconda environment with the parallelised spectral code [Dedalus installed](https://dedalus-project.org). Having installed Dedalus and activated the relevant conda environment both examples can be ran by executing the following commands. In both examples the option

`Adjoint_type = "Discrete" or "Continuous"`

allows one to select the gradient approximation used, while uncommenting and running the function 

`Adjoint_Gradient_Test(X_0,dX_0,	*objective_function_args)`

allows on to test the quality of the gradient approximation used.

**Example 2**

Optimisation code to search the *minimal perturbation* triggering transition to a localised state of the Swift-Hohenberg equation on a 1D periodic domain following [(D. Lecoanet, Phys. Rev. E., 2018)](https://link.aps.org/doi/10.1103/PhysRevE.97.012212). The default arguments contained in FWD_Solve_SH23.py are

`T=50, Npts = 256, dt = 0.1, M_0 = 0.0725`

corresponding to the optimisation time window, the number of Fourier modes, the time-step and applied perturbation amplitude.  Executing this script with 

`mpiexec -np 1 python3 FWD_Solve_SH23.py && python3 plot_figure_SH23_FULL.py`

optimises the perturbation using 1 core and plots the resulting solution.

**Example 3**

Optimisation code to search the "*best*" kinematic dynamo in a triply periodic box following [(A.P. Willis, PRL, 2012)](https://doi.org/10.1103/PhysRevLett.109.251101). The default arguments contained in FWD_Solve_PBox_IND_MHD.py are

`Rm=1, T=1, Npts = 24, dt = 0.001, Noise = True`

corresponding to the magentic Reynolds number, optimisation time window, the number of Fourier modes, the time-step and wether to initialise the velocity field with noise (True) or the analytical solution (False) given in [(A.P. Willis, PRL, 2012)](https://doi.org/10.1103/PhysRevLett.109.251101). The additional option

`Adjoint_type = "Discrete" or "Continuous"`

allow one to select the gradient approximation used. Executing this script with 

`mpiexec -np 4 python3 FWD_Solve_PBox_IND_MHD.py && python3 plot_figure_PBox_FULL.py`

optimises both the velocity and magnetic fields using 4 cores and plots the resulting solution.
