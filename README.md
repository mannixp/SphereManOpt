# SphereManOpt: Gradient descent on a spherical manifold(s)

[![DOI](https://zenodo.org/badge/453104430.svg)](https://zenodo.org/badge/latestdoi/453104430)

Optimisation code to solve the minimisation problem:

$\underset{\boldsymbol{X} \in \mathcal{S}}{\text{min}} \quad J(\boldsymbol{X}) \quad$  where  $\quad \mathcal{S} = \[ \boldsymbol{X} \quad | \quad \langle \boldsymbol{X}, \boldsymbol{X} \rangle = E \]$,

is the spherical manifold of radius $E$.

### Calling the optimiser

Given a (python) list of (numpy) vectors $X_i$

$X = \[X_0,X_1, ..., X_n\]$

a corresponding list of constraint amplitudes (float) $E_i$

$E = \[E_0, E_1, ...., E_n\]$

the accompanying routines 

`f,Grad_f,Inner_Product` 

which calculate the object-function $J(\boldsymbol{X})$, its Euclidean gradient $\nabla J(\boldsymbol{X})$ and the inner-product $\langle \boldsymbol{f}, \boldsymbol{g} \rangle$ respectively. Calling the routine

`RESIDUAL, FUNCT, X_opt = Optimise_On_Multi_Sphere(X,E,f,Grad_f,Inner_Product,args_f,args_IP)`

returns the optimal vector `X_opt` (list of numpy vectors) the residual errors `RESIDUAL` (list of floats) and the objective function evaluations `FUNCT` (list of floats) during the iterative optimisation procedure. The arguments `args_f = (), args_IP=()` are tuples (or dictionaries if `kwargs_f = (), kwargs_IP=()`) which supply the necessary arguments to `f,Grad_f` and `Inner_Product` respectively. 

In addition to the above the remaining *optional arguments*, whose default values are

`err_tol = 1e-06, max_iters = 200, alpha_k = 1., LS = 'LS_wolfe', CG = True, callback=None`

specifiy the termination conditions `err_tol,max_iters`, the maximum/initial step-size `alpha_k`, the [scipy line-search routine](https://github.com/scipy/scipy/blob/v1.9.0/scipy/optimize/_linesearch.py#L181-L313) ('Armijo' or 'LS_wolfe'), the gradient descent routine 'CG=True' (use a conjuagte-gradient update) or 'CG=False' (use a steepest-descent update) and provide the utility of a callback (function/callable) which takes the current iteration $k$ as its sole argument and allows the user to save or perform calculations on the current iterate's information.

### Verifying the gradient

A routine for checking the gradient  

`Adjoint_Gradient_Test(X,dX,f,Grad_f,Inner_Product,args_f,args_IP, epsilon = 1e-04)`

is also provided, where $\langle \boldsymbol{X}, \boldsymbol{X} \rangle = \langle \boldsymbol{dX}, \boldsymbol{dX} \rangle =1$ such that `epsilon` controls the relative size of the peturbation $\boldsymbol{dX}$. 

# Example problems

### Principle component analysis

Optimisation code to search the largest principle component of a symmetric matrix M. The parameter `DIM` controls the dimension of the random symmetric matrix generated. Executing this script with 

`python3 PCA_example.py`

finds the principle component, using steepest-descent (SD) and conjugate-gradient (CG) methods, plots the residual error of each and compares `X_opt` with the solution found using numpy's built-in eigen-vector solver.

The following examples require an Anaconda environment with the parallelised spectral code [Dedalus installed](https://dedalus-project.org). Having installed Dedalus and activated the relevant conda environment both examples can be ran by executing the following commands. In both examples the option

`Adjoint_type = "Discrete" or "Continuous"`

allows one to select the gradient approximation used, while uncommenting and running the function `Adjoint_Gradient_Test` allows one to test the quality of the gradient approximation used.

## Periodic Domains Fourier  

### Swift-Hohenberg equation

Optimisation code to search the *minimal perturbation* triggering transition to a localised state of the Swift-Hohenberg equation on a 1D periodic domain following [(D. Lecoanet & R.R. Kerswell, Phys. Rev. E., 2018)](https://link.aps.org/doi/10.1103/PhysRevE.97.012212). The default arguments contained in FWD_Solve_SH23.py are

`T=50, Npts = 256, dt = 0.1, M_0 = 0.0725`

corresponding to the optimisation time window, the number of Fourier modes, the time-step and applied perturbation amplitude. Executing this script with 

`mpiexec -np 1 python3 FWD_Solve_SH23.py && python3 plot_figure_SH23.py`

optimises the perturbation using 1 core and plots the resulting solution.

### Kinematic dynamo

Optimisation code to search the "*best*" kinematic dynamo in a triply periodic box following [(A.P. Willis, PRL, 2012)](https://doi.org/10.1103/PhysRevLett.109.251101). The default arguments contained in FWD_Solve_PBox_IND_MHD.py are

`Rm=1, T=1, Npts = 24, dt = 0.001, Noise = True`

corresponding to the magentic Reynolds number, optimisation time window, the number of Fourier modes, the time-step and wether to initialise the velocity field with noise (True) or the analytical solution (False) given in [(A.P. Willis, PRL, 2012)](https://doi.org/10.1103/PhysRevLett.109.251101). Executing this script with 

`mpiexec -np 4 python3 FWD_Solve_KDyn.py && python3 plot_figure_KDyn.py`

optimises both the velocity and magnetic fields using 4 cores and plots the resulting solution.


## Bounded Domains Chebyshev

### Swift-Hohenberg equation

Repeats the *Swift-Hohenberg equation* on a bounded domain by considering the Swift-Hohenberg equation discretised using a Chebyshev basis with bounary conditions not admitting a periodic solution. The default arguments contained in FWD_Solve_SH23.py are

`T=20, Npts = 256, dt = 0.01, M_0 = 0.0019`

corresponding to the optimisation time window, the number of Chebyshev polynomials, the time-step and applied perturbation amplitude.  Executing this script with 

`mpiexec -np 1 python3 FWD_Solve_SHB23.py && python3 plot_figure_SHB23.py`

optimises the perturbation using 1 core and plots the resulting solution.

### Optimal mixing

Optimisation code to search the optimal velocity perturbation in a 2D rectangular box following [(F. Marcotte & C.P. Caulfield, JFM, 2018)](https://doi.org/10.1017/jfm.2018.565). The default arguments contained in FWD_Solve_PBox_IND_MHD.py are

`Re=500,Pr=1,Ri=0.05,  T=5,E_0=0.02,  Nx,Nz = 256,128, dt=1e-03`

corresponding to the Reynolds,Prandtl and Richardson numbers, the optimisation time window and initial amplitude,  the number of Fourier modes,Chebyshev polynomials and finally the time-step (here increased to allow quicker execution on one's work station). Executing this script with 

`mpiexec -np 4 python3 FWD_Solve_Poiseuille.py && python3 plot_figure_Poiseuille.py`

optimises both the velocity and magnetic fields using 4 cores and plots the resulting solution.

## Citation

Please cite the following paper in any publication where you find the present codes useful:

Paul M. Mannix, Calum S. Skene, Didier Auroux & Florence Marcotte, Discrete adjoint-based control: A robust gradient descent procedure for optimisation with PDE and norm constraints, https://arxiv.org/abs/2210.17194

## Acknowledgements

P. M. M. and F. M. acknowledge support from the French program “T-ERC” from Agence Nationale de la Recherche (ANR) under grant agreement ANR-19-ERC7-0008). C. S. S. acknowledges partial support from a grant from the Simons Foundation (Grant No. 662962, GF). He would also like to acknowledge support of funding from the European Union Horizon 2020 research and innovation programme (grant agreement no. D5S-DLV-786780). This work was also supported by the French government, through the UCAJEDI Investments in the Future project managed by the ANR under grant agreement ANR-15-IDEX-0001.  This work was also supported by the French government, through the UCAJEDI Investments in the Future project managed by the ANR under grant agreement ANR-15-IDEX-0001. The authors are grateful to the OPAL infrastructure from Universit\'e C\^ote d’Azur, Universit\'e C\^ote d’Azur’s Center for High-Performance Computing computer facilities for providing resources and support.
