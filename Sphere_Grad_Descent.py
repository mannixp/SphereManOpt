import os
os.environ["OMP_NUM_THREADS"] = "1" # Improves performance apparently ????
from mpi4py import MPI

import numpy as np
import h5py,sys,copy
from warnings import warn

class LineSearchWarning(RuntimeWarning):
    pass;

# We acknowledge the scipy.optimize.line_search library for the
# modified
# LS_armijo_multiple, LS_wolfe_multiple
# unmodified 
# scalar_search_armijo, scalar_search_wolfe2, _cubicmin, _quadmin, _zoom
# functions present in this file.


###################################################################
class result():
    
    """
    class for result of optimize_rotation
    
    Inputs:
    components - integer the number of norm constraints 

    Returns:
    None
    """
    
    def __init__(self,components):
        
        self.N=components;
        self.X_opt=np.asarray([])
        
        self.Iterations=0
        self.Function_Evals=0
        self.Gradient_Evals=0
        
        self.Residual=[]
        self.Step_Size=[]
        self.Function_Value=[]     

    def __str__(self):

        error = [self.Residual[ii][self.Iterations-1] for ii in range(self.N) ];

        s= ( 'Optimize_rotation succeed \n'
             +'Total iterations     = '+str(self.Iterations)    +'\n'
             +'Function evaluations = '+str(self.Function_Evals)+'\n'
             +'Gradient evaluations = '+str(self.Gradient_Evals)+'\n'
             +'Residual error r_k   = '+str(error)              +'\n'
             +'Step size      α_k   = '+str(self.Step_Size[self.Iterations-1])     +'\n'
             +'J(X_opt)             = '+str(self.Function_Value[self.Iterations-1])+'\n'
             #+'  X_opt              = array('+str(self.X_opt)+')\n'
             );
        return s
##################################################################

#------------------------------------------------------------------------------
# Armijo line and scalar searches
#------------------------------------------------------------------------------

def LS_armijo_multiple(f, inner_prod, M_0, X_k, g_k, d_k,  old_fval, args_f=(), args_IP = (), kwargs_f = {}, kwargs_IP={}, alpha0=1.0, c1=1e-4):

    """
    Minimize over alpha, the function ``phi(α) = f( R_xk(α_k*d_k) )``,
    
    where X_k+1 = R_xk =  √M_0*(X_k + α_k*d_k)/|| X_k + α_k*d_k ||_2

    is the rectraction based update see:
    Nicolas Boumal. An introduction to optimization on smooth manifolds, 2020
    
    Parameters
    ----------
    f : callable f(x,*args_f,**kwargs_f)
        Objective function to be minimised
    inner_prod : callable IP(x,y,*args_IP,**kwargs_IP)
        Inner product
    M_0: list of floats
         Spherical manifold radius <X_0,X_0> = M_0     
    X_k : list of array_like
        Current point.
    g_k: list of array_like
        tangent gradient of current point    
    d_k : list of array_like
        **like** the Search direction.  
    old_fval : float
        Value of `f` at point `X_k`.
    args_f : tuple, optional
        Optional arguments to pass to J(X_k).
    args_IP : tuple, optional
        Optional arguments to pass to inner_prod <F,G>.
    c1 : float, optional
        Value to control stopping criterion.
    alpha0 : scalar, optional
        Value of `alpha` at start of the optimization.
    Returns
    -------
    alpha
    f_count
    f_val_at_alpha
    Notes
    -----
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57
    """

    X_k = np.atleast_1d(X_k)
    fc = [0]

    def phi(alpha1):
        
        fc[0] += 1;
        
        #apply norm constraints
        X_new = copy.deepcopy(X_k);
        for index,c_i in enumerate(M_0):
            X_new[index]  = Update_vector(X_k[index],alpha1,d_k[index],c_i,inner_prod,args_IP,kwargs_IP);
        return f( X_new, *args_f, **kwargs_f)

    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    # Compute the derivative w.r.t alpha at alpha=0        
    derphi0=0.    
    for index,_ in enumerate(M_0):
        derphi0 += inner_prod(g_k[index],d_k[index],*args_IP,**kwargs_IP);
    
    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
                                       alpha0=alpha0)
    return alpha, fc[0], phi1;

def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1.0, amin=1e-06):
    """Minimize over alpha, the function ``phi(alpha)``.
    Uses the interpolation algorithm (Armijo backtracking) as suggested by
    Wright and Nocedal in 'Numerical Optimization', 1999, pp. 56-57
    alpha > 0 is assumed to be a descent direction.
    Returns
    -------
    alpha
    phi1
    """
    phi_a0 = phi(alpha0)

    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0

    # Otherwise, compute the minimizer of a quadratic interpolant:

    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    if (phi_a1 <= phi0 + c1*alpha1*derphi0):
        return alpha1, phi_a1

    # Otherwise, loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.

    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + np.sqrt(abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1

#------------------------------------------------------------------------------
# Strong Wolfe line and scalar searches require
# 0 < c1 < c2 < 0.5 when using Fletcher-Reeves
# see H.Sato & T.Iwai, A New globally convergent Riemannian CG method, 2015
#------------------------------------------------------------------------------

def LS_wolfe_multiple(f, myfprime, inner_prod, M_0, X_k, g_k, d_k, old_fval=None, old_old_fval=None, args_f=(), args_IP = (), kwargs_f = {}, kwargs_IP={}, c1=1e-4, c2=0.4, amax=None, extra_condition=None, maxiter=10):
    
    """
    Find alpha that satisfies strong Wolfe conditions by ....

    Minimizing over alpha, the function ``phi(α) = f( R_xk(α_k*d_k) )``,
    
    where X_k+1 = R_xk =  √M_0*(X_k + α_k*d_k)/|| X_k + α_k*d_k ||_2

    is the rectraction based update see:
    Nicolas Boumal. An introduction to optimization on smooth manifolds, 2020

    Parameters
    ----------
    f : callable f(x,*args_f,**kwargs_f)
        Objective function.
    myfprime : callable f'(x,*args_f,**kwargs_f)
        Objective function gradient.
    inner_prod : callable IP(x,y,*args_IP,**kwargs_IP)
        Inner product
    M_0 : list of float(s)
        Constraint radii       
    xk : list of ndarray(s)
        Starting point.
    pk : list of ndarray(s)
        Search direction.
    gfk : list of ndarray(s), optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk.
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.
    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.
    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.
    """

    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(alpha1):
        fc[0] += 1;

        #apply norm constraints
        X_new = copy.deepcopy(X_k);
        for index,c_i in enumerate(M_0):
            X_new[index]  = Update_vector(X_k[index],alpha1,d_k[index],c_i,inner_prod,args_IP,kwargs_IP);
        
        return f( X_new, *args_f,**kwargs_f)    


    fprime = myfprime
        
    def derphi(alpha1):
        gc[0] += 1

        X_kp1 = copy.deepcopy(X_k);
        g_kp1 = copy.deepcopy(g_k);
        Tdkm1 = copy.deepcopy(g_k);
        derphi1=0.
        
        #apply norm constraints
        for index,c_i in enumerate(M_0):
            X_kp1[index]  = Update_vector(X_k[index],alpha1,d_k[index],c_i,inner_prod,args_IP,kwargs_IP); 
        
        # Calculate the Euclidean gradient
        Nab_Jkp1 = fprime(X_kp1,*args_f,**kwargs_f)
        
        # Compute the tangent gradient and perform vector transport of d_k 
        for index,_ in enumerate(M_0):
            g_kp1[index] = tangent_vector(X_kp1[index],Nab_Jkp1[index],inner_prod,args_IP,kwargs_IP)
            Tdkm1[index] = transport_vector(X_kp1[index],d_k[index],inner_prod,args_IP,kwargs_IP)
            derphi1     += inner_prod(g_kp1[index],Tdkm1[index],*args_IP,**kwargs_IP); # Compute the derivate w.r.t alpha1
        
        # Store current tangent gradient for later use
        gval[0]  = g_kp1;
        gval_alpha[0] = alpha1;

        return derphi1;    

    # Compute the derivative w.r.t alpha at alpha=0    
    derphi0=0.    
    for index,_ in enumerate(M_0):    
        derphi0 += inner_prod(g_k[index],d_k[index],*args_IP,**kwargs_IP);    


    extra_condition2 = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
            phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
            extra_condition2, maxiter=maxiter)

    if derphi_star is None:
        
        warn('The line search algorithm did not converge', LineSearchWarning)
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star

def scalar_search_wolfe2(phi, derphi, phi0=None,old_phi0=None, derphi0=None,c1=1e-4, c2=0.4, amax=None, extra_condition=None, maxiter=10):
    """
    Find alpha that satisfies strong Wolfe conditions.
    alpha > 0 is assumed to be a descent direction.
    Parameters
    ----------
    phi : callable phi(alpha)
        Objective scalar function.
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0.
    old_phi0 : float, optional
        Value of phi at previous point.
    derphi0 : float, optional
        Value of derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size.
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, phi_value)``
        returning a boolean. The line search accepts the value
        of ``alpha`` only if this callable returns ``True``.
        If the callable returns ``False`` for the step length,
        the algorithm will continue with new iterates.
        The callable is only called for iterates satisfying
        the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.
    Returns
    -------
    alpha_star : float or None
        Best alpha, or None if the line search algorithm did not converge.
    phi_star : float
        phi at alpha_star.
    phi0 : float
        phi at 0.
    derphi_star : float or None
        derphi at alpha_star, or None if the line search algorithm
        did not converge.
    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.
    """

    if phi0 is None:
        phi0 = phi(0.)

    if derphi0 is None:
        derphi0 = derphi(0.)

    alpha0 = 0
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0

    if alpha1 < 0:
        alpha1 = 1.0

    if amax is not None:
        alpha1 = min(alpha1, amax)

    phi_a1 = phi(alpha1)
    #derphi_a1 = derphi(alpha1) evaluated below

    phi_a0 = phi0
    derphi_a0 = derphi0

    if extra_condition is None:
        extra_condition = lambda alpha, phi: True

    for i in range(maxiter):
        if alpha1 == 0 or (amax is not None and alpha0 == amax):
            # alpha1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            alpha_star = None
            phi_star = phi0
            phi0 = old_phi0
            derphi_star = None

            if alpha1 == 0:
                msg = 'Rounding errors prevent the line search from converging'
            else:
                msg = "The line search algorithm could not find a solution " + \
                      "less than or equal to amax: %s" % amax

            warn(msg, LineSearchWarning)
            break

        not_first_iteration = i > 0
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or \
           ((phi_a1 >= phi_a0) and not_first_iteration):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break

        if (derphi_a1 >= 0):
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1

    else:
        # stopping test maxiter reached
        alpha_star = alpha1
        phi_star = phi_a1
        derphi_star = None
        warn('The line search algorithm did not converge', LineSearchWarning)

    return alpha_star, phi_star, phi0, derphi_star

def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
    If no minimizer can be found, return None.
    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D

    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin

def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.
    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin

def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    """
    Zoom stage of approximate linesearch satisfying strong Wolfe conditions.

    Part of the optimization algorithm in `scalar_search_wolfe2`.

    Notes
    -----
    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 61.
    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2*derphi0 and extra_condition(a_j, phi_aj):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_lo
                phi_hi = phi_lo
            else:
                phi_rec = phi_lo
                a_rec = a_lo
            a_lo = a_j
            phi_lo = phi_aj
            derphi_lo = derphi_aj
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star

#------------------------------------------------------------------------------
# Pure-Python Spherical Manifold Optimisation
#------------------------------------------------------------------------------

# These three functions are specific to the update formula/Retraction

# X_k+1 = √M_0*(X_k + α_k*d_k)/|| X_k + α_k*d_k ||_2

# args, kwargs should be based as args=(a,b,c), kwargs={'a':1,'b':2,'c':3} and only unpacked when calling inner_prod(f,g,*args,**kwargs)

def transport_vector(X_k,dkm1,inner_prod,args_IP=(),kwargs_IP={}):

	"""
	Return the vector transport for an arbitrary inner product

	Inputs:
	X_k    	   - parameter vector
	dkm1 	   - previous search direction
	inner_prod - callable function: takes args_IP = (),kwargs_IP = {}

	Returns:
	T(dkm1)    - vector transported to X_k tangent plane

	"""

	L2   = np.sqrt( inner_prod(X_k,X_k ,*args_IP,**kwargs_IP) );

	return dkm1 - ( inner_prod(X_k,dkm1,*args_IP,**kwargs_IP)/(L2**2) )*X_k; #*(np.sqrt(M_0)/L2); #not needed due to the fact = 

def tangent_vector(X_k,Nab_Jk,inner_prod,args_IP=(),kwargs_IP={}):

	"""
	Return the tangent vector for an arbitrary inner product

	Inputs:
	X_k    	   - parameter vector
	Nab_Jk 	   - Euclidean vector
	inner_prod - callable function: takes args_IP = (),kwargs_IP = {}

	Returns:
	gk - tangent vector

	"""

	return Nab_Jk - ( inner_prod(X_k,Nab_Jk,*args_IP,**kwargs_IP)/inner_prod(X_k,X_k,*args_IP,**kwargs_IP) )*X_k;

def Update_vector(X_k,alpha_k,d_k,M_0,inner_prod,args_IP=(),kwargs_IP={}):

    """

    Update the parameter vector in the search direction alpha_k*d_k

    Inputs:
    X_k    - vector parameter 
    alpha_k- float  step-size
    d_k    - vector search direction
    M_0    - float spherical manifold size < X_0,X_0 > = M_0
    inner_prod - callable function: takes args_IP = (),kwargs_IP = {}

    Returns:

    X_k+1  - vector new parameter vector

    Notes this is the rectraction based update see:
    Nicolas Boumal. An introduction to optimization on smooth manifolds, 2020

    """

    #Xn = np.sqrt( M_0 );
    #dn = np.sqrt( inner_prod(d_k,d_k,*args) );
    #return np.cos(alpha_k*dn)*X_k + np.sin(alpha_k*dn)*d_k*(Xn/dn); 

    f    = X_k + alpha_k*d_k;
    L2_f = inner_prod(f,f,*args_IP,**kwargs_IP);

    return f*np.sqrt(M_0/L2_f)

def Optimise_On_Multi_Sphere(X_0, M_0, f, myfprime, inner_prod, args_f = (), args_IP=(), kwargs_f = {}, kwargs_IP={}, err_tol = 1e-06, max_iters = 200, alpha_k = 1., LS = 'LS_wolfe', CG = True, callback=None, verbose=True):
	
    """
    Function to perform the minimisation of J(X) via 
    gradient based descent  Grad_f(X) on a spherical 
    manifold <X,X> = M_0.

    Inputs:
    X_0        - list of initial parameter vector guess i.e. [x_0,x_1, .... , x_N]
    M_0        - list of spherical manifold radius of each vector i.e. [c_0,c_1, .... , c_N]
    f 	   	   - callable returns      J(X_k) takes unpacked *args_f, **kwargs_f
    Grad_f     - callable returns Grad_J(X_k) takes unpacked *args_f, **kwargs_f
    inner_prod - callable returns <F,G>		  takes unpacked *args_IP,**kwargs_IP

    Returns:

    RESIDUAL - vector of error at each iterations
    FUNCT    - vector of function of evaluations
    X_opt    - list of optimal vectors

    """

    if LS == 'LS_wolfe': 
        LS = LS_wolfe_multiple;
    elif LS == 'LS_armijo':
        LS = LS_armijo_multiple;

    error = np.ones(len(M_0)); 
    func_evals=0;
    grad_evals=0;
    alpha_max = alpha_k;

    RESIDUAL = [];
    for val in error:
        RESIDUAL.append([]);

    # Initialise the class for data handling
    R = result(len(M_0))
    f_txt = open("optimize_result.txt", "a")

    # Normalise X_k so that <X,X> = M_0
    J_k_old = None;
    X_k = [ x_i*np.sqrt( c_i/inner_prod(x_i,x_i,*args_IP,**kwargs_IP) ) for x_i,c_i in zip(X_0,M_0) ];
    J_k = f(X_k,*args_f,**kwargs_f); func_evals+=1;

    while (max(error) > err_tol) and (R.Iterations < max_iters):

        # Reuse the gradient computed if using a strong wolfe line-search
        if (LS == LS_wolfe_multiple) and (R.Iterations > 1):
            g_k = derphi_star;
        else:
            Nab_Jk = myfprime(X_k,*args_f,**kwargs_f);
            g_k    = [tangent_vector(u,du,inner_prod,args_IP,kwargs_IP) for u,du in zip(X_k,Nab_Jk)];
            grad_evals +=1;

        
        # Select a search direction d_k via 
        # SD-steepest descent or CG - conjugate gradient
        if (R.Iterations > 1) and (CG==True):
            
            # Conjuagte-gradient
            beta_k_FR = 0.; 
            beta_k_PR = 0.;
            Tg_km1    = copy.deepcopy(g_k);
            Td_km1    = copy.deepcopy(g_k);

            for ii,_ in enumerate(g_k):
                
                beta_k_FR += inner_prod(g_k[ii],g_k[ii],*args_IP,**kwargs_IP)/inner_prod(g_km1[ii],g_km1[ii],*args_IP,**kwargs_IP)
                
                Tg_km1[ii] = transport_vector(X_k[ii],g_km1[ii],inner_prod,args_IP,kwargs_IP);
                beta_k_PR += ( inner_prod(g_k[ii],g_k[ii],*args_IP,**kwargs_IP) - inner_prod(g_k[ii],Tg_km1[ii],*args_IP,**kwargs_IP) )/inner_prod(g_km1[ii],g_km1[ii],*args_IP,**kwargs_IP);

                Td_km1[ii] = transport_vector(X_k[ii],d_k[ii],inner_prod,args_IP,kwargs_IP)

            # Use the Fletcher-Reeves + Polak Rib\`ere update 
            # of H. Sato Riemannian conjugate gradient methods 2021
            # to select the parameter ß_k
            beta_k = max(0.,min(beta_k_FR,beta_k_PR));
            
            d_k = [-1.*g_ki + beta_k*T_ki for g_ki,T_ki in zip(g_k,Td_km1)];
        
        else:   
            # Gradient descent    
            d_k = [-1.*g_i for g_i in g_k];


        # Perform a line-search for the step-size α_k to ensure descent 
        if (R.Iterations == 0) or (LS == LS_armijo_multiple):
            alpha_k, f_evals, J_k = LS_armijo_multiple(f, inner_prod, M_0, X_k, g_k, d_k,  J_k,  args_f, args_IP, kwargs_f, kwargs_IP, alpha0 = alpha_k)
            func_evals+=f_evals;
        else:
            alpha_k, f_evals, g_evals, J_k, J_k_old, derphi_star = LS(f, myfprime, inner_prod, M_0, X_k, g_k, d_k, J_k,J_k_old, args_f, args_IP, kwargs_f, kwargs_IP, amax=alpha_max);
            grad_evals+=g_evals;
            func_evals+=f_evals;

        # Update the parameter vector - applying the norm constraints    
        for index,c_i in enumerate(M_0):
            
            if alpha_k == None:
                print("\n Couldn't find a descent direction .... Terminating \n");      
                return R.Residual,R.Function_Value,R.X_opt;
            else:
                X_k[index]   = Update_vector(X_k[index],alpha_k,d_k[index],c_i,inner_prod,args_IP,kwargs_IP);
                error[index] = inner_prod(g_k[index],g_k[index],*args_IP,**kwargs_IP)**0.5;


        # Update the optimisation state        
        R.X_opt = X_k;

        R.Iterations+=1;
        R.Function_Evals+=func_evals; 
        R.Gradient_Evals+=grad_evals; 
        
        for ii,_ in enumerate(error):
            RESIDUAL[ii].append(error[ii]);

        R.Residual=RESIDUAL;
        R.Step_Size.append(alpha_k)
        R.Function_Value.append(-1.*J_k)

        g_km1 = copy.deepcopy(g_k);
        func_evals=0;
        grad_evals=0;

        # Save the optimisation state   
        if callback != None:
            callback(R.Iterations);
        
        try:
            if MPI.COMM_WORLD.rank == 0:
                f_h5 = h5py.File('DAL_PROGRESS.h5', 'w');
                for item in vars(R).items():
                    f_h5.create_dataset(item[0], data = item[1])
                f_h5.close()
            
            '''
            # Save the different errors 
            DAL_file = h5py.File('DAL_PROGRESS.h5', 'w')
            
            # Problem Params
            DAL_file['RESIDUAL'] = RESIDUAL;
            DAL_file['FUNCT']    = FUNCT;
            DAL_file['X_opt']    = X_k
            
            DAL_file.close();
            '''    

        except:
            pass; 
              
        # Print out the optimisation status    
        if verbose : print(R,flush=True);                       
        f_txt.write(str(R))
        f_txt.write('\n')
        f_txt.flush()

    f_txt.close()    
    return R.Residual,R.Function_Value,R.X_opt;

def plot_optimisation(THETA,FUNCT):
	
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8,6));
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis


    # Plot J(X)
    x = np.arange(0,len(FUNCT),1);
    #ax1.semilogy(x[0:-2],abs(FUNCT - np.ones(len(x))*FUNCT[-1])[0:-2],color='tab:red',linewidth=1.5, markersize=3, linestyle=':');
    ax1.plot(x,FUNCT,color='tab:red',linewidth=3, markersize=3, linestyle=':');

    #f = abs(np.roll(FUNCT,-1) - np.ones(len(x))*FUNCT[-1])/abs(FUNCT - np.ones(len(x))*FUNCT[-1]);print('r^2 = ',np.mean(f[5:-5]),'\n')

    # Unpack r_k then plot
    linestyles = ['-.','-'];
    for i,r_k in enumerate(THETA): 
        x = np.arange(0,len(r_k),1);
        ax2.semilogy(x,r_k,color='tab:blue',linewidth=3, markersize=3,label=r"c_%i"%i, linestyle=linestyles[i])
        
    ax1.tick_params(axis='y', labelcolor='tab:red',labelsize=26)
    ax1.tick_params(axis='x',labelsize=26)
    ax1.set_ylabel(r'$|\hat{J}_k(\hat{X}_k)|$',color='tab:red',fontsize=26)
    ax1.set_xlabel(r'Iteration $k$',fontsize=26)
    ax1.set_xlim([0,np.max(x)])
    #ax1.set_ylim([1e-12,1e02])  
    ax1.set_xticks(ax1.get_xticks()[::2])
    ax1.set_yticks(ax1.get_yticks()[::2])

    ax2.tick_params(axis='y', labelcolor='tab:blue',labelsize=26)
    #ax2.set_ylabel(r'$r_k$',color='tab:blue',fontsize=26)
    ax2.set_yticks(ax2.get_yticks()[::2])
    ax2.set_ylim([1e-06,1])  
    ax2.legend(fontsize=18)

    plt.grid()
    plt.tight_layout(pad=1, w_pad=1.5)
    fig.savefig("Mix_DISC_SD_A.pdf",dpi=1200);
    plt.show();

    return None;

