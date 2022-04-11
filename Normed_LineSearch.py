import os;
os.environ["OMP_NUM_THREADS"] = "1" # Improves performance apparently ????
import numpy as np

from warnings       import warn
#from scipy.optimize import _minpack2 as minpack2
from scipy.optimize import minpack2
#from . import _minpack as minpack

from FWD_Solve_PBox_IND_MHD import Inner_Prod,Inner_Prod_3

class LineSearchWarning(RuntimeWarning):
    pass

##########################################################################
# Enforece the normalization constraint - rotation or tangent specified within
# Applies within that search direction p_k =  - dJ/dw_k
##########################################################################
def Norm_constraint(domain, w_k, dJ_k,   alpha,M_0):

    """
    Enforce the normalisation constraint:

    ||w||^2 = <w,w> = M_0   or      ||w||_2 = M_0^(1/2),
    
    following the rotation or tangent method => Douglas, S. IEEE 1998.

    returns the updated vector w_k and the tangent vector hg_k

    Notation:

    - B_0 = w_k
    - d(B_0)/dr = wp_k
    - (-1)*Nabla J(B_0) = g_k, assuming -1 => gradient descent
    - (-1)*d/dr( Nabla J(B_0) )= d/dr( g_k ), assuming -1 => gradient descent
    - v_k = d||w||_2/dw = w_k 
    
    Method: 

    Update step
    - w_{k+1} = cos(alpha)*w_k + sin(alpha)*u_k, 
    - wp_{k+1} = cos(alpha)*wp_k + sin(alpha)*(du_k/dr), 

    where hg_k = g_k - ( <w_k,g_k>/<w_k,w_k> )*w_k, # Tangent vector
    
    and    u_k = sqrt(<w,w>/<hg_k,hg_k>)*hg_k,      # Normalised Tangetn vector
    """
    '''
    import logging
    root = logging.root
    for h in root.handlers:
        h.setLevel("INFO");
        #h.setLevel("DEBUG")
    logger = logging.getLogger(__name__)

    # 1) Catch zero step-sizes/gradients before proceeding
    if (np.linalg.norm(dJ_k,2) == 0.):
        logger.debug("Zero gradient - no update \n")
        return w_k, None, None;
    '''  

    # Check w_k norm
    #logger.debug('Norm ||w_k||^2 = %e, M_0 = %e'%( Inner_Prod(domain,w_k,w_k) ,M_0) );

    # A. Compute the tangent vector # Assuming -1 => gradient descent
    g_k  = (-1.)*dJ_k; 
    #gp_k = (-1.)*dJp_k;

    # Same constant for both as d/dr(C1) = 0 
    C1 = Inner_Prod_3(domain,w_k, g_k)/Inner_Prod_3(domain,w_k,w_k);

    hg_k  =  g_k  -  w_k*C1; # Should M_0 be replaced by <w_k,w_k> = ||w_k||^2_2
    #hgp_k =  gp_k - wp_k*C1;

    # B. Normalise it!

    # Same constant for both as d/dr(C2) = 0 
    C2 = np.sqrt(M_0/Inner_Prod_3(domain,hg_k,hg_k));
    u_k  = C2*hg_k
    #up_k = C2*hgp_k;
    #logger.debug('Rotation Method C1 = %e = 0?, C2 = %e'%(C1,C2) );
    #logger.debug('Norm ||u_k||^2 = %e, M_0 = %e'%(Inner_Prod(domain,u_k,u_k),M_0) );

    # C. Check method works
    X = np.cos(alpha)*w_k + np.sin(alpha)*u_k; 
    
    #N_x = Inner_Prod(domain,X,X)
    #logger.debug('Constraint via Rotation method ||Bx0||^2 = %e, M_0 = %e'%(N_x,M_0) );

    # D. Supply dX/dr also
    #dXdr = np.cos(alpha)*wp_k + np.sin(alpha)*up_k;

    return X, -1.*hg_k; #, -1.*hgp_k;

#------------------------------------------------------------------------------
# Armijo line and scalar searches 
# Ammended to included norm constraint ||xk + s*pk ||_2 = M_0^(1/2)
#------------------------------------------------------------------------------
def line_search_armijo(f, con , Constraints,  xk,pk,gfk,  old_fval, args=(), c1=1e-4, alpha0=1):

    """Minimize over alpha, the function ``phi(alpha) = f(xk+alpha pk)``.
    Parameters
    ----------
    f : callable
        Function to be minimized.
    xk : array_like
        Current point.
    xpk: array_like
        Derivative d/dr(xk) -- of current point    
    pk : array_like
        **like** the Search direction.
        x_k+1 = x_k*cos(alpha) + d*sin(alpha);
        x_k+1 = x_k + [x_k*(cos(alpha) -1) + d*sin(alpha)];
        x_k+1 = x_k +  p_k(alpha);
        Currently using p_k ~ alpha*d;
    dr_pk : array_like
        Partiral derivative w.r.t to radial direction r of search direction pk at point `xk`.     
    gfk : array_like
        Gradient of `f` at point `xk`.   
    old_fval : float
        Value of `f` at point `xk`.
    args : tuple, optional
        Optional arguments.
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
    domain = args[0];

    xk = np.atleast_1d(xk)
    fc = [0]

    #'''
    # Modified
    def phi(alpha1):
        fc[0] += 1;
        
        # Split the vector + apply norm constraints
        XK  = np.split(xk,2); # Must not update xk nor pk here!!, ensure this does not update the reference all the time
        PK  = np.split(pk,2);
        EQs = Constraints;
        
        if   con == 0:
            
            XKP1 = Norm_constraint(domain, XK[con],PK[con], alpha1, EQs[con])[0];    
            return f( np.concatenate((XKP1,XK[1])) , *args)
        
        elif con == 1:

            XKP1 = Norm_constraint(domain, XK[con],PK[con], alpha1, EQs[con])[0];    
            return f( np.concatenate((XK[0],XKP1)) , *args)
        else:
            
            for i in range(len(EQs)):
                XK[i] = Norm_constraint(domain, XK[i],PK[i], alpha1, EQs[i])[0];    

            return f( np.concatenate((XK[0],XK[1])) , *args)  

    '''

    # Original
    def phi(alpha1):
        fc[0] += 1;
        
        # Split the vector + apply norm constraints
        XK  = np.split(xk,2); # Must not update xk nor pk here!!, ensure this does not update the reference all the time
        PK  = np.split(pk,2);
        EQs = Constraints;
        for i in range(len(EQs)):
            XK[i] = Norm_constraint(domain, XK[i],PK[i], alpha1, EQs[i])[0];    

        return f( np.concatenate((XK[0],XK[1])) , *args)    
    '''
     
    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    # Modified 
    if  (con == 0) or (con == 1):
        hgk = (-1.)*np.split(pk, 2)[con];
        gfk =       np.split(gfk,2)[con];
        derphi0 = Inner_Prod_3(domain,gfk,hgk);
    else:
        hgk = (-1.)*pk;
        derphi0 = Inner_Prod(domain,gfk,hgk);

    # Original
    #hgk = (-1.)*pk;
    #derphi0 = Inner_Prod(domain,gfk,hgk);

    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
                                       alpha0=alpha0)
    return alpha, fc[0], phi1

def line_search_armijo_old_ver(f, Constraints,  xk,pk,gfk,  old_fval, args=(), c1=1e-4, alpha0=1):

    """Minimize over alpha, the function ``phi(alpha) = f(xk+alpha pk)``.
    Parameters
    ----------
    f : callable
        Function to be minimized.
    xk : array_like
        Current point.
    xpk: array_like
        Derivative d/dr(xk) -- of current point    
    pk : array_like
        **like** the Search direction.
        x_k+1 = x_k*cos(alpha) + d*sin(alpha);
        x_k+1 = x_k + [x_k*(cos(alpha) -1) + d*sin(alpha)];
        x_k+1 = x_k +  p_k(alpha);
        Currently using p_k ~ alpha*d;
    dr_pk : array_like
        Partiral derivative w.r.t to radial direction r of search direction pk at point `xk`.     
    gfk : array_like
        Gradient of `f` at point `xk`.   
    old_fval : float
        Value of `f` at point `xk`.
    args : tuple, optional
        Optional arguments.
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
    domain = args[0];

    xk = np.atleast_1d(xk)
    fc = [0]

    '''
    # Modified
    def phi(alpha1):
        fc[0] += 1;
        
        # Split the vector + apply norm constraints
        XK  = np.split(xk,2); # Must not update xk nor pk here!!, ensure this does not update the reference all the time
        PK  = np.split(pk,2);
        EQs = Constraints;
        
        if   con == 0:
            
            XKP1 = Norm_constraint(domain, XK[con],PK[con], alpha1, EQs[con])[0];    
            return f( np.concatenate((XKP1,XK[1])) , *args)
        
        elif con == 1:

            XKP1 = Norm_constraint(domain, XK[con],PK[con], alpha1, EQs[con])[0];    
            return f( np.concatenate((XK[0],XKP1)) , *args)
        else:
            
            for i in range(len(EQs)):
                XK[i] = Norm_constraint(domain, XK[i],PK[i], alpha1, EQs[i])[0];    

            return f( np.concatenate((XK[0],XK[1])) , *args)  

    '''

    # Original
    def phi(alpha1):
        fc[0] += 1;
        
        # Split the vector + apply norm constraints
        XK  = np.split(xk,2); # Must not update xk nor pk here!!, ensure this does not update the reference all the time
        PK  = np.split(pk,2);
        EQs = Constraints;
        for i in range(len(EQs)):
            XK[i] = Norm_constraint(domain, XK[i],PK[i], alpha1, EQs[i])[0];    

        return f( np.concatenate((XK[0],XK[1])) , *args)    
    #'''
     
    if old_fval is None:
        phi0 = phi(0.)
    else:
        phi0 = old_fval  # compute f(xk) -- done in past loop

    '''    
    # Modified 
    if  (con == 0) or (con == 1):
        hgk = (-1.)*np.split(pk, 2)[con];
        gfk =       np.split(gfk,2)[con];
        derphi0 = Inner_Prod_3(domain,gfk,hgk);
    else:
        hgk = (-1.)*pk;
        derphi0 = Inner_Prod(domain,gfk,hgk);
    '''
        
    # Original
    hgk = (-1.)*pk;
    derphi0 = Inner_Prod(domain,gfk,hgk);

    alpha, phi1 = scalar_search_armijo(phi, phi0, derphi0, c1=c1,
                                       alpha0=alpha0)
    return alpha, fc[0], phi1

def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=1e-06):
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

    '''
    # 1) Compute quadratic minimiser
    alpha_st = _quadmin(0., phi0, derphi0, alpha0, phi_a0)

    # 2) Check the step-length is in the feasible region
    if (amin < alpha_st < alpha0) and (alpha_st < np.pi):
        #print("\n STEP SIZE In RANGE (amin,amax)")
        
        # 3) check that the minimiser is indeed better 
        phi_st = phi(alpha_st);
        #print("C1 phi_n+1 = %e < phi_n   = %e \n"%(phi_st,phi_a0) );
        #print("C2 phi_n+1 = %e < wolfe I = %e \n"%(phi_st,phi0 + c1*alpha_st*derphi0) );
        if (phi_st < phi_a0) and (phi_st <= phi0 + c1*alpha_st*derphi0):
            print("C1 Quad min accepted: phi_n+1 = %e < phi_n = %e "%(phi_st,phi_a0) );
            print("C2 phi_n+1 = %e < wolfe I = %e             \n"%(phi_st,phi0 + c1*alpha_st*derphi0) );
            return alpha_st,phi_st;
        else:
            print(" \n")    
    '''    

    # 4) Check Wolfe I condition
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
# Minpack's Wolfe line and scalar searches
# 
# MINPACK-1 Project. June 1983.
# Argonne National Laboratory.
# Jorge J. More' and David J. Thuente.
#
# MINPACK-2 Project. November 1993.
# Argonne National Laboratory and University of Minnesota.
# Brett M. Averick, Richard G. Carter, and Jorge J. More'.
#------------------------------------------------------------------------------

def line_search_wolfe1(f, fprime, Constraints, xk, pk, gfk=None,old_fval=None, old_old_fval=None,args=(),c1=1e-4,c2=0.9,amax=50,amin=1e-6,xtol=1e-14):

	"""
	As `scalar_search_wolfe1` but do a line search to direction `pk`
	Parameters
	----------
	f : callable
		Function `f(x)`
	fprime : callable
		Gradient of `f`
	xk : array_like
		Current point
	pk : array_like
		Search direction
	
	gfk : array_like, optional
		Gradient of `f` at point `xk`
	old_fval : float, optional
		Value of `f` at point `xk`
	old_old_fval : float, optional
		Value of `f` at point preceding `xk`
	
	The rest of the parameters are the same as for `scalar_search_wolfe1`.
	
	Returns
	-------
	stp, f_count, g_count, fval, old_fval
		As in `line_search_wolfe1`
	gval : array
		Gradient of `f` at the final point
	
	"""

	domain = args[0];

	if gfk is None:
		gfk = fprime(xk, *args)

	# Declares these as arrays, to allow mutable/pass by reference to phi,derphi    
	gval = [gfk]
	gc = [0]
	fc = [0]

	def phi(s):
		fc[0] += 1; # Increment counter

		# Split the vector + apply norm constraints
		XK  = np.split(xk,2); # Must not update xk nor pk here!!, ensure this does not update the reference all the time
		PK  = np.split(pk,2);
		for i in range(len(Constraints)):
			XK[i] = Norm_constraint(domain, XK[i],PK[i], s, Constraints[i])[0];

		XK_new = np.concatenate((XK[0],XK[1]));   
		return f( XK_new, *args)

	def derphi(s):
		gc[0] += 1; # Increment counter

		# Split the vector + apply norm constraints
		XK  = np.split(xk,2); # Must not update xk nor pk here!!, ensure this does not update the reference all the time
		PK  = np.split(pk,2);
		for i in range(len(Constraints)):
			XK[i] = Norm_constraint(domain, XK[i],PK[i], s, Constraints[i])[0];

		XK_new = np.concatenate((XK[0],XK[1]));    
		gval[0] = fprime( XK_new, *args)

		#return np.dot(gval[0], pk); # Use correct inner product
		return (-1.)*Inner_Prod(domain,pk,gval[0])

	#derphi0 = np.dot(gfk, pk); # Use correct inner product
	derphi0 = (-1.)*Inner_Prod(domain,pk,gfk);

	stp, fval, old_fval = scalar_search_wolfe1(phi, derphi, old_fval, old_old_fval, derphi0,c1=c1, c2=c2, amax=amax, amin=amin, xtol=xtol)

	return stp, fc[0], gc[0], fval, old_fval, gval[0];

def scalar_search_wolfe1(phi, derphi, phi0=None, old_phi0=None, derphi0=None,				 			 c1=1e-4,c2=0.9,amax=50,amin=1e-6,xtol=1e-14):
    """
    Scalar function search for alpha that satisfies strong Wolfe conditions
    alpha > 0 is assumed to be a descent direction.
    Parameters
    ----------
    phi : callable phi(alpha)
        Function at point `alpha`
    derphi : callable phi'(alpha)
        Objective function derivative. Returns a scalar.
    phi0 : float, optional
        Value of phi at 0
    old_phi0 : float, optional
        Value of phi at previous point
    derphi0 : float, optional
        Value derphi at 0
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax, amin : float, optional
        Maximum and minimum step size
    xtol : float, optional
        Relative tolerance for an acceptable step.
    Returns
    -------
    alpha : float
        Step size, or None if no suitable step was found
    phi : float
        Value of `phi` at the new point `alpha`
    phi0 : float
        Value of `phi` at `alpha=0`
    Notes
    -----
    Uses routine DCSRCH from MINPACK.
    """

    if phi0 is None:
        phi0 = phi(0.)
    if derphi0 is None:
        derphi0 = derphi(0.)

    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        if alpha1 < 0:
            alpha1 = 1.0
    else:
        alpha1 = 1.0

    phi1 = phi0
    derphi1 = derphi0
    isave = np.zeros((2,), np.intc)
    dsave = np.zeros((13,), float)
    task = b'START'

    maxiter = 100
    for i in range(maxiter):
        stp, phi1, derphi1, task = minpack2.dcsrch(alpha1, phi1, derphi1,
                                                   c1, c2, xtol, task,
                                                   amin, amax, isave, dsave)
        print("stp   = ",stp);
        print("alpha1= ",alpha1);
        print("Stp error = ",abs(stp-alpha1)/stp);
        print("task ",task)
        if task[:2] == b'FG':
            alpha1 = stp
            phi1 = phi(stp)
            derphi1 = derphi(stp)
        else:
            break
    else:
        # maxiter reached, the line search did not converge
        stp = None

    if task[:5] == b'ERROR' or task[:4] == b'WARN':
        stp = None  # failed

    print("stp  = ",stp );
    print("phi  = ",phi );
    print("phi0 = ",phi0);    
    return stp, phi1, phi0;


#------------------------------------------------------------------------------
# Minpack's Wolfe line and scalar searches
#------------------------------------------------------------------------------


def line_search_wolfe2(f, myfprime, Constraints, xk, pk, gfk=None, old_fval=None,old_old_fval=None, args=(), c1=1e-4, c2=0.9, amax=None, extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.
    Parameters
    ----------
    f : callable f(x,*args)
    Objective function.
    myfprime : callable f'(x,*args)
    Objective function gradient.
    xk : ndarray
    Starting point.
    pk : ndarray
    Search direction.
    gfk : ndarray, optional
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
    Examples
    --------
    >>> from scipy.optimize import line_search
    A objective function and its gradient are defined.
    >>> def obj_func(x):
    ...     return (x[0])**2+(x[1])**2
    >>> def obj_grad(x):
    ...     return [2*x[0], 2*x[1]]
    We can find alpha that satisfies strong Wolfe conditions.
    >>> start_point = np.array([1.8, 1.7])
    >>> search_gradient = np.array([-1.0, -1.0])
    >>> line_search(obj_func, obj_grad, start_point, search_gradient)
    (1.0, 2, 1, 1.1300000000000001, 6.13, [1.6, 1.4])
    """
    domain = args[0];

    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]

    def phi(s):
        fc[0] += 1; # Increment counter

        # Split the vector + apply norm constraints
        XK  = np.split(xk,2); # Must not update xk nor pk here!!, ensure this does not update the reference all the time
        PK  = np.split(pk,2);
        for i in range(len(Constraints)):
            XK[i] = Norm_constraint(domain, XK[i],PK[i], s, Constraints[i])[0]; # As second indice is the negative tangent gradient

        XK_new = np.concatenate((XK[0],XK[1]));   
        return f( XK_new, *args)

    fprime = myfprime

    def derphi(s):
        if s != 0.0:

            gc[0] += 1; # Increment counter

            # Split the vector + apply norm constraints
            XK  = np.split(xk,2); # Must not update xk nor pk here!!, ensure this does not update the reference all the time
            PK  = np.split(pk,2);
            for i in range(len(Constraints)):
                XK[i] = Norm_constraint(domain, XK[i],PK[i], s, Constraints[i])[0]; # As second indice is the negative tangent gradient

            XK_new = np.concatenate((XK[0],XK[1]));    
            gval[0] = fprime( XK_new, *args)
            gval_alpha[0] = s;

            dp_da = (-1.)*XK_new*np.sin(s) + (-1.)*pk*np.cos(s); # hg_k = -1*pk

            DERPHI = Inner_Prod(domain,gval[0],dp_da)
            print('\n\n derphi(s) = ',DERPHI,'\n\n');
            return DERPHI;

        elif float(s) == 0.0:

            dp_da = (-1.)*pk; 
            DERPHI = Inner_Prod(domain,gfk,dp_da)
            print('\n\n derphi0 = ',DERPHI,'\n\n');
            return DERPHI; # hg_k = -1*pk

    if gfk is None:
        gfk = fprime(xk, *args)
    derphi0 = derphi(0.)  

    if extra_condition is not None: #???
        # Add the current gradient as argument, to avoid needless
        # re-evaluation
        def extra_condition2(alpha, phi):
            if gval_alpha[0] != alpha:
                derphi(alpha)
            x = xk + alpha * pk
            return extra_condition(alpha, x, phi, gval[0])
    else:
        extra_condition2 = None

    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,extra_condition2, maxiter=maxiter)

    if derphi_star is None:
        warn('The line search algorithm did not converge', LineSearchWarning)
    else:
        # derphi_star is a number (derphi) -- so use the most recently
        # calculated gradient used in computing it derphi = gfk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derphi_star = gval[0]

    return alpha_star, fc[0], gc[0], phi_star, old_fval, derphi_star


def scalar_search_wolfe2(phi, derphi, phi0=None,old_phi0=None, derphi0=None,c1=1e-4, c2=0.9, amax=None,extra_condition=None, maxiter=10):
    """Find alpha that satisfies strong Wolfe conditions.
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
    #'''
    if old_phi0 is not None and derphi0 != 0:
        alpha1 = min(1.0, 1.01*2*(phi0 - old_phi0)/derphi0)
        #alpha1 = min(amax, 2*(phi0 - old_phi0)/derphi0)
    else:
        alpha1 = 1.0
        #alpha1 = amax

    if alpha1 < 0:
        alpha1 = 1.0
        #alpha1 = amax

    # Should take care of the angle condition on alpha    
    if amax is not None:
        alpha1 = min(alpha1, amax); 
    
    alpha1 = 0.5*amax;
    phi_a1 = phi(alpha1); 
    #derphi_a1 = derphi(alpha1) evaluated below

    alpha_1 = _quadmin(alpha0, phi0, derphi0, alpha1, phi_a1)
    phi_a1 = phi(alpha1); print("alpha1 =",alpha1)

    phi_a0 = phi0
    derphi_a0 = derphi0;

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
            print("Satisfies cond (i)")
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            break

        derphi_a1 = derphi(alpha1)
        if (abs(derphi_a1) <= -c2*derphi0):
            print("Satisfies cond (ii)")
            if extra_condition(alpha1, phi_a1):
                alpha_star = alpha1
                phi_star = phi_a1
                derphi_star = derphi_a1
                break

        if (derphi_a1 >= 0):
            print("Satisfies cond (iii)"); # Correct but I don't understand
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha1, alpha0, phi_a1,
                              phi_a0, derphi_a1, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)
            '''
            alpha_star, phi_star, derphi_star = \
                        _zoom(alpha0, alpha1, phi_a0,
                              phi_a1, derphi_a0, phi, derphi,
                              phi0, derphi0, c1, c2, extra_condition)'''            
            break

        alpha2 = 2 * alpha1  # increase by factor of two on each iteration
        if amax is not None:
            alpha2 = min(alpha2, amax)
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi(alpha1)
        derphi_a0 = derphi_a1; print("alpha =",alpha1)

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


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.
    
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
        print("IN ZOOM WHILE LOOP .... (a_lo,a_hi) = (%e,%e)\n"%(a_lo,a_hi))
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
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha
                print("Just reducing alpha as _quadmin didn't work",a_j)

        # Check new value of a_j

        phi_aj = phi(a_j); print("phi_aj %e > phi0 %e + c1*a_j*derphi0 %e "%(phi_aj,phi0,c1*a_j*derphi0));
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
     