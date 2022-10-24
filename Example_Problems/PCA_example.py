import numpy as np
import sys

'''
Compute the largest principle component of a symetric matrix

min 	J(X)  = −(1/2)*X^T M X,
 X
s.t.    ||X|| = 1, where M = M^T, x^T M x > 0

to run this script:	python3 PCA_example.py
'''

def Hessian_Matrix(DIM):

	"""
	Generates a symmetric & positive definite squre matrix of dimension DIM

	Input:  DIM integer
	Output: M   numpy matrix
	"""
	PD= -1.;
	x = np.random.rand(DIM);

	while PD < 0.: 
		
		M  = np.random.randn(DIM,DIM);
		M  = 0.5*(M + M.T); # Make it symmetric
		PD = np.dot(x, M@x );

	return M;

def Objective_Gradient(xk,*args_f,**kwargs_f):

	"""
	Computes the:

	1) gradient 		  g_k = -\nabla J_k = MX
	2) objective function J_k = -1/2 X^T M X = 1/2 X^T g_k

	Input:  M   numpy matrix
	Input:  X   numpy vector
	Output: J_k,g_k float, numpy vector
	"""

	xk = X[0];

	g_k = -np.matmul(M,xk)
	J_k = (1./2.)*np.dot(xk,g_k)

	if Return_J_only == True:
		return J_k;
	else:	
		return J_k, [g_k];

def Objective(X,*args_f,**kwargs_f):

	"""
	Computes the:

	1) gradient 		  g_k = - \nabla J_k = -MX
	2) objective function J_k = 1/2 X^T M X = - 1/2 X^T g_k

	Input:  M   numpy matrix
	Input:  X   numpy vector
	Output: J_k,g_k float, numpy vector
	"""

	xk = X[0];
	g_k = -np.matmul(M,xk)
	J_k = (1./2.)*np.dot(xk,g_k)

	return J_k;

def Gradient(X,*args_f,**kwargs_f):

	"""
	Computes the:

	1) gradient 		  g_k = - \nabla J_k = -MX
	2) objective function J_k = 1/2 X^T M X = - 1/2 X^T g_k

	Input:  M   numpy matrix
	Input:  X   numpy vector
	Output: J_k,g_k float, numpy vector
	"""

	xk = X[0];

	g_k = -np.matmul(M,xk)
	#J_k = (1./2.)*np.dot(xk,g_k)

	'''
	import math
	theta = math.acos(np.dot(xk,g_k)/(np.linalg.norm(xk,2)*np.linalg.norm(g_k,2) ) )

	print("   ||∂/∂x_k(J_k)|| = ",np.linalg.norm(g_k,2) );
	print("Angle sin(theta_k) = ",np.sin(theta));
	print("Residual  	  r_k = ",np.linalg.norm(g_k,2)*np.sin(theta));
	print("\n");
	'''

	return [g_k];

def Vector_Inner_Product(f,g,*args_IP):

	return np.dot(f,g);

if __name__ == "__main__":

	DIM = 100; 
	M 	= Hessian_Matrix(DIM);
	X_0 = np.random.rand(DIM); 
	M_0 = 1.;
	
	# Positional arguments
	args_f  = (M,True); args_IP = (); #
	
	# Keyword arguments
	#kwargs_f= {"M":M,"Return_J_only":True}; kwargs_IP = {};

	# 1) Calculate the solution via an EVP
	from numpy import linalg as LA
	eigenValues, eigenVectors = LA.eig(M)
	idx = eigenValues.argsort()[::-1]   
	eigenValues = eigenValues[idx]
	eigenVectors = eigenVectors[:,idx]
	v = eigenVectors[:,0]
	print('Eig-vector = ',v,'\n');

	sys.path.insert(0,'../')

	from Sphere_Grad_Descent import Optimise_On_Multi_Sphere, plot_optimisation

	# 2) Calculate the solution via steepest-descent (SD)

	RESIDUAL_SD, FUNCT_SD, x_opt_SD = Optimise_On_Multi_Sphere([X_0],[M_0],Objective,Gradient,Vector_Inner_Product,args_f,args_IP,LS = 'LS_armijo', CG = False,verbose=False);

	print("Error of SD = ",LA.norm(abs(v)-abs(x_opt_SD[0]),2));
	plot_optimisation(RESIDUAL_SD,FUNCT_SD);

	# 3) Calculate the solution via conjugate-gradient (CG)
	
	RESIDUAL_CG, FUNCT_CG, x_opt_CG = Optimise_On_Multi_Sphere([X_0],[M_0],Objective,Gradient,Vector_Inner_Product,args_f,args_IP,LS = 'LS_wolfe', CG = True,verbose=False);
	
	print("Error of CG = ",LA.norm(abs(v)-abs(x_opt_SD[0]),2));
	plot_optimisation(RESIDUAL_CG,FUNCT_CG);
	
	kappa = LA.cond(M)
	print("R^2 = ",pow( (kappa-1.)/(kappa + 1.),2) )


